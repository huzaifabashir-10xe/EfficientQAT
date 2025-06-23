import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from quantize.triton_utils.kernels import dequant_dim0, dequant_dim1
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from tqdm import tqdm
import gc  
from quantize.utils import get_named_linears,set_op_by_name

logger = getLogger(__name__)


class TritonModuleMixin:
    @classmethod
    # Placeholder for Triton GPU kernel warm-up (if needed)
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class QuantLinear(nn.Module, TritonModuleMixin):
    """
    Implements a low-bit real quantized linear layer using packed integer weights.

    This is used during inference after training with fake quantization.
    The weights are packed into INT2/INT3/INT4/INT8 format with scale and zero-point per group.

    Args:
        bits (int): Number of quantization bits (2, 3, 4, 8).
        group_size (int): Group size for per-group quantization.
        infeatures (int): Number of input features.
        outfeatures (int): Number of output features.
        bias (bool): Whether to include a bias term.
        trainable (bool): If True, enables gradient flow through scales (default: False).
    """

    QUANT_TYPE = "triton"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        **kwargs
    ):
        super().__init__()
        # if bits not in [2, 4, 8]:
        #     raise NotImplementedError("Only 2,4,8 bits are supported.")
        # if infeatures % 32 != 0 or outfeatures % 32 != 0:
        #     raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1          # e.g., 15 for 4-bit
        
        # qweight: packed low-bit integer weights stored as int32
        self.register_buffer(
            'qweight',
            torch.zeros((math.ceil(infeatures / (32 // self.bits)), outfeatures), dtype=torch.int32)
        )

        # scales: per-group scaling factors (learnable if trainable=True)
        self.register_parameter(
            'scales',
            torch.nn.Parameter(torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16))
        )

        # qzeros: packed zero-points
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(infeatures / self.group_size), math.ceil(outfeatures / (32 // self.bits))), dtype=torch.int32)
        )

        # g_idx: group index per input channel (for compatibility with GPTQ-style APIs)
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )   # not used, just for consistent with GPTQ models
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.use_fake = False           # if True, uses dequantized weights for faster training

    def post_init(self):
        #Placeholder if you want to run something after installation
        pass


    def use_fake_quantization(self, del_quant=False,transpose=False):
        """
        Enables use of dequantized float weights (i.e., fake quantization).
        This is useful during QAT or inspection. Use fake quantization for faster training but consume more memory

        Args:
            del_quant (bool): If True, deletes qweight/qzeros to save memory.
            transpose (bool): If True, transposes the dequantized weight.
        """

        # Dequantize integer weights and zero-points
        weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
         # Apply dequantization formula: w = (q - z) * s
        dim0, dim1 = weight.shape
        zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
        weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0,1).contiguous()
        self.register_buffer(
            'weight',
            weight
        )
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.qzeros
            del self.g_idx
        
    def pack(self, linear, scales, zeros, g_idx=None):
        """
        Compresses a full-precision Linear layer into low-bit format (quantizes & packs weights).

        Args:
            linear (nn.Linear): Original linear layer to quantize.
            scales (Tensor): Scaling factors per group.
            zeros (Tensor): Zero points per group.
            g_idx (Tensor, optional): Group index (unused, recomputed).
        """
        
        W = linear.weight.data.clone()
        # Flatten or transpose based on original module type
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()
    
        g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.int32)

        scale_zeros = zeros * scales
        self.scales = nn.Parameter(scales.half())
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        # Quantize weights to integers
        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round(
                    (
                        W[:, idx] + scale_zeros[g_idx[idx]]) / self.scales[g_idx[idx]]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        # Pack quantized weights (bit-packing)
        i = 0
        row = 0
        qweight = np.zeros((math.ceil(intweight.shape[0]/(32//self.bits)), intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros = zeros.numpy().astype(np.uint32)
        self.zeros_dim0, self.zeros_dim1 = zeros.shape
        qzeros = np.zeros((zeros.shape[0], math.ceil(zeros.shape[1] / (32 // self.bits))), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 3, 4, 8]:
                for j in range(i, min(i + (32 // self.bits), zeros.shape[1])):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        """
        Performs matrix multiplication with the quantized (or fake) weights.

        Args:
            x (Tensor): Input tensor of shape [batch, in_features]

        Returns:
            Tensor: Output tensor [batch, out_features]
        """
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0,1)
        else:
            # Dequantize weights from INT to FP on the fly
            weight = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
            dim0, dim1 = weight.shape
            # dim2 = (dim1*dim0)//self.group_size
            zeros = dequant_dim1(self.qzeros, self.bits, self.maxq, self.zeros_dim0, self.zeros_dim1)
            weight = ((weight.view(-1, self.group_size, dim1) - zeros.view(-1, 1, dim1)) * self.scales.view(-1, 1, dim1)).reshape(dim0, dim1)
        # out = torch.matmul(x, weight)
        out = torch.matmul(x, weight.to(x.dtype))
        out = out + self.bias if self.bias is not None else out
        return out


def load_quantized_model(model_path, wbits, group_size):
    """
    Loads a pre-trained quantized model from disk, replaces all nn.Linear layers
    with QuantLinear, and loads packed weight checkpoint.

    Args:
        model_path (str): Path to saved quantized model directory.
        wbits (int): Bit width (2/3/4/8).
        group_size (int): Quantization group size.

    Returns:
        Tuple: (Quantized model, tokenizer)
    """

    print(f"Loading quantized model from {model_path}")

    # import pdb;pdb.set_trace()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    config = AutoConfig.from_pretrained(model_path)
    # Initialize empty model with correct architecture
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config,torch_dtype=torch.float16, trust_remote_code=True)
    # Replace all nn.Linear layers with QuantLinear
    layers = model.model.layers
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer, torch.nn.Linear)
        for name, module in named_linears.items():
            q_linear = QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    
    # Load pre-packed quantized weights into QuantLinear modules
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")

    return model, tokenizer

__all__ = ["QuantLinear","load_omniq_quantized"]
