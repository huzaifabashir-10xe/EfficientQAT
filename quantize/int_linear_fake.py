import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantizer import UniformAffineQuantizer





class QuantLinear(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.

    This class wraps a standard Linear layer and adds support for fake quantization
    of weights using UniformAffineQuantizer. During quantization-aware training (QAT),
    it allows toggling between using quantized weights and full-precision weights
    using the `set_quant_state` method.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        weight (nn.Parameter): Learnable weights from the original Linear layer.
        bias (Tensor or None): Optional bias term.
        weight_quantizer (UniformAffineQuantizer): Module that performs fake quantization of weights.
        use_weight_quant (bool): Flag indicating whether quantization is active.

    """
    def __init__(
        self,
        org_module: nn.Linear,
        wbits=4,
        group_size=64
    ):
        """
        Initializes the QuantLinear module.

        Args:
            org_module (nn.Linear): A pretrained nn.Linear layer whose weights and bias are reused.
            wbits (int): Bit-width to quantize weights to (e.g., 4 for 4-bit quantization).
            group_size (int): Number of elements per group for group-wise quantization.
        """
        super().__init__()
        self.fwd_kwargs = dict()            # Store arguments for forward logic
        self.fwd_func = F.linear            # default linear function
        self.register_parameter('weight',org_module.weight) # trainable
        
        # Store bias as a non-trainable buffer (still used in forward pass if present)
        if org_module.bias is not None:
            self.register_buffer('bias',org_module.bias)
        else:
            self.bias = None
        
        # Store input/output feature dimensions
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        # de-activate the quantized forward default (normal linear behaviour)
        self.use_weight_quant = False
        # initialize quantizer for weight (but do not quantize yet)
        self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight)
        # Optional flag that can be used for temporary trainable weight usage (not used here)        
        self.use_temporary_parameter = False

    
    
    def forward(self, input: torch.Tensor):
        """
        Forward pass through the quantized (or original) Linear layer.

        Args:
            input (Tensor): The input tensor of shape (batch, seqlen, in_features)

        Returns:
            Tensor: Output after applying (quantized or FP) linear transformation
        """
        if self.use_weight_quant:
            # Apply fake quantization to weights
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            # Use full-precision weights
            weight = self.weight
            bias = self.bias

        # Apply linear transformation (F.linear = input @ weight.T + bias)
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)


        return out

    def set_quant_state(self, weight_quant: bool = False):
        """
        Enables or disables weight quantization.

        Args:
            weight_quant (bool): If True, forward pass will use quantized weights;
                                 if False, it will use full-precision weights.
        """
        self.use_weight_quant = weight_quant




