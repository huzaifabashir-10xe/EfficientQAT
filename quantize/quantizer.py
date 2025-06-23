import torch
import torch.nn as nn

import pdb

# minimum clipping value to avoid division by zero or very small scales
CLIPMIN = 1e-4



def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.

    During the forward pass, returns rounded values.
    During the backward pass, allows gradients to flow as if rounding didn't happen.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Same shape, STE-applied rounding.
    """
    return (x.round() - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    """
    Straight-Through Estimator (STE) for clamping.

    Like round_ste, allows gradient to flow through clamping.

    Args:
        x (Tensor): Input tensor.
        min (float): Minimum clamp value.
        max (float): Maximum clamp value.

    Returns:
        Tensor: STE-clamped tensor.
    """
    return (x.clamp(min,max) - x).detach() + x

def clamp_ste(x: torch.Tensor, min, max):
    return (x.clamp(min,max) - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    """
    Uniform Affine Fake Quantizer for weights using Min-Max initialization.

    This module performs fake quantization during training using learned scale and zero_point.

    Args:
        n_bits (int): Bit-width for quantization (e.g., 4 or 8 bits).
        group_size (int): Size of groups over which scale/zero_point is computed. -1 = per-tensor.
        weight (Tensor): Initial weight tensor used to compute initial scale and zero_point.

    Attributes:
        scale (nn.Parameter): Learnable scale factors per group.
        zero_point (nn.Parameter): Learnable zero points per group.
        qmin (int): Minimum quantized value (usually 0).
        qmax (int): Maximum quantized value (e.g., 15 for 4-bit).
    """
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight=None,
    ):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        # Compute group size: use full dimension if group_size=-1
        self.group_size = group_size if group_size != -1 else weight.shape[-1]
        assert weight.shape[-1] % group_size == 0
        self.enable = True  # Whether fake quant is active
        
        # init scale and zero point through Max-Min quantization
        with torch.no_grad():
            if weight is not None:
                # Reshape weights to [num_groups, group_size]
                x = weight.reshape(-1,self.group_size)
                # Get min max for each group
                xmin = x.amin([-1], keepdim=True)
                xmax =  x.amax([-1], keepdim=True)
                range = xmax - xmin

                # Calculate initial scale and clamp to safe values
                scale = range / (2**self.n_bits-1)
                scale = scale.clamp(min=1e-4, max=1e4)
                # Compute zero point for symmetric affine quant
                zero_point = -(xmin/scale).clamp(min=-1e4, max=1e4) 
                # Register as learnable parameters (used in QAT)
                self.scale = nn.Parameter(scale)
                self.zero_point = nn.Parameter(zero_point.round())
            

    def change_n_bits(self, n_bits):
        """
        Dynamically change the quantization bit-width (e.g., from 8-bit to 4-bit).
        """
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** (n_bits) - 1)
        
    def fake_quant(self, x):
        """
        Simulates quantization (round + clamp) and then dequantizes to float.

        Args:
            x (Tensor): Input tensor to quantize.

        Returns:
            Tensor: Dequantized result using quantization formula.
        """

        # Clamp and round learned scale and zero-point with STE
        scale = clamp_ste(self.scale,1e-4, 1e4)
        round_zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)
        
        # Reshape input to groups of shape [*, group_size]
        dim1, dim2 = x.shape
        x = x.reshape(-1, self.group_size)
        # Simulate quantization
        x_int = round_ste(x / scale)        
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        # Clamp to quantized range
        x_int = x_int.clamp(self.qmin, self.qmax)
        # Simulate dequantization
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        # Reshape back to original
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        """
        Forward pass with optional fake quantization.

        Args:
            x (Tensor): Input tensor (e.g., Linear layer weight).

        Returns:
            Tensor: Quantized + dequantized tensor (fake quantization), or original tensor if disabled.
        """
        if self.n_bits >= 16 or not self.enable:
            return x

        x_dequant = self.fake_quant(x)
        return x_dequant

        

    