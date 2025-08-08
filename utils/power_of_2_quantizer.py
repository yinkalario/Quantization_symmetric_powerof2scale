"""
Power-of-2 Scale Quantizer Implementation.

This module implements custom quantizers that constrain scale factors to be powers of 2,
which enables efficient hardware implementation using bit shifts instead of multiplication.

Author: Yin Cao
Date: August 8, 2025
"""

# Standard library imports
from typing import Any, Dict, Tuple

# Third-party imports
import numpy as np
import torch
from torch import nn


class PowerOf2ScaleQuantizer:
    """
    Custom quantizer that constrains scale factors to be powers of 2.

    This enables efficient hardware implementation where multiplication
    becomes bit shifting operations.

    Author: Yin Cao
    """

    def __init__(self, bitwidth: int = 8, symmetric: bool = True):
        """
        Initialize the power-of-2 scale quantizer.

        Args:
            bitwidth: Number of bits for quantization (default: 8)
            symmetric: Whether to use symmetric quantization (default: True)
        """
        self.bitwidth = bitwidth
        self.symmetric = symmetric
        self.quant_min = -(2 ** (bitwidth - 1)) if symmetric else 0
        self.quant_max = 2 ** (bitwidth - 1) - 1 if symmetric else 2 ** bitwidth - 1

    def compute_power_of_2_scale(self, tensor: torch.Tensor) -> Tuple[float, int]:
        """
        Compute power-of-2 scale factor for the given tensor.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Tuple of (scale_factor, zero_point)
        """
        # Find the maximum absolute value
        if self.symmetric:
            max_val = torch.max(torch.abs(tensor)).item()
            min_val = -max_val
            zero_point = 0
        else:
            max_val = torch.max(tensor).item()
            min_val = torch.min(tensor).item()
            zero_point = self.quant_min

        # Compute the range
        value_range = max_val - min_val

        # Find the power of 2 scale that covers the range
        # scale = value_range / (quant_max - quant_min)
        quant_range = self.quant_max - self.quant_min
        ideal_scale = value_range / quant_range

        # Find the nearest power of 2 scale
        # scale = 2^(-n) where n is an integer
        if ideal_scale > 0:
            log2_scale = np.log2(ideal_scale)
            # Round to nearest integer to get power of 2
            n = round(-log2_scale)
            power_of_2_scale = 2.0 ** (-n)
        else:
            power_of_2_scale = 1.0

        return power_of_2_scale, zero_point

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int, int]:
        """
        Quantize tensor and return quantized tensor, scale, zero_point, and exponent.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Tuple of (quantized_tensor, scale, zero_point, exponent)
        """
        scale, zero_point = self.compute_power_of_2_scale(tensor)

        # Compute exponent for power-of-2 scale
        if scale > 0:
            exponent = round(-np.log2(scale))
        else:
            exponent = 0

        # Quantize the tensor
        quantized_tensor = self.quantize_tensor(tensor)

        return quantized_tensor, scale, zero_point, exponent

    def quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor using power-of-2 scale with straight-through estimator.

        This implements fake quantization that allows gradients to flow through
        for quantization-aware training.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Quantized tensor with gradient flow preserved
        """
        scale, zero_point = self.compute_power_of_2_scale(tensor)

        # Quantize: q = round(x / scale) + zero_point
        quantized = tensor / scale + zero_point

        # Apply straight-through estimator for round operation
        # Forward: quantized values, Backward: straight-through gradients
        quantized_round = quantized.round()
        quantized = quantized + (quantized_round - quantized).detach()

        # Apply straight-through estimator for clamp operation
        # Forward: clamped values, Backward: straight-through gradients
        quantized_clamped = quantized.clamp(self.quant_min, self.quant_max)
        quantized = quantized + (quantized_clamped - quantized).detach()

        # Dequantize: x = (q - zero_point) * scale
        dequantized = (quantized - zero_point) * scale

        return dequantized

    def get_scale_info(self, tensor: torch.Tensor) -> dict:
        """
        Get detailed information about the power-of-2 scale computation.

        Args:
            tensor: Input tensor

        Returns:
            Dictionary with scale information
        """
        scale, zero_point = self.compute_power_of_2_scale(tensor)

        # Find the exponent (n where scale = 2^(-n))
        if scale > 0:
            exponent = -int(round(np.log2(scale)))
        else:
            exponent = 0

        return {
            'scale': scale,
            'zero_point': zero_point,
            'exponent': exponent,
            'power_of_2_representation': f"2^({-exponent})",
            'is_power_of_2': abs(scale - 2**(-exponent)) < 1e-10,
            'bitwidth': self.bitwidth,
            'symmetric': self.symmetric,
            'quant_min': self.quant_min,
            'quant_max': self.quant_max
        }


def apply_power_of_2_quantization(
    model: nn.Module,
    sample_input: torch.Tensor,
    bitwidth: int = 8
) -> nn.Module:
    """
    Apply power-of-2 scale quantization to a model.

    Args:
        model: PyTorch model to quantize
        sample_input: Sample input for the model
        bitwidth: Quantization bitwidth

    Returns:
        Model with power-of-2 quantized weights and activations
    """
    quantizer = PowerOf2ScaleQuantizer(bitwidth=bitwidth, symmetric=True)

    # Apply quantization to model parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.shape) > 1:  # Skip bias terms
                quantized_param = quantizer.quantize_tensor(param.data)
                param.data.copy_(quantized_param)

                # Print quantization info for debugging
                info = quantizer.get_scale_info(param.data)
                print(f"Layer {name}: scale={info['scale']:.6f} "
                      f"({info['power_of_2_representation']}), "
                      f"exponent={info['exponent']}")

    return model


def demonstrate_power_of_2_quantization():
    """
    Demonstrate power-of-2 scale quantization with examples.
    """
    print("Power-of-2 Scale Quantization Demonstration")
    print("=" * 50)

    quantizer = PowerOf2ScaleQuantizer(bitwidth=8, symmetric=True)

    # Example tensors with different ranges
    test_tensors = [
        torch.randn(10, 10) * 0.1,    # Small values
        torch.randn(10, 10) * 1.0,    # Medium values
        torch.randn(10, 10) * 10.0,   # Large values
        torch.tensor([0.3, -0.7, 1.2, -1.8, 0.9])  # Specific values
    ]

    for i, tensor in enumerate(test_tensors):
        print(f"\nExample {i+1}:")
        print(f"Original tensor range: [{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

        # Get quantization info
        info = quantizer.get_scale_info(tensor)
        print(f"Power-of-2 scale: {info['scale']:.6f} = {info['power_of_2_representation']}")
        print(f"Exponent: {info['exponent']}")
        print(f"Zero point: {info['zero_point']}")
        print(f"Is exact power of 2: {info['is_power_of_2']}")

        # Show hardware benefit
        if info['exponent'] > 0:
            print(f"Hardware implementation: x >> {info['exponent']} (right shift)")
        elif info['exponent'] < 0:
            print(f"Hardware implementation: x << {-info['exponent']} (left shift)")
        else:
            print("Hardware implementation: x (no shift needed)")

        # Quantize and show error
        quantized = quantizer.quantize_tensor(tensor)
        error = torch.mean(torch.abs(tensor - quantized)).item()
        print(f"Quantization error (MAE): {error:.6f}")


class MultiBitwidthPowerOf2Quantizer:
    """
    Multi-bitwidth power-of-2 quantizer that supports different bitwidths
    for weights, inputs, outputs, and biases based on configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-bitwidth quantizer from configuration.

        Args:
            config: Configuration dictionary with quantization settings
        """
        self.config = config
        quant_config = config['quantization']

        # Create quantizers for different tensor types
        self.weight_quantizer = PowerOf2ScaleQuantizer(
            bitwidth=quant_config['weight']['bitwidth'],
            symmetric=quant_config['weight']['symmetric']
        )

        self.input_quantizer = PowerOf2ScaleQuantizer(
            bitwidth=quant_config['input']['bitwidth'],
            symmetric=quant_config['input']['symmetric']
        )

        self.output_quantizer = PowerOf2ScaleQuantizer(
            bitwidth=quant_config['output']['bitwidth'],
            symmetric=quant_config['output']['symmetric']
        )

        self.bias_quantizer = PowerOf2ScaleQuantizer(
            bitwidth=quant_config['bias']['bitwidth'],
            symmetric=quant_config['bias']['symmetric']
        )

    def quantize_weights(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize weight tensor."""
        quantized, scale, zero_point, exponent = self.weight_quantizer.quantize(tensor)

        info = {
            'scale': float(scale),
            'zero_point': int(zero_point),
            'exponent': int(exponent),
            'power_of_2': (
                f"2^(-{exponent})" if exponent > 0
                else f"2^({-exponent})" if exponent < 0
                else "1"
            ),
            'hardware_op': (
                f"x >> {exponent}" if exponent > 0
                else f"x << {-exponent}" if exponent < 0
                else "x"
            ),
            'bitwidth': self.weight_quantizer.bitwidth
        }

        return quantized, info

    def quantize_inputs(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize input/activation tensor."""
        quantized, scale, zero_point, exponent = self.input_quantizer.quantize(tensor)

        info = {
            'scale': float(scale),
            'zero_point': int(zero_point),
            'exponent': int(exponent),
            'power_of_2': (
                f"2^(-{exponent})" if exponent > 0
                else f"2^({-exponent})" if exponent < 0
                else "1"
            ),
            'hardware_op': (
                f"x >> {exponent}" if exponent > 0
                else f"x << {-exponent}" if exponent < 0
                else "x"
            ),
            'bitwidth': self.input_quantizer.bitwidth
        }

        return quantized, info

    def quantize_outputs(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize output tensor."""
        quantized, scale, zero_point, exponent = self.output_quantizer.quantize(tensor)

        info = {
            'scale': float(scale),
            'zero_point': int(zero_point),
            'exponent': int(exponent),
            'power_of_2': (
                f"2^(-{exponent})" if exponent > 0
                else f"2^({-exponent})" if exponent < 0
                else "1"
            ),
            'hardware_op': (
                f"x >> {exponent}" if exponent > 0
                else f"x << {-exponent}" if exponent < 0
                else "x"
            ),
            'bitwidth': self.output_quantizer.bitwidth
        }

        return quantized, info

    def quantize_biases(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize bias tensor."""
        quantized, scale, zero_point, exponent = self.bias_quantizer.quantize(tensor)

        info = {
            'scale': float(scale),
            'zero_point': int(zero_point),
            'exponent': int(exponent),
            'power_of_2': (
                f"2^(-{exponent})" if exponent > 0
                else f"2^({-exponent})" if exponent < 0
                else "1"
            ),
            'hardware_op': (
                f"x >> {exponent}" if exponent > 0
                else f"x << {-exponent}" if exponent < 0
                else "x"
            ),
            'bitwidth': self.bias_quantizer.bitwidth
        }

        return quantized, info


if __name__ == "__main__":
    demonstrate_power_of_2_quantization()
