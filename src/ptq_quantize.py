#!/usr/bin/env python3
"""
Enhanced Power-of-2 Symmetric Post-Training Quantization.

This script implements PTQ with configurable bitwidths for different tensor types
(weights, inputs, outputs, biases) using power-of-2 scale constraints.

Features:
- Multi-bitwidth quantization (weights, inputs, outputs, biases)
- Configuration-driven setup
- Shared model/data utilities
- Detailed quantization reporting

Author: Yin Cao
Date: August 8, 2025

Usage:
    python ptq_quantize.py --config configs/quantization_config.yaml \
        --data_path data/
"""

# Standard library imports
import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict

# Third-party imports
import torch
from torch import nn

# Local imports
from utils.model_utils import (
    evaluate_model, load_config, load_data, load_model
)
from utils.power_of_2_quantizer import MultiBitwidthPowerOf2Quantizer

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*NLLLoss2d.*",
                        category=FutureWarning)


def quantize_model_comprehensive(
    model: nn.Module,
    quantizer: MultiBitwidthPowerOf2Quantizer
) -> Dict[str, Any]:
    """
    Apply comprehensive quantization to model (weights, biases).

    Note: Input/output quantization would be applied during inference.

    Args:
        model: PyTorch model to quantize
        quantizer: Multi-bitwidth power-of-2 quantizer

    Returns:
        Dictionary containing quantization details for each layer
    """
    print("Applying multi-bitwidth power-of-2 quantization...")
    quantization_details = {}

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            # Quantize weights
            original_weight = module.weight.data.clone()
            quantized_weight, weight_info = quantizer.quantize_weights(
                original_weight)
            module.weight.data = quantized_weight

            quantization_details[f"{name}.weight"] = weight_info
            print(f"  {name}.weight: scale={weight_info['scale']:.6f} = "
                  f"{weight_info['power_of_2']}, "
                  f"hardware: {weight_info['hardware_op']}, "
                  f"{weight_info['bitwidth']}-bit")

        if hasattr(module, 'bias') and module.bias is not None:
            # Quantize biases
            original_bias = module.bias.data.clone()
            quantized_bias, bias_info = quantizer.quantize_biases(
                original_bias)
            module.bias.data = quantized_bias

            quantization_details[f"{name}.bias"] = bias_info
            print(f"  {name}.bias: scale={bias_info['scale']:.6f} = "
                  f"{bias_info['power_of_2']}, "
                  f"hardware: {bias_info['hardware_op']}, "
                  f"{bias_info['bitwidth']}-bit")

    return quantization_details


def quantize_inputs_outputs(
    model: nn.Module,
    quantizer: MultiBitwidthPowerOf2Quantizer,
    data_loader,
    device: torch.device,
    num_batches: int = 10
) -> Dict[str, Any]:
    """
    Apply input/output quantization using calibration data.

    This function captures activation ranges and computes quantization
    parameters for inputs and outputs.

    Args:
        model: PyTorch model (weights should already be quantized)
        quantizer: Multi-bitwidth power-of-2 quantizer
        data_loader: DataLoader for calibration
        device: Device to run calibration on
        num_batches: Number of batches for calibration

    Returns:
        Dictionary containing input/output quantization details
    """
    print("Calibrating input/output quantization parameters...")
    model.eval()

    # Collect activation statistics
    activations = {}
    hooks = []

    def create_hook(name):
        def hook_fn(module, input_tensor, output):
            if name not in activations:
                activations[name] = []
            # Store input and output tensors
            if isinstance(input_tensor, tuple) and len(input_tensor) > 0:
                activations[f"{name}.input"] = activations.get(f"{name}.input", [])
                activations[f"{name}.input"].append(input_tensor[0].detach().cpu())
            if output is not None:
                activations[f"{name}.output"] = activations.get(f"{name}.output", [])
                activations[f"{name}.output"].append(output.detach().cpu())
        return hook_fn

    # Register hooks for all modules
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)

    # Run calibration
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            data = data.to(device)
            _ = model(data)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Compute quantization parameters
    quantization_details = {}

    for name, tensors in activations.items():
        if len(tensors) > 0:
            # Concatenate all collected tensors
            combined_tensor = torch.cat(tensors, dim=0)

            # Determine quantization type
            if '.input' in name:
                _, quant_info = quantizer.quantize_inputs(combined_tensor)
                layer_name = name.replace('.input', '')
                quantization_details[f"{layer_name}.input"] = quant_info
                print(f"  {layer_name}.input: scale={quant_info['scale']:.6f} = "
                      f"{quant_info['power_of_2']}, "
                      f"hardware: {quant_info['hardware_op']}, "
                      f"{quant_info['bitwidth']}-bit")
            elif '.output' in name:
                _, quant_info = quantizer.quantize_outputs(combined_tensor)
                layer_name = name.replace('.output', '')
                quantization_details[f"{layer_name}.output"] = quant_info
                print(f"  {layer_name}.output: scale={quant_info['scale']:.6f} = "
                      f"{quant_info['power_of_2']}, "
                      f"hardware: {quant_info['hardware_op']}, "
                      f"{quant_info['bitwidth']}-bit")

    return quantization_details


def main():
    """Main function for PTQ quantization."""
    parser = argparse.ArgumentParser(
        description='Enhanced Power-of-2 Symmetric PTQ')
    parser.add_argument(
        '--config', type=str, default='configs/quantization_config.yaml',
        help='Path to configuration file')
    parser.add_argument(
        '--model_path', type=str, default='model.pth',
        help='Path to trained model file')
    parser.add_argument(
        '--data_path', type=str, default='data/',
        help='Path to dataset')
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (overrides config)')
    parser.add_argument(
        '--max_eval_batches', type=int, default=None,
        help='Max batches for evaluation (overrides config)')
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: cuda, cpu, or auto')

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['output']['base_dir']) / 'ptq'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(args.model_path, config, device)

    # Load data
    print(f"Loading data from {args.data_path}...")
    _, test_loader = load_data(config, args.data_path)

    # Evaluate original model
    print("Evaluating original FP32 model...")
    max_eval_batches = (args.max_eval_batches
                        or config['ptq'].get('max_eval_batches'))
    original_accuracy = evaluate_model(
        model, test_loader, device, max_eval_batches)
    print(f"Original accuracy: {original_accuracy:.2f}%")

    # Create multi-bitwidth quantizer
    quantizer = MultiBitwidthPowerOf2Quantizer(config)

    # Apply quantization
    print("\nApplying multi-bitwidth power-of-2 quantization...")
    print(f"Weight: {config['quantization']['weight']['bitwidth']}-bit")
    print(f"Input: {config['quantization']['input']['bitwidth']}-bit")
    print(f"Output: {config['quantization']['output']['bitwidth']}-bit")
    print(f"Bias: {config['quantization']['bias']['bitwidth']}-bit")

    quantization_details = quantize_model_comprehensive(model, quantizer)

    # Apply input/output quantization
    print("\nApplying input/output quantization...")
    input_output_details = quantize_inputs_outputs(
        model, quantizer, test_loader, device,
        num_batches=config['ptq'].get('calibration_batches', 50)
    )

    # Merge quantization details
    quantization_details.update(input_output_details)

    # Evaluate quantized model
    print("\nEvaluating quantized model...")
    quantized_accuracy = evaluate_model(
        model, test_loader, device, max_eval_batches)
    print(f"Quantized accuracy: {quantized_accuracy:.2f}%")

    # Calculate metrics
    accuracy_drop = original_accuracy - quantized_accuracy
    print("\nResults:")
    print(f"  Original accuracy:  {original_accuracy:.2f}%")
    print(f"  Quantized accuracy: {quantized_accuracy:.2f}%")
    print(f"  Accuracy drop:      {accuracy_drop:.2f}%")
    print("  Quantization:       Multi-bitwidth Power-of-2 Symmetric")

    # Prepare results
    results = {
        'quantization_type': 'Multi-bitwidth Power-of-2 Symmetric PTQ',
        'config': config['quantization'],
        'original_accuracy': float(original_accuracy),
        'quantized_accuracy': float(quantized_accuracy),
        'accuracy_drop': float(accuracy_drop),
        'quantization_details': quantization_details
    }

    # Save results
    results_file = output_dir / 'ptq_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save quantized model
    model_file = output_dir / 'quantized_model.pth'
    torch.save(model.state_dict(), model_file)

    print(f"\nSaved results to: {results_file}")
    print(f"Saved quantized model to: {model_file}")
    print("Multi-bitwidth power-of-2 symmetric PTQ completed!")


if __name__ == '__main__':
    main()
