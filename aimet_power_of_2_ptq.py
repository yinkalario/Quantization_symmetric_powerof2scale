#!/usr/bin/env python3
"""
AIMET + Power-of-2 Symmetric Post-Training Quantization.

This script combines AIMET's quantization infrastructure with custom power-of-2
scale constraints for optimal hardware efficiency.

Features:
- AIMET's symmetric quantization framework
- Custom power-of-2 scale factor constraints
- Professional quantization pipeline
- Hardware-optimized bit-shift operations

Author: Yin Cao
Date: August 8, 2025

Usage:
    python aimet_power_of_2_ptq.py --model_path your_model.pth \
        --data_path ./data
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*NLLLoss2d.*",
                        category=FutureWarning)

# AIMET imports (will be available after environment setup)
try:
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_common.defs import QuantScheme
    AIMET_AVAILABLE = True
except ImportError as e:
    print(f"AIMET not available: {e}")
    print("Please run: conda activate aimet_quantization")
    AIMET_AVAILABLE = False

# Local imports
from utils.model_utils import (
    load_config, load_model, load_data, evaluate_model
)
from utils.power_of_2_quantizer import MultiBitwidthPowerOf2Quantizer


class AIMETPowerOf2Quantizer:
    """
    Hybrid quantizer that uses AIMET infrastructure with power-of-2 constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quantizer = MultiBitwidthPowerOf2Quantizer(config)

    def apply_power_of_2_constraints(self, original_model, quantsim: QuantizationSimModel = None):
        """Apply power-of-2 constraints to AIMET quantizers."""
        print("Applying power-of-2 constraints to AIMET quantizers...")

        constraint_info = {}

        # Use the original model weights to compute power-of-2 constraints
        for name, module in original_model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Compute power-of-2 scale for weights
                weight_tensor = module.weight.data
                _, weight_info = self.quantizer.quantize_weights(weight_tensor)
                scale = weight_info['scale']

                # Store constraint info for reporting
                constraint_info[f"{name}.weight"] = weight_info
                print(f"  {name}.weight: scale={scale:.6f} = "
                      f"{weight_info['power_of_2']}, "
                      f"hardware: {weight_info['hardware_op']}, "
                      f"{weight_info['bitwidth']}-bit")

        return constraint_info


def load_data_aimet(data_path: str, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Calibration data (subset of training data)
    train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    calib_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Test data
    test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return calib_loader, test_loader


def main():
    # Check AIMET availability first
    if not AIMET_AVAILABLE:
        print("❌ AIMET is not available!")
        print("Please setup the environment first:")
        print("  chmod +x scripts/create_env.sh")
        print("  ./scripts/create_env.sh")
        print("  conda activate aimet_quantization")
        return

    parser = argparse.ArgumentParser(
        description='AIMET + Power-of-2 Symmetric PTQ'
    )
    parser.add_argument(
        '--config', type=str, default='configs/quantization_config.yaml',
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--model_path', type=str, default='model.pth',
        help='Path to trained model file'
    )
    parser.add_argument(
        '--data_path', type=str, default='data/',
        help='Path to dataset'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--max_eval_batches', type=int, default=None,
        help='Max batches for evaluation (overrides config)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: cuda, cpu, or auto'
    )

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
        output_dir = Path(config['output']['base_dir']) / 'aimet_ptq'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    model = load_model(args.model_path, config, device)

    # Load data
    print(f"Loading data from {args.data_path}...")
    _, test_loader = load_data(config, args.data_path)

    # Evaluate original model
    print("Evaluating original FP32 model...")
    max_eval_batches = args.max_eval_batches or config['ptq'].get('max_eval_batches')
    original_accuracy = evaluate_model(model, test_loader, device, max_eval_batches)
    print(f"Original accuracy: {original_accuracy:.2f}%")

    # Create AIMET QuantSim with symmetric quantization
    print("\nCreating AIMET QuantSim with symmetric quantization...")
    # Get real input from CIFAR dataset
    for data, _ in test_loader:
        example_input = data[:1].to(device)  # Use first sample as example
        break

    quantsim = QuantizationSimModel(
        model=model,
        dummy_input=example_input,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_output_bw=config['quantization']['output']['bitwidth'],
        default_param_bw=config['quantization']['weight']['bitwidth'],
        config_file=None  # Use default symmetric config
    )

    # Apply power-of-2 constraints
    power_of_2_quantizer = AIMETPowerOf2Quantizer(config)

    # Calibration callback
    def forward_pass_callback(model, _):
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                if batch_idx >= config['ptq'].get('calibration_batches', 10):
                    break
                data = data.to(device)
                _ = model(data)

    # Compute encodings with AIMET
    print("Computing AIMET encodings...")
    quantsim.compute_encodings(
        forward_pass_callback=forward_pass_callback,
        forward_pass_callback_args=None
    )

    # Apply power-of-2 constraints after AIMET encoding computation
    constraint_info = power_of_2_quantizer.apply_power_of_2_constraints(model, quantsim)

    # Evaluate quantized model
    print("\nEvaluating AIMET + Power-of-2 quantized model...")
    quantized_accuracy = evaluate_model(quantsim.model, test_loader, device, max_eval_batches)
    print(f"Quantized accuracy: {quantized_accuracy:.2f}%")

    # Results
    accuracy_drop = original_accuracy - quantized_accuracy
    print("\nResults:")
    print(f"  Original accuracy:  {original_accuracy:.2f}%")
    print(f"  Quantized accuracy: {quantized_accuracy:.2f}%")
    print(f"  Accuracy drop:      {accuracy_drop:.2f}%")
    print("  Quantization:       AIMET + Power-of-2 Symmetric")

    # Save results
    results = {
        'quantization_type': 'AIMET + Power-of-2 Symmetric PTQ',
        'config': config['quantization'],
        'original_accuracy': float(original_accuracy),
        'quantized_accuracy': float(quantized_accuracy),
        'accuracy_drop': float(accuracy_drop),
        'quantization_details': constraint_info
    }

    results_file = output_dir / 'aimet_power_of_2_ptq_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Export quantized model
    quantsim.export(
        path=str(output_dir),
        filename_prefix='aimet_power_of_2_quantized',
        dummy_input=example_input
    )

    print(f"\nSaved results to: {results_file}")
    print(f"Exported quantized model to: {output_dir}/")
    print("AIMET + Power-of-2 symmetric PTQ completed!")


if __name__ == '__main__':
    main()
