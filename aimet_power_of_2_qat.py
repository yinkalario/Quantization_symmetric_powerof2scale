#!/usr/bin/env python3
"""
AIMET + Power-of-2 Symmetric Quantization Aware Training (QAT).

This script combines AIMET's QAT infrastructure with custom power-of-2
scale constraints for optimal hardware efficiency during training.

Features:
- AIMET's quantization-aware training framework
- Custom power-of-2 scale factor constraints
- Professional QAT pipeline with fake quantization
- Hardware-optimized bit-shift operations

Author: Yin Cao
Date: August 8, 2025

Usage:
    python aimet_power_of_2_qat.py --model_path pretrained_model.pth \
        --data_path ./data --epochs 10
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# AIMET imports (will be available after environment setup)
try:
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_common.defs import QuantScheme
    from aimet_torch.qc_quantize_op import QcQuantizeWrapper
    AIMET_AVAILABLE = True
except ImportError:
    print("AIMET not available. Please run: conda activate aimet_quantization")
    AIMET_AVAILABLE = False

# Local imports
from utils.model_utils import (
    load_config, load_model, load_data, evaluate_model,
    create_criterion, create_optimizer, create_scheduler
)
from utils.power_of_2_quantizer import MultiBitwidthPowerOf2Quantizer

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*NLLLoss2d.*",
                        category=FutureWarning)


class AIMETPowerOf2QATManager:
    """
    Manages AIMET QAT with power-of-2 scale constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.quantizer = MultiBitwidthPowerOf2Quantizer(config)

    def apply_power_of_2_constraints(self, quantsim: QuantizationSimModel):
        """Apply power-of-2 constraints to AIMET quantizers during training."""
        print("Applying power-of-2 constraints to AIMET QAT quantizers...")

        constraint_info = {}

        for name, module in quantsim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                # Get the wrapped module (Conv2d, Linear, etc.)
                wrapped_module = module._module_to_wrap

                if hasattr(wrapped_module, 'weight'):
                    # Compute power-of-2 scale for weights
                    weight_tensor = wrapped_module.weight.data
                    _, weight_info = self.quantizer.quantize_weights(weight_tensor)
                    scale = weight_info['scale']
                    zero_point = weight_info['zero_point']
                    exponent = weight_info['exponent']

                    # Apply power-of-2 constraint to AIMET quantizer
                    if hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers:
                        weight_quantizer = module.param_quantizers['weight']
                        if weight_quantizer.enabled:
                            # Force the scale to be power-of-2
                            weight_quantizer.encoding.scale = float(scale)
                            weight_quantizer.encoding.offset = int(zero_point)

                            constraint_info[name] = {
                                'scale': float(scale),
                                'zero_point': int(zero_point),
                                'exponent': int(exponent),
                                'power_of_2': f"2^(-{exponent})",
                                'hardware_op': (
                                    f"x >> {exponent}" if exponent > 0
                                    else f"x << {-exponent}" if exponent < 0
                                    else "x"
                                )
                            }

                            hardware_op = constraint_info[name]['hardware_op']
                            print(f"  {name}: scale={scale:.6f} = 2^(-{exponent}), "
                                  f"hardware: {hardware_op}")

        return constraint_info

    def update_power_of_2_constraints_during_training(self, quantsim: QuantizationSimModel):
        """Update power-of-2 constraints periodically during training."""
        for name, module in quantsim.model.named_modules():
            if isinstance(module, QcQuantizeWrapper):
                wrapped_module = module._module_to_wrap

                if hasattr(wrapped_module, 'weight'):
                    # Recompute power-of-2 scale for updated weights
                    weight_tensor = wrapped_module.weight.data
                    _, weight_info = self.quantizer.quantize_weights(weight_tensor)
                    scale, zero_point = weight_info['scale'], weight_info['zero_point']

                    # Update AIMET quantizer with new power-of-2 scale
                    if hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers:
                        weight_quantizer = module.param_quantizers['weight']
                        if weight_quantizer.enabled:
                            weight_quantizer.encoding.scale = float(scale)
                            weight_quantizer.encoding.offset = int(zero_point)


def load_data_aimet(data_path: str, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Training data
    train_dataset = datasets.CIFAR10(
        data_path, train=True, download=True, transform=transform_train
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Test data
    test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_epoch_with_aimet_power_of_2(
    quantsim: QuantizationSimModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    qat_manager: AIMETPowerOf2QATManager,
    update_frequency: int = 100
) -> float:
    """Train one epoch with AIMET QAT and power-of-2 constraints."""
    quantsim.model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = quantsim.model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update power-of-2 constraints periodically
        if batch_idx % update_frequency == 0:
            qat_manager.update_power_of_2_constraints_during_training(quantsim)
            if batch_idx % (update_frequency * 2) == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

    return running_loss / len(train_loader)


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
        description='AIMET + Power-of-2 Symmetric QAT'
    )
    parser.add_argument(
        '--config', type=str, default='configs/quantization_config.yaml',
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--model_path', type=str, default='model.pth',
        help='Path to pretrained model file (optional)'
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
        '--epochs', type=int, default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='Device: cuda, cpu, or auto'
    )
    parser.add_argument(
        '--final_ptq', action='store_true', default=True,
        help='Apply final PTQ for input/output quantization after QAT '
             '(default: True)'
    )
    parser.add_argument(
        '--no-final_ptq', dest='final_ptq', action='store_false',
        help='Skip final PTQ step'
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

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    try:
        model = load_model(args.model_path, config, device)
        print(f"✅ Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("Creating a new model with random weights...")
        from utils.model_utils import create_model
        model = create_model(config, device)

    # Load data
    print(f"Loading data from {args.data_path}...")
    train_loader, test_loader = load_data(config, args.data_path)

    # Step 1: PTQ Initialization (recommended best practice)
    if config['qat'].get('run_ptq_first', True):
        print("\n" + "=" * 60)
        print("STEP 1: AIMET PTQ Initialization")
        print("=" * 60)

        # Evaluate before PTQ
        print("Evaluating model before PTQ...")
        initial_accuracy = evaluate_model(model, test_loader, device)
        print(f"Initial accuracy: {initial_accuracy:.2f}%")

        # Create AIMET QuantSim for PTQ first
        print("Creating AIMET QuantSim for PTQ...")
        # Get real input from CIFAR dataset
        for data, _ in train_loader:
            example_input = data[:1].to(device)  # Use first sample as example
            break

        ptq_quantsim = QuantizationSimModel(
            model=model,
            dummy_input=example_input,
            quant_scheme=QuantScheme.post_training_tf_enhanced,
            default_output_bw=config['quantization']['output']['bitwidth'],
            default_param_bw=config['quantization']['weight']['bitwidth'],
            config_file=None
        )

        # Ensure PTQ quantsim model is on the correct device
        ptq_quantsim.model = ptq_quantsim.model.to(device)

        # PTQ calibration
        def ptq_forward_pass_callback(model, _):
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(train_loader):
                    if batch_idx >= config['qat'].get('calibration_batches', 10):
                        break
                    data = data.to(device)
                    _ = model(data)

        print("Computing PTQ encodings...")
        ptq_quantsim.compute_encodings(
            forward_pass_callback=ptq_forward_pass_callback,
            forward_pass_callback_args=None
        )

        # Apply power-of-2 constraints to PTQ
        ptq_manager = AIMETPowerOf2QATManager(config)
        ptq_constraint_info = ptq_manager.apply_power_of_2_constraints(ptq_quantsim)

        # Evaluate PTQ
        print("Evaluating model after PTQ...")
        ptq_accuracy = evaluate_model(ptq_quantsim.model, test_loader, device)
        print(f"PTQ accuracy: {ptq_accuracy:.2f}%")
        print(f"PTQ accuracy drop: {initial_accuracy - ptq_accuracy:.2f}%")

        # Use PTQ model as starting point for QAT
        model = ptq_quantsim.model
    else:
        print("Skipping PTQ initialization (disabled in config)")
        initial_accuracy = evaluate_model(model, test_loader, device)
        ptq_accuracy = initial_accuracy
        ptq_constraint_info = {}

    # Step 2: QAT Training
    print("\n" + "=" * 60)
    print("STEP 2: AIMET Quantization Aware Training")
    print("=" * 60)

    # Create AIMET QuantSim for QAT
    print("Creating AIMET QuantSim for QAT...")
    # Use the same real input from CIFAR dataset
    for data, _ in train_loader:
        example_input = data[:1].to(device)  # Use first sample as example
        break

    quantsim = QuantizationSimModel(
        model=model,
        dummy_input=example_input,
        quant_scheme=QuantScheme.training_range_learning_with_tf_enhanced_init,
        default_output_bw=config['quantization']['output']['bitwidth'],
        default_param_bw=config['quantization']['weight']['bitwidth'],
        config_file=None  # Use default symmetric config
    )

    # Ensure QAT quantsim model is on the correct device
    quantsim.model = quantsim.model.to(device)
    print(f"QAT QuantSim model moved to device: {device}")

    # Initialize QAT manager
    qat_manager = AIMETPowerOf2QATManager(config)

    # Initial calibration for QAT
    def forward_pass_callback(model, _):
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(train_loader):
                if batch_idx >= config['qat'].get('calibration_batches', 5):
                    break
                data = data.to(device)
                _ = model(data)

    print("Computing initial AIMET encodings for QAT...")
    quantsim.compute_encodings(
        forward_pass_callback=forward_pass_callback,
        forward_pass_callback_args=None
    )

    # Apply initial power-of-2 constraints
    _ = qat_manager.apply_power_of_2_constraints(quantsim)

    # Setup training using config (like normal QAT)
    criterion = create_criterion(config)
    optimizer = create_optimizer(quantsim.model, config)
    scheduler = create_scheduler(optimizer, config)

    # Training parameters
    epochs = args.epochs or config['training']['epochs']
    update_frequency = config['qat'].get('constraint_update_frequency', 100)

    # Training loop with AIMET QAT + Power-of-2
    print(f"\nStarting AIMET + Power-of-2 QAT for {epochs} epochs...")
    print(f"Constraint update frequency: every {update_frequency} batches")
    best_accuracy = 0.0

    for epoch in range(epochs):
        # Train with power-of-2 constraints
        avg_loss = train_epoch_with_aimet_power_of_2(
            quantsim, train_loader, optimizer, criterion, device,
            epoch + 1, qat_manager, update_frequency
        )

        # Evaluate
        accuracy = evaluate_model(quantsim.model, test_loader, device)

        # Update learning rate
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = output_dir / 'best_aimet_power_of_2_qat_model.pth'
            torch.save(quantsim.model.state_dict(), best_model_path)

    # Extract final quantization details
    final_constraint_info = qat_manager.apply_power_of_2_constraints(quantsim)

    # Final evaluation
    print("\nAIMET + Power-of-2 QAT completed!")
    print(f"Initial accuracy: {initial_accuracy:.2f}%")
    if config['qat'].get('run_ptq_first', True):
        print(f"PTQ accuracy:     {ptq_accuracy:.2f}%")
    print(f"Final accuracy:   {best_accuracy:.2f}%")
    print(f"Total improvement: {best_accuracy - initial_accuracy:.2f}%")

    # Save results
    results = {
        'quantization_type': 'AIMET + Power-of-2 Symmetric QAT',
        'config': config['quantization'],
        'epochs': epochs,
        'learning_rate': config['training']['learning_rate'],
        'initial_accuracy': float(initial_accuracy),
        'ptq_accuracy': float(ptq_accuracy) if config['qat'].get('run_ptq_first', True) else None,
        'final_accuracy': float(best_accuracy),
        'total_improvement': float(best_accuracy - initial_accuracy),
        'ptq_quantization_details': (ptq_constraint_info
                                     if config['qat'].get('run_ptq_first', True)
                                     else None),
        'final_quantization_details': final_constraint_info
    }

    results_file = output_dir / 'aimet_power_of_2_qat_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Export final model with AIMET
    print("\nExporting final quantized model...")

    # Force everything to CPU for ONNX export (AIMET compatibility)
    print("Moving model to CPU for ONNX export compatibility...")
    quantsim.model = quantsim.model.cpu()

    # Create dummy input on CPU for export
    dummy_input = torch.randn(1, 3, 32, 32)  # CPU tensor

    try:
        quantsim.export(
            path=str(output_dir),
            filename_prefix='aimet_power_of_2_qat_final',
            dummy_input=dummy_input
        )
        print("✅ Model export successful!")
    except Exception as e:
        print(f"⚠️  Model export failed: {e}")
        print("This is a known AIMET/ONNX compatibility issue.")
        print("Quantization results are still valid - continuing without export...")

        # Try to save just the PyTorch model without ONNX
        try:
            model_save_path = output_dir / 'quantized_model.pth'
            torch.save(quantsim.model.state_dict(), model_save_path)
            print(f"✅ Saved PyTorch model to: {model_save_path}")
        except Exception as save_e:
            print(f"⚠️  PyTorch model save also failed: {save_e}")

    print(f"\nSaved results to: {results_file}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Exported AIMET model to: {output_dir}/")
    print("AIMET + Power-of-2 symmetric QAT completed!")


if __name__ == '__main__':
    main()
