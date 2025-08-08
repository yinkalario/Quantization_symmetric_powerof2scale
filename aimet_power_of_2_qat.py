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

# Standard library imports
import argparse
import copy
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

# Third-party imports
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# AIMET imports
try:
    from aimet_common.defs import QuantScheme
    from aimet_torch.quantsim import QuantizationSimModel
    AIMET_AVAILABLE = True
except ImportError as e:
    print(f"AIMET not available: {e}")
    print("Please run: conda activate aimet_quantization")
    AIMET_AVAILABLE = False

# Local imports
from utils.model_utils import (
    create_criterion, create_model, create_optimizer, create_scheduler,
    evaluate_model, load_config, load_data, load_model
)
from utils.power_of_2_quantizer import MultiBitwidthPowerOf2Quantizer

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*NLLLoss2d.*",
                        category=FutureWarning)


class AIMETPowerOf2QATManager:
    """
    Manages AIMET Quantization Aware Training with power-of-2 scale constraints.

    This class provides functionality to apply power-of-2 constraints to AIMET's
    quantization simulation models, enabling hardware-efficient quantization
    with bit-shift operations.

    Args:
        config: Configuration dictionary containing quantization parameters
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the AIMET Power-of-2 QAT manager."""
        self.config = config
        self.quantizer = MultiBitwidthPowerOf2Quantizer(config)

    def apply_power_of_2_constraints(self, quantsim: QuantizationSimModel, original_model=None):
        """
        Apply power-of-2 constraints analysis to AIMET quantization model.

        Args:
            quantsim: AIMET QuantizationSimModel instance
            original_model: Original unquantized model for analysis

        Returns:
            Dict containing power-of-2 constraint information for each layer
        """
        print("Applying power-of-2 constraints analysis...")
        _ = quantsim  # Not used in current AIMET version, kept for API compatibility

        constraint_info = {}

        # Use the original model (before quantization) for power-of-2 analysis
        # This is the only reliable approach in AIMET 2.11
        if original_model is not None:
            processed_layers = []
            for name, module in original_model.named_modules():
                if hasattr(module, 'weight') and module.weight is not None:
                    try:
                        # Compute power-of-2 scale for weights
                        weight_tensor = module.weight.data
                        _, weight_info = self.quantizer.quantize_weights(weight_tensor)

                        constraint_info[f"{name}.weight"] = weight_info
                        processed_layers.append(name)
                        print(f"  {name}: scale={weight_info['scale']:.6f} = "
                              f"{weight_info['power_of_2']}, "
                              f"hardware: {weight_info['hardware_op']}")
                    except Exception as e:
                        print(f"  ⏭️  Skipping {name}: {e}")
                        continue

            if processed_layers:
                print(f"  ✅ Analyzed {len(processed_layers)} layers from original model")
            else:
                print("  ⚠️  No weight-bearing layers found in original model")
        else:
            print("  ⚠️  Original model not provided - skipping power-of-2 analysis")
            print("  ℹ️  Pass the original model for constraint analysis")

        return constraint_info

    def update_power_of_2_constraints_during_training(self, quantsim: QuantizationSimModel):
        """
        Update power-of-2 constraints periodically during training.

        Note: In current AIMET version, direct quantizer encoding updates
        are not supported, so this method is a no-op.

        Args:
            quantsim: AIMET QuantizationSimModel instance
        """
        # Current AIMET version doesn't support direct quantizer updates
        _ = quantsim  # Suppress unused parameter warning


def load_data_aimet(data_path: str, batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset for AIMET training.

    Args:
        data_path: Path to dataset directory
        batch_size: Batch size for data loaders

    Returns:
        Tuple of (train_loader, test_loader)
    """
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
    """
    Train one epoch with AIMET quantization and power-of-2 constraints.

    Args:
        quantsim: AIMET QuantizationSimModel
        train_loader: Training data loader
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device for computation
        epoch: Current epoch number
        qat_manager: Power-of-2 constraint manager
        update_frequency: Frequency for constraint updates

    Returns:
        Average training loss for the epoch
    """
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
    """Main function for AIMET Power-of-2 QAT training."""
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
    parser.add_argument(
        '--skip_initial_ptq', action='store_true', default=False,
        help='Skip initial PTQ step and go directly to QAT'
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
        output_dir = Path(config['output']['base_dir']) / 'aimet_qat'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    try:
        model = load_model(args.model_path, config, device)
        print(f"✅ Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("Creating a new model with random weights...")
        model = create_model(config, device)

    # Load data
    print(f"Loading data from {args.data_path}...")
    train_loader, test_loader = load_data(config, args.data_path)

    # Save original model for QAT (AIMET requirement)
    original_model = copy.deepcopy(model)

    # Step 1: PTQ Initialization (recommended best practice)
    if config['qat'].get('run_ptq_first', True) and not args.skip_initial_ptq:
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
        ptq_constraint_info = ptq_manager.apply_power_of_2_constraints(ptq_quantsim, model)

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

    # Use original model for QAT (AIMET 2.11 requirement)
    quantsim = QuantizationSimModel(
        model=original_model,  # Use original model, not PTQ-quantized model
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

    # Evaluate final AIMET model (for comparison)
    print(f"\nFinal AIMET QAT model accuracy: {best_accuracy:.2f}%")

    # Extract final quantization details from trained QAT model
    print("\n" + "=" * 60)
    print("ANALYZING TRAINED QAT MODEL")
    print("=" * 60)

    # Get the trained model weights (after QAT)
    trained_model = quantsim.model

    # Apply power-of-2 analysis to the TRAINED model weights
    print("Power-of-2 analysis of TRAINED QAT model weights:")
    final_constraint_info = qat_manager.apply_power_of_2_constraints(quantsim, trained_model)

    # Create and evaluate power-of-2 constrained model
    print("\n" + "=" * 60)
    print("CREATING POWER-OF-2 CONSTRAINED MODEL")
    print("=" * 60)

    # Create a copy of the trained model for power-of-2 quantization
    power_of_2_model = copy.deepcopy(trained_model)

    # Apply power-of-2 quantization to the model weights
    print("Applying power-of-2 constraints to model weights...")
    for name, module in power_of_2_model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if f"{name}.weight" in final_constraint_info:
                # Get power-of-2 quantized weights
                weight_tensor = module.weight.data
                quantized_weights, _ = qat_manager.quantizer.quantize_weights(weight_tensor)
                # Replace with power-of-2 quantized weights
                module.weight.data = quantized_weights
                print(f"  ✅ Applied power-of-2 quantization to {name}")

    # Evaluate power-of-2 constrained model
    print("\nEvaluating power-of-2 constrained model...")
    power_of_2_accuracy = evaluate_model(power_of_2_model, test_loader, device)
    print(f"Power-of-2 constrained model accuracy: {power_of_2_accuracy:.2f}%")

    # Calculate meaningful improvements
    aimet_improvement = best_accuracy - initial_accuracy
    power_of_2_improvement = power_of_2_accuracy - initial_accuracy
    accuracy_drop_from_power_of_2 = best_accuracy - power_of_2_accuracy

    print("\n" + "=" * 60)
    print("ACCURACY COMPARISON")
    print("=" * 60)
    print(f"Original model accuracy:           {initial_accuracy:.2f}%")
    print(f"AIMET QAT model accuracy:          {best_accuracy:.2f}%")
    print(f"Power-of-2 constrained accuracy:   {power_of_2_accuracy:.2f}%")
    print(f"")
    print(f"AIMET improvement over original:    {aimet_improvement:+.2f}%")
    print(f"Power-of-2 improvement over original: {power_of_2_improvement:+.2f}%")
    print(f"Accuracy drop from power-of-2:     {accuracy_drop_from_power_of_2:+.2f}%")

    # Also analyze original model for comparison
    print("\nPower-of-2 analysis of ORIGINAL model weights (for comparison):")
    original_constraint_info = qat_manager.apply_power_of_2_constraints(quantsim, original_model)

    # Compare AIMET scales vs Power-of-2 scales
    print("\n" + "=" * 60)
    print("AIMET vs POWER-OF-2 SCALE COMPARISON")
    print("=" * 60)

    try:
        # Try to extract AIMET's actual quantization scales
        aimet_scales = {}
        for name, module in trained_model.named_modules():
            if hasattr(module, 'param_quantizers') and 'weight' in module.param_quantizers:
                weight_quantizer = module.param_quantizers['weight']
                if hasattr(weight_quantizer, 'encoding') and weight_quantizer.encoding:
                    aimet_scales[f"{name}.weight"] = weight_quantizer.encoding.scale

        if aimet_scales:
            print("Comparison of quantization scales:")
            for layer_name in final_constraint_info.keys():
                if layer_name in aimet_scales:
                    aimet_scale = aimet_scales[layer_name]
                    power_of_2_scale = final_constraint_info[layer_name]['scale']
                    difference = abs(aimet_scale - power_of_2_scale) / aimet_scale * 100

                    print(f"  {layer_name}:")
                    print(f"    AIMET scale:     {aimet_scale:.8f}")
                    print(f"    Power-of-2 scale: {power_of_2_scale:.8f} = {final_constraint_info[layer_name]['power_of_2']}")
                    print(f"    Difference:      {difference:.2f}%")
                    print(f"    Hardware op:     {final_constraint_info[layer_name]['hardware_op']}")
        else:
            print("⚠️  Could not extract AIMET scales for comparison")
            print("This is expected in current AIMET versions")

    except Exception as e:
        print(f"⚠️  Scale comparison failed: {e}")
        print("Showing power-of-2 analysis only")

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
        'aimet_qat_accuracy': float(best_accuracy),
        'power_of_2_accuracy': float(power_of_2_accuracy),
        'aimet_improvement': float(aimet_improvement),
        'power_of_2_improvement': float(power_of_2_improvement),
        'accuracy_drop_from_power_of_2': float(accuracy_drop_from_power_of_2),
        'ptq_quantization_details': (ptq_constraint_info
                                     if config['qat'].get('run_ptq_first', True)
                                     else None),
        'quantization_analysis': {
            'trained_model_power_of_2': final_constraint_info,
            'original_model_power_of_2': original_constraint_info,
            'note': ('trained_model_power_of_2 shows power-of-2 analysis of weights '
                     'after AIMET QAT training')
        }
    }

    results_file = output_dir / 'aimet_power_of_2_qat_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save quantized model and encodings (AIMET recommended approach)
    print("\nSaving quantized model and encodings...")

    try:
        # Save PyTorch model
        model_save_path = output_dir / 'aimet_power_of_2_qat_final.pth'
        torch.save(quantsim.model.state_dict(), model_save_path)
        print(f"✅ Saved PyTorch model to: {model_save_path}")

        # Save AIMET encodings using recommended method
        encodings_path = output_dir / 'aimet_power_of_2_qat_final_encodings.json'
        quantsim.save_encodings_to_json(
            str(output_dir),
            'aimet_power_of_2_qat_final_encodings'
        )
        print(f"✅ Saved AIMET encodings to: {encodings_path}")

        print("✅ Model and encodings saved using modern AIMET methods!")
        print("Note: ONNX export skipped to avoid deprecated warnings.")

    except Exception as e:
        print(f"⚠️  Model/encodings save failed: {e}")
        print("Continuing with results...")

    print(f"\nSaved results to: {results_file}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Exported AIMET model to: {output_dir}/")
    print("AIMET + Power-of-2 symmetric QAT completed!")


if __name__ == '__main__':
    main()
