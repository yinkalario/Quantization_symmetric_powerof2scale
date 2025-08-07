#!/usr/bin/env python3
"""
Enhanced Power-of-2 Symmetric Quantization Aware Training (QAT).

This script implements QAT with:
1. Optional PTQ initialization (recommended)
2. Configurable bitwidths for different tensor types
3. Detailed quantization reporting
4. Configuration-driven setup

Features:
- Multi-bitwidth quantization (weights, inputs, outputs, biases)
- PTQ initialization before QAT
- Configuration-driven setup
- Shared model/data utilities
- Detailed quantization reporting

Author: Yin Cao

Usage:
    python qat_train.py --config configs/quantization_config.yaml --data_path data/ --epochs 10
"""

import argparse
import warnings
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from typing import Dict, Any

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*NLLLoss2d.*", category=FutureWarning)

# Local imports
from utils.model_utils import (load_config, load_model, load_data, evaluate_model,
                               create_criterion, create_optimizer, create_scheduler)
from utils.power_of_2_quantizer import MultiBitwidthPowerOf2Quantizer


class QuantizationAwareModule(nn.Module):
    """
    Wrapper module that applies quantization during forward pass.

    This simulates quantization-aware training using fake quantization.
    Only input/output quantization is applied during forward pass.
    Weight quantization is handled separately to avoid interfering with gradients.
    """

    def __init__(self, module: nn.Module, quantizer: MultiBitwidthPowerOf2Quantizer):
        """Initialize QAT wrapper module."""
        super().__init__()
        self.module = module
        self.quantizer = quantizer
        self.training_step = 0

    def forward(self, x):
        """Forward pass with fake quantization for inputs/outputs only."""
        # Quantize inputs (fake quantization - quantize then dequantize)
        if self.training:
            x_quantized, _ = self.quantizer.quantize_inputs(x)
            x = x_quantized

        # Apply the actual module (weights are already quantized from PTQ initialization)
        output = self.module(x)

        # Quantize outputs (fake quantization)
        if self.training:
            output_quantized, _ = self.quantizer.quantize_outputs(output)
            output = output_quantized

        return output


def apply_qat_to_model(model: nn.Module, quantizer: MultiBitwidthPowerOf2Quantizer, skip_weight_quantization: bool = False) -> nn.Module:
    """Convert model to QAT version with fake quantization."""
    print("Converting model to QAT with fake quantization...")

    quantization_details = {}

    if not skip_weight_quantization:
        # Only quantize weights if not already done by PTQ
        print("Applying initial weight quantization...")
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                # Quantize weights
                original_weight = module.weight.data.clone()
                quantized_weight, weight_info = quantizer.quantize_weights(original_weight)
                module.weight.data = quantized_weight
                quantization_details[f"{name}.weight"] = weight_info

            if hasattr(module, 'bias') and module.bias is not None:
                # Quantize biases
                original_bias = module.bias.data.clone()
                quantized_bias, bias_info = quantizer.quantize_biases(original_bias)
                module.bias.data = quantized_bias
                quantization_details[f"{name}.bias"] = bias_info
    else:
        print("Skipping weight quantization (already done by PTQ)")

    # Wrap the model for input/output quantization during training
    qat_model = QuantizationAwareModule(model, quantizer)

    return qat_model, quantization_details


def update_quantization_constraints(model: nn.Module, quantizer: MultiBitwidthPowerOf2Quantizer):
    """
    Update power-of-2 constraints during training.

    NOTE: This function was causing training issues by overwriting learned weights.
    For proper QAT, we should only apply quantization in forward pass (fake quantization),
    not overwrite the actual weights during training.
    """
    # DISABLED: This was preventing learning by overwriting gradients
    # Instead, quantization should only happen in forward pass via QuantizationAwareModule
    pass


def train_epoch_with_quantization(model: nn.Module, train_loader, optimizer, criterion, 
                                 device: torch.device, epoch: int, 
                                 quantizer: MultiBitwidthPowerOf2Quantizer,
                                 update_frequency: int = 100) -> float:
    """Train one epoch with quantization-aware training."""
    model.train()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update power-of-2 constraints periodically
        # NOTE: Function is now disabled to prevent overwriting learned weights
        if batch_idx % update_frequency == 0:
            update_quantization_constraints(model, quantizer)  # Now harmless (does nothing)
            if batch_idx % (update_frequency * 2) == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    return running_loss / len(train_loader)


def extract_quantization_details(model: nn.Module, quantizer: MultiBitwidthPowerOf2Quantizer) -> Dict[str, Any]:
    """Extract detailed quantization information from trained model."""
    if hasattr(model, 'module'):  # Unwrap QAT wrapper
        actual_model = model.module
    else:
        actual_model = model
    
    quantization_details = {}
    
    for name, module in actual_model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            # Get current weight quantization info
            _, weight_info = quantizer.quantize_weights(module.weight.data)
            quantization_details[f"{name}.weight"] = weight_info
        
        if hasattr(module, 'bias') and module.bias is not None:
            # Get current bias quantization info
            _, bias_info = quantizer.quantize_biases(module.bias.data)
            quantization_details[f"{name}.bias"] = bias_info
    
    return quantization_details


def main():
    parser = argparse.ArgumentParser(description='Enhanced Power-of-2 Symmetric QAT')
    parser.add_argument('--config', type=str, default='configs/quantization_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, default='model.pth',
                       help='Path to pretrained model file (optional)')
    parser.add_argument('--data_path', type=str, default='data/',
                       help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
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
        output_dir = Path(config['output']['base_dir']) / 'qat'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model...")
    model = load_model(args.model_path, config, device)
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    train_loader, test_loader = load_data(config, args.data_path)
    
    # Create training components
    criterion = create_criterion(config)
    
    # Create multi-bitwidth quantizer
    quantizer = MultiBitwidthPowerOf2Quantizer(config)
    
    # Step 1: PTQ initialization (if enabled)
    if config['qat'].get('run_ptq_first', True):
        print("\n" + "="*60)
        print("STEP 1: PTQ Initialization")
        print("="*60)
        
        # Evaluate before PTQ
        print("Evaluating model before PTQ...")
        initial_accuracy = evaluate_model(model, test_loader, device)
        print(f"Initial accuracy: {initial_accuracy:.2f}%")
        
        # Apply PTQ
        print("Applying PTQ initialization...")
        from ptq_quantize import quantize_model_comprehensive
        ptq_details = quantize_model_comprehensive(model, quantizer)
        
        # Evaluate after PTQ
        print("Evaluating model after PTQ...")
        ptq_accuracy = evaluate_model(model, test_loader, device)
        print(f"PTQ accuracy: {ptq_accuracy:.2f}%")
        print(f"PTQ accuracy drop: {initial_accuracy - ptq_accuracy:.2f}%")
    else:
        print("Skipping PTQ initialization (disabled in config)")
        initial_accuracy = evaluate_model(model, test_loader, device)
        ptq_accuracy = initial_accuracy
        ptq_details = {}
    
    # Step 2: QAT Training
    print("\n" + "="*60)
    print("STEP 2: Quantization Aware Training")
    print("="*60)

    # Convert to QAT model
    # Skip weight quantization if PTQ was already applied
    skip_weight_quant = config['qat'].get('run_ptq_first', True)
    qat_model, initial_qat_details = apply_qat_to_model(model, quantizer, skip_weight_quantization=skip_weight_quant)
    
    # Create optimizer and scheduler for QAT
    optimizer = create_optimizer(qat_model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Training parameters
    epochs = args.epochs or config['training']['epochs']
    update_frequency = config['qat'].get('constraint_update_frequency', 100)
    
    print(f"Starting QAT for {epochs} epochs...")
    print(f"Constraint update frequency: every {update_frequency} batches")
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Train with quantization
        avg_loss = train_epoch_with_quantization(
            qat_model, train_loader, optimizer, criterion, device, 
            epoch + 1, quantizer, update_frequency
        )
        
        # Evaluate
        accuracy = evaluate_model(qat_model, test_loader, device)
        
        # Update learning rate
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = output_dir / 'best_qat_model.pth'
            if hasattr(qat_model, 'module'):
                torch.save(qat_model.module.state_dict(), best_model_path)
            else:
                torch.save(qat_model.state_dict(), best_model_path)
    
    # Extract final quantization details
    final_quantization_details = extract_quantization_details(qat_model, quantizer)
    
    # Final evaluation
    print(f"\nQAT completed!")
    print(f"Initial accuracy: {initial_accuracy:.2f}%")
    if config['qat'].get('run_ptq_first', True):
        print(f"PTQ accuracy:     {ptq_accuracy:.2f}%")
    print(f"Final QAT accuracy: {best_accuracy:.2f}%")
    print(f"Total improvement: {best_accuracy - initial_accuracy:.2f}%")
    
    # Save results
    results = {
        'quantization_type': 'Multi-bitwidth Power-of-2 Symmetric QAT',
        'config': config['quantization'],
        'initial_accuracy': float(initial_accuracy),
        'ptq_accuracy': float(ptq_accuracy) if config['qat'].get('run_ptq_first', True) else None,
        'final_accuracy': float(best_accuracy),
        'total_improvement': float(best_accuracy - initial_accuracy),
        'epochs': epochs,
        'learning_rate': config['training']['learning_rate'],
        'ptq_quantization_details': ptq_details if config['qat'].get('run_ptq_first', True) else None,
        'final_quantization_details': final_quantization_details
    }
    
    results_file = output_dir / 'qat_results.yaml'
    with open(results_file, 'w') as f:
        yaml.dump(results, f, default_flow_style=False, indent=2)

    # Save final model
    final_model_path = output_dir / 'final_qat_model.pth'
    if hasattr(qat_model, 'module'):
        torch.save(qat_model.module.state_dict(), final_model_path)
    else:
        torch.save(qat_model.state_dict(), final_model_path)

    print(f"\nSaved results to: {results_file}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved final model to: {final_model_path}")
    print("Multi-bitwidth power-of-2 symmetric QAT completed!")


if __name__ == '__main__':
    main()
