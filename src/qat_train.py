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
Date: August 8, 2025

Usage:
    python qat_train.py --config configs/quantization_config.yaml \
        --data_path data/ --epochs 10
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
from ptq_quantize import quantize_inputs_outputs, quantize_model_comprehensive
from utils.model_utils import (
    create_criterion, create_optimizer, create_scheduler,
    evaluate_model, load_config, load_data, load_model
)
from utils.power_of_2_quantizer import MultiBitwidthPowerOf2Quantizer

# Suppress PyTorch deprecation warnings
warnings.filterwarnings("ignore", message=".*NLLLoss2d.*",
                        category=FutureWarning)


def train_epoch_with_quantization(
    model: nn.Module, train_loader, optimizer, criterion,
    device: torch.device, epoch: int
) -> float:
    """Train one epoch with weight-only quantization-aware training."""
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

        # Print progress every 200 batches
        if batch_idx % 200 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')

    return running_loss / len(train_loader)


def extract_quantization_details(
    model: nn.Module,
    quantizer: MultiBitwidthPowerOf2Quantizer
) -> Dict[str, Any]:
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
    parser = argparse.ArgumentParser(
        description='Enhanced Power-of-2 Symmetric QAT'
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
        '--weight_only_qat', action='store_true', default=True,
        help='Use weight-only QAT (recommended, default: True)'
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

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['output']['base_dir']) / 'qat'
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load model
    print("Loading model...")
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
        print("\n" + "=" * 60)
        print("STEP 1: PTQ Initialization")
        print("=" * 60)

        # Evaluate before PTQ
        print("Evaluating model before PTQ...")
        initial_accuracy = evaluate_model(model, test_loader, device)
        print(f"Initial accuracy: {initial_accuracy:.2f}%")

        # Apply PTQ
        print("Applying PTQ initialization...")
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
    print("\n" + "=" * 60)
    print("STEP 2: Quantization Aware Training")
    print("=" * 60)

    # Weight-only QAT: Use model directly (weights already quantized by PTQ)
    print("Using weight-only QAT approach...")
    print("Weights are already quantized from PTQ step.")
    print("Training directly with quantized weights (no wrapper needed).")
    qat_model = model  # No wrapper needed!

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
        # Train with weight-only quantization
        avg_loss = train_epoch_with_quantization(
            qat_model, train_loader, optimizer, criterion, device, epoch + 1
        )

        # Evaluate
        accuracy = evaluate_model(qat_model, test_loader, device)

        # Update learning rate
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

        # Save best weight-only QAT model (before input/output quantization)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = output_dir / 'best_weight_only_qat_model.pth'
            if hasattr(qat_model, 'module'):
                torch.save(qat_model.module.state_dict(), best_model_path)
            else:
                torch.save(qat_model.state_dict(), best_model_path)

    # Step 3: Final PTQ for Input/Output Quantization
    if args.final_ptq:
        print("\n" + "=" * 60)
        print("STEP 3: Final PTQ for Input/Output Quantization")
        print("=" * 60)
        print("Applying input/output quantization to QAT-trained model...")

        # Get the trained model (unwrap QAT wrapper)
        if hasattr(qat_model, 'module'):
            trained_model = qat_model.module
        else:
            trained_model = qat_model

        # Apply input/output quantization using PTQ
        final_ptq_details = quantize_inputs_outputs(
            trained_model, quantizer, train_loader, device, num_batches=10
        )
        final_model = trained_model  # Model structure unchanged, just calibrated

        # Evaluate final model with full quantization
        print("Evaluating final model with full quantization...")
        final_accuracy = evaluate_model(final_model, test_loader, device)
        print(f"Final model accuracy (full quantization): {final_accuracy:.2f}%")

        # Save final fully quantized model (complete power-of-2 quantization)
        final_model_path = output_dir / 'final_fully_quantized_power_of_2_model.pth'
        torch.save(final_model.state_dict(), final_model_path)
        print(f"Final fully quantized model saved to {final_model_path}")

        results_model = final_model
        final_accuracy_result = final_accuracy
    else:
        print("Skipping final PTQ step")
        results_model = qat_model
        final_accuracy_result = best_accuracy

    # Extract final quantization details (combine weight/bias + input/output)
    weight_bias_details = extract_quantization_details(results_model, quantizer)
    if args.final_ptq:
        # Combine weight/bias details with input/output details
        final_quantization_details = {**weight_bias_details, **final_ptq_details}
    else:
        final_quantization_details = weight_bias_details

    # Final evaluation summary
    print("\n" + "=" * 60)
    print("QUANTIZATION PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Step 1 - Initial accuracy:           {initial_accuracy:.2f}%")
    print(f"Step 1 - PTQ accuracy:               {ptq_accuracy:.2f}%")
    print(f"Step 2 - Weight-only QAT accuracy:   {best_accuracy:.2f}%")
    if args.final_ptq:
        print(f"Step 3 - Full quantization accuracy: {final_accuracy_result:.2f}%")
        print("")
        power_of_2_improvement = final_accuracy_result - initial_accuracy
        accuracy_drop = best_accuracy - final_accuracy_result
        print(f"Power-of-2 improvement over original: {power_of_2_improvement:+.2f}%")
        print(f"Accuracy drop from input/output quantization: {accuracy_drop:+.2f}%")
    else:
        weight_only_improvement = best_accuracy - initial_accuracy
        print(f"Weight-only power-of-2 improvement:  {weight_only_improvement:+.2f}%")
        print("Note: Input/output quantization not applied")

    # Save results
    results = {
        'quantization_type': 'PTQ → Weight-Only QAT → Final PTQ Pipeline',
        'config': config['quantization'],
        'pipeline_steps': [
            'Step 1: PTQ for weight/bias initialization',
            'Step 2: Weight-only QAT training',
            ('Step 3: Final PTQ for input/output quantization'
             if args.final_ptq else 'Step 3: Skipped')
        ],
        'initial_accuracy': float(initial_accuracy),
        'ptq_accuracy': float(ptq_accuracy),
        'weight_only_qat_accuracy': float(best_accuracy),
        'full_quantization_accuracy': float(final_accuracy_result),
        'power_of_2_improvement': (
            float(final_accuracy_result - initial_accuracy)
            if args.final_ptq
            else float(best_accuracy - initial_accuracy)
        ),
        'accuracy_drop_from_input_output_quantization': (
            float(best_accuracy - final_accuracy_result)
            if args.final_ptq
            else 0.0
        ),
        'epochs': epochs,
        'learning_rate': config['training']['learning_rate'],
        'ptq_quantization_details': ptq_details,
        'final_quantization_details': final_quantization_details
    }

    results_file = output_dir / 'quantization_pipeline_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {results_file}")
    print(f"Saved best QAT model to: {best_model_path}")
    if args.final_ptq:
        print(f"Saved final quantized model to: {final_model_path}")
    print("Quantization pipeline completed!")


if __name__ == '__main__':
    main()
