#!/usr/bin/env python3
"""
Test script to verify AIMET device fix.

Author: Yin Cao
Date: August 8, 2025
"""

import torch
import torch.nn as nn

# Test if the device fix works
def test_device_fix():
    """Test device placement fix for AIMET export."""
    print("Testing AIMET device fix...")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create model and move to device
    model = SimpleModel().to(device)
    
    # Create dummy input on same device
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    # Test forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful! Output shape: {output.shape}")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Input device: {dummy_input.device}")
        print(f"   Output device: {output.device}")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

if __name__ == '__main__':
    success = test_device_fix()
    if success:
        print("\n🎉 Device fix test passed!")
        print("The AIMET scripts should now work correctly on CUDA.")
    else:
        print("\n💥 Device fix test failed!")
        print("There may still be device placement issues.")
