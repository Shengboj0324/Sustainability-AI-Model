#!/usr/bin/env python3
"""
Training Diagnostics Script
Validates the training setup before running the full training loop.
"""

import sys
import torch
import timm
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

def check_pytorch_installation():
    """Check PyTorch installation and device availability."""
    print("="*80)
    print("PYTORCH INSTALLATION CHECK")
    print("="*80)
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ Python version: {sys.version}")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available")
    
    # Check MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon) available")
    else:
        print("✗ MPS not available")
    
    print()

def check_model_availability():
    """Check if the model can be loaded."""
    print("="*80)
    print("MODEL AVAILABILITY CHECK")
    print("="*80)
    
    model_name = "eva02_base_patch14_224"
    
    try:
        print(f"Attempting to load model: {model_name}")
        model = timm.create_model(model_name, pretrained=False, num_classes=30)
        print(f"✓ Model loaded successfully")
        
        # Check model config
        if hasattr(model, 'default_cfg'):
            cfg = model.default_cfg
            print(f"  Model config:")
            print(f"    Input size: {cfg.get('input_size', 'N/A')}")
            print(f"    Mean: {cfg.get('mean', 'N/A')}")
            print(f"    Std: {cfg.get('std', 'N/A')}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        
        del model
        print()
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print()
        return False

def check_transform_pipeline():
    """Check transform pipeline."""
    print("="*80)
    print("TRANSFORM PIPELINE CHECK")
    print("="*80)
    
    input_size = 224
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Test with dummy image
    dummy_img = Image.new('RGB', (640, 480), color=(128, 128, 128))
    
    try:
        transformed = transform(dummy_img)
        print(f"✓ Transform successful")
        print(f"  Input size: 640x480")
        print(f"  Output shape: {transformed.shape}")
        print(f"  Expected: torch.Size([3, {input_size}, {input_size}])")
        
        if transformed.shape == torch.Size([3, input_size, input_size]):
            print(f"  ✓ Output shape matches expected")
        else:
            print(f"  ✗ Output shape mismatch!")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Transform failed: {e}")
        print()
        return False

def check_forward_pass():
    """Check model forward pass."""
    print("="*80)
    print("FORWARD PASS CHECK")
    print("="*80)
    
    model_name = "eva02_base_patch14_224"
    input_size = 224
    batch_size = 2
    num_classes = 30
    
    try:
        # Load model
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, input_size, input_size)
        
        print(f"Model: {model_name}")
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: torch.Size([{batch_size}, {num_classes}])")
        
        if output.shape == torch.Size([batch_size, num_classes]):
            print(f"  ✓ Output shape matches expected")
        else:
            print(f"  ✗ Output shape mismatch!")
            return False
        
        del model, dummy_input, output
        print()
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        print()
        return False

def main():
    """Run all diagnostic checks."""
    print("\n" + "="*80)
    print("TRAINING DIAGNOSTICS")
    print("="*80 + "\n")
    
    checks = [
        ("PyTorch Installation", check_pytorch_installation),
        ("Model Availability", check_model_availability),
        ("Transform Pipeline", check_transform_pipeline),
        ("Forward Pass", check_forward_pass),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

