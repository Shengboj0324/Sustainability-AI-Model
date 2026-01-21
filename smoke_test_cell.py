#%%
# SMOKE TEST - Run this cell to validate the notebook setup
# This should complete in < 3 minutes and catch common failure modes

import torch
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def run_smoke_test():
    """
    Comprehensive smoke test for the training notebook.
    Tests: imports, GPU, data paths, model creation, single batch training.
    """
    print("="*60)
    print("SMOKE TEST - Validating Notebook Setup")
    print("="*60)
    
    # Test 1: GPU Availability
    print("\n[1/6] Testing GPU availability...")
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ No GPU available - training will be slow")
    
    # Test 2: Data Paths
    print("\n[2/6] Testing data paths...")
    data_sources = VISION_CONFIG["data"]["sources"]
    found_sources = 0
    for source in data_sources:
        path = Path(source["path"])
        if path.exists():
            print(f"✓ Found: {source['name']} at {source['path']}")
            found_sources += 1
        else:
            print(f"✗ Missing: {source['name']} at {source['path']}")
    
    if found_sources == 0:
        print("❌ No data sources found! Please attach datasets.")
        return False
    print(f"✓ Found {found_sources}/{len(data_sources)} data sources")
    
    # Test 3: Model Creation
    print("\n[3/6] Testing model creation...")
    try:
        test_model = create_vision_model(VISION_CONFIG)
        param_count = sum(p.numel() for p in test_model.parameters()) / 1e6
        print(f"✓ Model created successfully: {param_count:.2f}M parameters")
        del test_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 4: Dataset Loading
    print("\n[4/6] Testing dataset loading...")
    try:
        test_dataset = UnifiedWasteDataset(
            sources_config=VISION_CONFIG["data"]["sources"],
            target_classes=TARGET_CLASSES,
            transform=None
        )
        print(f"✓ Dataset loaded: {len(test_dataset)} images")
        if len(test_dataset) == 0:
            print("❌ Dataset is empty!")
            return False
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test 5: DataLoader
    print("\n[5/6] Testing dataloader...")
    try:
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        batch = next(iter(test_loader))
        images, labels = batch
        print(f"✓ DataLoader working: batch shape {images.shape}, labels shape {labels.shape}")
    except Exception as e:
        print(f"❌ DataLoader failed: {e}")
        return False
    
    # Test 6: Single Forward Pass
    print("\n[6/6] Testing single forward pass...")
    try:
        device = get_device()
        test_model = create_vision_model(VISION_CONFIG).to(device)
        test_model.eval()
        
        with torch.no_grad():
            images = images.to(device)
            outputs = test_model(images)
            print(f"✓ Forward pass successful: output shape {outputs.shape}")
        
        del test_model, images, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ SMOKE TEST PASSED - Ready to train!")
    print("="*60)
    return True

# Run the smoke test
if __name__ == "__main__":
    success = run_smoke_test()
    if not success:
        raise RuntimeError("Smoke test failed! Fix issues before training.")

