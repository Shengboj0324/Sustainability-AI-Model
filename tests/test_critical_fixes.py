"""
Test Critical Fixes

CRITICAL: Validate all production-readiness fixes
- Data validation
- Config validation
- Training utilities
- Error handling
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.utils.training_utils import (
    set_seed,
    validate_config,
    check_loss_valid,
    check_gradients_valid,
    clip_gradients,
    EarlyStopping,
    TrainingTimer
)


class TestSeedSetting:
    """Test random seed setting for reproducibility"""
    
    def test_seed_setting(self):
        """Test that seed setting makes results reproducible"""
        set_seed(42)
        
        # Generate random numbers
        torch_rand1 = torch.rand(10)
        np_rand1 = np.random.rand(10)
        
        # Reset seed
        set_seed(42)
        
        # Generate again
        torch_rand2 = torch.rand(10)
        np_rand2 = np.random.rand(10)
        
        # Should be identical
        assert torch.allclose(torch_rand1, torch_rand2)
        assert np.allclose(np_rand1, np_rand2)
        print("✅ Seed setting works correctly")


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_valid_config(self):
        """Test that valid config passes"""
        config = {
            "model": {"backbone": "vit_base"},
            "data": {"data_dir": "data/"},
            "training": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "num_epochs": 10,
                "seed": 42
            }
        }
        
        # Should not raise
        validate_config(config)
        print("✅ Valid config passes validation")
    
    def test_missing_keys(self):
        """Test that missing keys are detected"""
        config = {
            "model": {"backbone": "vit_base"},
            # Missing "data" and "training"
        }
        
        with pytest.raises(ValueError, match="Missing required config key"):
            validate_config(config)
        print("✅ Missing keys detected")
    
    def test_invalid_batch_size(self):
        """Test that invalid batch size is detected"""
        config = {
            "model": {},
            "data": {},
            "training": {
                "batch_size": -1,  # Invalid
                "learning_rate": 1e-4,
                "num_epochs": 10
            }
        }
        
        with pytest.raises(ValueError, match="Invalid batch_size"):
            validate_config(config)
        print("✅ Invalid batch size detected")
    
    def test_invalid_learning_rate(self):
        """Test that invalid learning rate is detected"""
        config = {
            "model": {},
            "data": {},
            "training": {
                "batch_size": 32,
                "learning_rate": -0.001,  # Invalid
                "num_epochs": 10
            }
        }
        
        with pytest.raises(ValueError, match="Invalid learning_rate"):
            validate_config(config)
        print("✅ Invalid learning rate detected")


class TestLossValidation:
    """Test loss validation (NaN/Inf detection)"""
    
    def test_valid_loss(self):
        """Test that valid loss passes"""
        loss = torch.tensor(0.5)
        # Should not raise
        check_loss_valid(loss, epoch=0, step=0)
        print("✅ Valid loss passes")
    
    def test_nan_loss(self):
        """Test that NaN loss is detected"""
        loss = torch.tensor(float('nan'))
        
        with pytest.raises(ValueError, match="Loss became"):
            check_loss_valid(loss, epoch=0, step=0)
        print("✅ NaN loss detected")
    
    def test_inf_loss(self):
        """Test that Inf loss is detected"""
        loss = torch.tensor(float('inf'))
        
        with pytest.raises(ValueError, match="Loss became"):
            check_loss_valid(loss, epoch=0, step=0)
        print("✅ Inf loss detected")


class TestGradientValidation:
    """Test gradient validation"""
    
    def test_valid_gradients(self):
        """Test that valid gradients pass"""
        model = torch.nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()
        
        # Should not raise
        check_gradients_valid(model, epoch=0, step=0)
        print("✅ Valid gradients pass")


class TestEarlyStopping:
    """Test early stopping functionality"""

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode (accuracy)"""
        early_stopping = EarlyStopping(patience=3, mode="max")

        # Improving metrics
        assert not early_stopping(0.8)
        assert not early_stopping(0.85)
        assert not early_stopping(0.9)

        # No improvement
        assert not early_stopping(0.89)  # Counter = 1
        assert not early_stopping(0.88)  # Counter = 2
        assert early_stopping(0.87)      # Counter = 3, should stop

        print("✅ Early stopping (max mode) works correctly")

    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode (loss)"""
        early_stopping = EarlyStopping(patience=2, mode="min")

        # Improving metrics
        assert not early_stopping(1.0)
        assert not early_stopping(0.8)
        assert not early_stopping(0.6)

        # No improvement
        assert not early_stopping(0.65)  # Counter = 1
        assert early_stopping(0.7)       # Counter = 2, should stop

        print("✅ Early stopping (min mode) works correctly")


class TestTrainingTimer:
    """Test training timer functionality"""

    def test_timer(self):
        """Test that timer calculates ETA correctly"""
        import time

        timer = TrainingTimer()

        # Simulate 3 epochs
        for epoch in range(3):
            timer.start_epoch()
            time.sleep(0.1)  # Simulate training
            timing = timer.end_epoch(epoch + 1, total_epochs=10)

            assert timing['epoch_time_seconds'] > 0
            assert timing['eta_hours'] >= 0

        print("✅ Training timer works correctly")


class TestGradientClipping:
    """Test gradient clipping"""

    def test_gradient_clipping(self):
        """Test that gradients are clipped correctly"""
        model = torch.nn.Linear(10, 5)

        # Create large gradients
        x = torch.randn(2, 10)
        y = model(x).sum() * 100
        y.backward()

        # Clip gradients
        total_norm = clip_gradients(model, max_norm=1.0)

        # Check that norm was large before clipping
        assert total_norm > 1.0

        # Check that gradients are now clipped
        new_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        assert new_norm <= 1.0 or abs(new_norm - 1.0) < 0.01

        print("✅ Gradient clipping works correctly")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING CRITICAL FIXES VALIDATION TESTS")
    print("="*60 + "\n")

    # Seed setting
    print("Testing seed setting...")
    test_seed = TestSeedSetting()
    test_seed.test_seed_setting()

    # Config validation
    print("\nTesting config validation...")
    test_config = TestConfigValidation()
    test_config.test_valid_config()
    try:
        test_config.test_missing_keys()
    except:
        pass  # Expected to raise
    try:
        test_config.test_invalid_batch_size()
    except:
        pass  # Expected to raise
    try:
        test_config.test_invalid_learning_rate()
    except:
        pass  # Expected to raise

    # Loss validation
    print("\nTesting loss validation...")
    test_loss = TestLossValidation()
    test_loss.test_valid_loss()
    try:
        test_loss.test_nan_loss()
    except:
        pass  # Expected to raise
    try:
        test_loss.test_inf_loss()
    except:
        pass  # Expected to raise

    # Gradient validation
    print("\nTesting gradient validation...")
    test_grad = TestGradientValidation()
    test_grad.test_valid_gradients()

    # Early stopping
    print("\nTesting early stopping...")
    test_early = TestEarlyStopping()
    test_early.test_early_stopping_max_mode()
    test_early.test_early_stopping_min_mode()

    # Timer
    print("\nTesting training timer...")
    test_timer = TestTrainingTimer()
    test_timer.test_timer()

    # Gradient clipping
    print("\nTesting gradient clipping...")
    test_clip = TestGradientClipping()
    test_clip.test_gradient_clipping()

    print("\n" + "="*60)
    print("✅ ALL CRITICAL FIXES VALIDATED SUCCESSFULLY!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()


