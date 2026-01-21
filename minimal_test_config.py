#%%
# MINIMAL TEST CONFIGURATION
# Use this for quick validation (< 10 minutes)
# Replace VISION_CONFIG with this for testing

MINIMAL_TEST_CONFIG = {
    "model": {
        "backbone": "eva02_base_patch14_224.mim_in22k",  # Smaller model
        "pretrained": True,
        "num_classes": 30,
        "drop_rate": 0.1,
        "drop_path_rate": 0.1
    },
    "data": {
        "input_size": 224,  # Smaller input size
        "num_workers": 2,
        "pin_memory": False,
        "sources": [
            # Use only 1-2 sources for quick testing
            {
                "name": "master_30",
                "path": "/kaggle/input/recyclable-and-household-waste-classification/images",
                "type": "master"
            }
        ]
    },
    "training": {
        "batch_size": 4,  # Larger batch for faster testing
        "grad_accum_steps": 4,  # Smaller accumulation
        "learning_rate": 1e-4,  # Higher LR for faster convergence
        "weight_decay": 0.01,
        "num_epochs": 2,  # Just 2 epochs for testing
        "patience": 5,
        "use_amp": True,
        "max_grad_norm": 1.0
    }
}

# To use this config, replace the train_vision_model call with:
# model = train_vision_model(MINIMAL_TEST_CONFIG)

