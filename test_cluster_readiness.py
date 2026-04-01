#!/usr/bin/env python3
"""
Test script to verify cluster readiness for block training.
Run this on your cluster before attempting full training.
"""

import sys
import traceback

def test_imports():
    """Test all required imports."""
    print("=== Testing Imports ===")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False

    try:
        import lightning
        print(f"✓ Lightning {lightning.__version__}")
    except ImportError as e:
        print(f"✗ Lightning: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False

    try:
        from trainer_block import BlockLanguageModeling
        print("✓ BlockLanguageModeling import")
    except ImportError as e:
        print(f"✗ BlockLanguageModeling: {e}")
        return False

    try:
        from pipeline_block import GSLMBlockPipeline
        print("✓ GSLMBlockPipeline import")
    except ImportError as e:
        print(f"✗ GSLMBlockPipeline: {e}")
        return False

    try:
        from losses_block import BlockFlowLoss
        print("✓ BlockFlowLoss import")
    except ImportError as e:
        print(f"✗ BlockFlowLoss: {e}")
        return False

    return True

def test_config():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")

    try:
        import yaml
        import munch

        with open("conf/270m_block.yaml", "r") as f:
            conf_dict = yaml.safe_load(f)

        conf = munch.munchify(conf_dict)

        # Check required settings
        assert conf.optimizer.loss_function == "BLOCK_FM", f"Expected BLOCK_FM, got {conf.optimizer.loss_function}"
        assert hasattr(conf.model, "block_size"), "Missing block_size in config"

        print(f"✓ Config loaded: block_size={conf.model.block_size}")
        print(f"✓ Loss function: {conf.optimizer.loss_function}")

        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_model_init():
    """Test model initialization without data."""
    print("\n=== Testing Model Initialization ===")

    try:
        import yaml
        import munch
        from trainer_block import BlockLanguageModeling

        # Load config
        with open("conf/270m_block.yaml", "r") as f:
            conf = munch.munchify(yaml.safe_load(f))

        # Create dummy args
        class Args:
            def __init__(self):
                self.reduction = "block"
                self.ignore_eos = False
                self.use_k_future_tokens = 0

        args = Args()

        # Try to initialize model (this will test most components)
        print("Initializing BlockLanguageModeling...")
        model = BlockLanguageModeling(args, conf)

        print("✓ Model initialized successfully")
        print(f"✓ Pipeline type: {type(model.gslm_pipeline).__name__}")
        print(f"✓ Loss function type: {type(model.loss_fn).__name__}")

        return True

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        traceback.print_exc()
        return False

def test_device_compatibility():
    """Test CUDA availability."""
    print("\n=== Testing Device Compatibility ===")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")

        print(f"BF16 supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A'}")

        return True
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Cluster Readiness for Block Training\n")

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Model Initialization", test_model_init),
        ("Device Compatibility", test_device_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} test crashed: {e}")
            results.append((name, False))

    print(f"\n{'='*50}")
    print("🎯 RESULTS:")

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("Your cluster is ready for block training.")
        print("\nRecommended command:")
        print("python trainer_block.py \\")
        print("  --conf conf/270m_block.yaml \\")
        print("  --save_path /data/cicicai/flow_slm/checkpoints/test_run_block \\")
        print("  --override \"{'optimizer': {'lr': 1e-5, 'loss_function': 'BLOCK_FM'}, 'training': {'batch_size': 8}}\" \\")
        print("  --hf_training_data --training_data MLSEn+people \\")
        print("  --strategy deepspeed_stage_2")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print("Fix the failing components before attempting training.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)