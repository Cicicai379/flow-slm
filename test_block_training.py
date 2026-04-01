#!/usr/bin/env python3
"""Test script for block training pipeline without requiring real data."""

import torch
import yaml
import munch
import tempfile
import os
from trainer_block import BlockLanguageModeling

def create_dummy_batch(batch_size=2, seq_len=100, sr=24000):
    """Create dummy batch for testing."""
    # Generate dummy waveforms
    wav_samples = int(seq_len * sr / 100)  # Approximate samples for seq_len frames
    wavs = torch.randn(batch_size, wav_samples)
    wav_len = torch.ones(batch_size) * 0.8  # 80% of max length
    ids = [f"dummy_{i}" for i in range(batch_size)]

    return ids, wavs, wav_len

def test_block_training():
    """Test block training pipeline locally."""
    print("Testing block training pipeline...")

    # Load configuration
    with open("conf/270m_block.yaml", "r") as f:
        conf_dict = yaml.safe_load(f)

    conf = munch.munchify(conf_dict)

    # Create dummy args
    class Args:
        def __init__(self):
            self.reduction = "block"
            self.ignore_eos = False
            self.use_k_future_tokens = 0

    args = Args()

    try:
        print("1. Initializing BlockLanguageModeling...")
        model = BlockLanguageModeling(args, conf)

        print("2. Creating dummy batch...")
        batch = create_dummy_batch(batch_size=2, seq_len=50)

        print("3. Running forward pass...")
        model.eval()  # Set to eval to avoid training mode issues

        with torch.no_grad():
            total_loss, flow_loss_val, token_loss_val, token_acc = model.forward(batch, reduction="block")

        print(f"✓ Forward pass successful!")
        print(f"  - Total loss: {total_loss}")
        print(f"  - Flow loss: {flow_loss_val}")
        print(f"  - Token loss: {token_loss_val}")
        print(f"  - Token accuracy: {token_acc}")

        print("4. Testing training step...")
        model.train()

        # Test training step (this will call forward internally)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Monkey-patch log method to avoid Lightning overhead
                def dummy_log(*args, **kwargs):
                    pass
                model.log = dummy_log

                loss = model.training_step(batch, 0)
                print(f"✓ Training step successful! Loss: {loss}")

        except Exception as e:
            print(f"✗ Training step failed: {e}")
            raise

        print("\n✓ All tests passed! Block training pipeline is working correctly.")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    # Test if required files exist
    if not os.path.exists("conf/270m_block.yaml"):
        print("Error: conf/270m_block.yaml not found")
        exit(1)

    success = test_block_training()
    exit(0 if success else 1)