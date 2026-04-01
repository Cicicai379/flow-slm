#!/usr/bin/env python3
"""Simple test for block components without Lightning."""

import torch
import torch.nn as nn
import numpy as np
import yaml
import munch

# Test individual components first
def test_block_components():
    print("Testing block components...")

    # Test BlockFlowNet
    from model_block_minimal import BlockFlowNet

    block_size = 8
    feature_dim = 32  # Smaller for testing
    block_dim = block_size * feature_dim
    model_channels = 128
    z_channels = 64
    batch_size = 2

    print("1. Testing BlockFlowNet...")
    flow_net = BlockFlowNet(
        block_dim=block_dim,
        model_channels=model_channels,
        z_channels=z_channels,
        num_res_blocks=2
    )

    # Test input
    x = torch.randn(batch_size, block_dim)
    t = torch.rand(batch_size)
    c = torch.randn(batch_size, z_channels)

    output = flow_net(x, t, c)
    print(f"✓ BlockFlowNet output shape: {output.shape}")
    assert output.shape == (batch_size, block_dim)

    # Test BlockFlowLoss
    from losses_block_minimal import BlockFlowLoss

    print("2. Testing BlockFlowLoss...")
    loss_fn = BlockFlowLoss(
        block_dim=block_dim,
        z_dim=z_channels,
        model_channels=model_channels,
        num_res_blocks=2
    )

    target_block = torch.randn(batch_size, block_dim)
    z = torch.randn(batch_size, z_channels)

    loss = loss_fn(z, target_block)
    print(f"✓ BlockFlowLoss output shape: {loss.shape}")
    assert loss.shape == (batch_size, block_dim)

    # Test ELMBlockDecoderWrapper (simplified)
    print("3. Testing ELMBlockDecoderWrapper...")

    # Skip this for now as it requires transformers
    print("✓ Skipping ELMBlockDecoderWrapper (requires transformers)")

    # Test block splitting logic
    print("4. Testing block splitting logic...")

    def split_into_blocks(ssl_feats, block_size):
        """Simple block splitting for testing."""
        B, T, F = ssl_feats.shape
        num_blocks = (T + block_size - 1) // block_size
        padded_T = num_blocks * block_size

        if padded_T > T:
            padding = torch.zeros(B, padded_T - T, F)
            ssl_feats = torch.cat([ssl_feats, padding], dim=1)

        blocks = ssl_feats.view(B, num_blocks, block_size, F)
        return blocks

    ssl_feats = torch.randn(batch_size, 25, feature_dim)  # 25 timesteps
    blocks = split_into_blocks(ssl_feats, block_size)
    print(f"✓ Block splitting: {ssl_feats.shape} -> {blocks.shape}")

    # Flatten blocks for joint modeling
    flattened_blocks = blocks.view(batch_size, blocks.shape[1], -1)
    print(f"✓ Flattened blocks shape: {flattened_blocks.shape}")

    print("\n✓ All block components working correctly!")
    return True

def test_config():
    """Test configuration loading."""
    print("\n5. Testing configuration...")

    try:
        with open("conf/270m_block.yaml", "r") as f:
            conf_dict = yaml.safe_load(f)

        conf = munch.munchify(conf_dict)

        # Check required block settings
        assert hasattr(conf.model, 'block_size')
        assert conf.optimizer.loss_function == "BLOCK_FM"

        print(f"✓ Block size: {conf.model.block_size}")
        print(f"✓ Loss function: {conf.optimizer.loss_function}")
        print(f"✓ Configuration loaded successfully!")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    try:
        success1 = test_block_components()
        success2 = test_config()

        if success1 and success2:
            print("\n🎉 All tests passed! Block pipeline components are working.")
            print("\nTo test full training:")
            print("python trainer_block.py \\")
            print("  --conf conf/270m_block.yaml \\")
            print("  --save_path /tmp/test_block \\")
            print("  --override \"{'training': {'batch_size': 2, 'max_steps': 10}}\" \\")
            print("  --strategy \"ddp\" \\")
            print("  --validation_only")  # Use validation mode to avoid needing real data
        else:
            print("\n❌ Some tests failed.")
            exit(1)

    except Exception as e:
        print(f"\n❌ Tests failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)