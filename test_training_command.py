#!/usr/bin/env python3
"""
Test command for block training - adapted for local testing without data dependencies.
This simulates what would work on a cluster with proper datasets.
"""

import os
import sys

def create_test_command():
    """Generate a test command for block training."""

    print("=== Block Training Test Command ===\n")

    # Base command structure based on your cluster example
    base_cmd = [
        "python trainer_block.py",
        "--conf conf/270m_block.yaml",
        "--save_path /tmp/test_run_block",
        "--override \"{'optimizer': {'lr': 1e-5, 'loss_function': 'BLOCK_FM'}, 'training': {'batch_size': 2, 'max_steps': 10}}\"",
    ]

    print("For cluster training (with real data):")
    cluster_cmd = base_cmd + [
        "--hf_training_data --training_data MLSEn+people",
        "--strategy deepspeed_stage_2"
    ]
    print(" \\\n  ".join(cluster_cmd))

    print("\nFor local CPU testing (validation only, no real data needed):")
    local_cmd = base_cmd + [
        "--validation_only",
        "--strategy ddp",  # simpler strategy for local
        "--override \"{'training': {'batch_size': 1, 'max_steps': 2, 'num_workers': 1}}\""
    ]
    print(" \\\n  ".join(local_cmd))

    print(f"\nConfig file exists: {os.path.exists('conf/270m_block.yaml')}")
    print(f"Statistics files exist: {os.path.exists('statistics/mimi_mean.npy')}")

    print("\nKey differences from original Flow-SLM:")
    print("✓ loss_function: 'BLOCK_FM' (instead of 'FM')")
    print("✓ Uses GSLMBlockPipeline for joint block modeling")
    print("✓ BlockFlowLoss for non-factorized block denoising")
    print("✓ Autoregressive processing between blocks")
    print("✓ No causal masking within blocks (full interaction)")

if __name__ == "__main__":
    create_test_command()