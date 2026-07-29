#!/usr/bin/env python3
"""
Compare JAX and PyTorch model checkpoints for consistency.

This script performs two levels of comparison:
1. Weight-level: Compare parameter values between checkpoints
2. Runtime-level: Compare action outputs on identical SRB inputs

Usage:
    python scripts/compare_jax_vs_pytorch.py

Environment:
    OPENPI_DISABLE_COMPILE=1  — disable torch.compile for cleaner debugging
"""

import os
import sys

# Set environment before importing torch
os.environ.setdefault("OPENPI_DISABLE_COMPILE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import safetensors.torch
import torch

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openpi.training import config as _config
from openpi.policies import policy_config


# ============================================================================
# 1. Weight-level comparison
# ============================================================================


def load_safetensors_weights(path: str) -> dict[str, torch.Tensor]:
    """Load weights from a safetensors file."""
    return safetensors.torch.load_file(path)


def compare_weight_dicts(
    name_a: str,
    weights_a: dict[str, torch.Tensor],
    name_b: str,
    weights_b: dict[str, torch.Tensor],
) -> dict:
    """Compare two weight dictionaries and return detailed statistics."""
    keys_a = set(weights_a.keys())
    keys_b = set(weights_b.keys())
    common_keys = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a

    print(f"\n{'='*80}")
    print(f"Weight comparison: {name_a} vs {name_b}")
    print(f"{'='*80}")
    print(f"Keys in {name_a}: {len(keys_a)}")
    print(f"Keys in {name_b}: {len(keys_b)}")
    print(f"Common keys: {len(common_keys)}")
    print(f"Only in {name_a}: {len(only_a)}")
    print(f"Only in {name_b}: {len(only_b)}")

    if only_a:
        print(f"\nSample keys only in {name_a}:")
        for k in sorted(only_a)[:10]:
            print(f"  {k}  shape={weights_a[k].shape}")
    if only_b:
        print(f"\nSample keys only in {name_b}:")
        for k in sorted(only_b)[:10]:
            print(f"  {k}  shape={weights_b[k].shape}")

    # Compare common keys
    results = []
    for key in sorted(common_keys):
        a = weights_a[key].float().numpy()
        b = weights_b[key].float().numpy()
        if a.shape != b.shape:
            results.append({"key": key, "status": "shape_mismatch", "shape_a": a.shape, "shape_b": b.shape})
            continue
        diff = np.abs(a - b)
        results.append({
            "key": key,
            "status": "ok",
            "mean_abs": float(diff.mean()),
            "max_abs": float(diff.max()),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "relative_mean": float(diff.mean() / (np.abs(a).mean() + 1e-8)),
        })

    # Report statistics
    ok_results = [r for r in results if r["status"] == "ok"]
    mismatch_results = [r for r in results if r["status"] == "shape_mismatch"]

    if mismatch_results:
        print(f"\nShape mismatches: {len(mismatch_results)}")
        for r in mismatch_results:
            print(f"  {r['key']}: {r['shape_a']} vs {r['shape_b']}")

    if ok_results:
        mean_abss = [r["mean_abs"] for r in ok_results]
        max_abs = [r["max_abs"] for r in ok_results]
        rmses = [r["rmse"] for r in ok_results]

        print(f"\nCommon-key statistics ({len(ok_results)} params):")
        print(f"  Mean absolute diff (avg): {np.mean(mean_abss):.6f}")
        print(f"  Mean absolute diff (max): {np.max(mean_abss):.6f}")
        print(f"  Max absolute diff (avg):  {np.mean(max_abs):.6f}")
        print(f"  Max absolute diff (max):  {np.max(max_abs):.6f}")
        print(f"  RMSE (avg):               {np.mean(rmses):.6f}")
        print(f"  RMSE (max):               {np.max(rmses):.6f}")

        # Top-10 most different
        top_diff = sorted(ok_results, key=lambda r: r["max_abs"], reverse=True)[:10]
        print(f"\nTop-10 most different parameters:")
        for r in top_diff:
            print(f"  {r['key']}: mean_abs={r['mean_abs']:.6f}, max_abs={r['max_abs']:.6f}, relative={r['relative_mean']:.4f}")

    return {
        "keys_a": len(keys_a),
        "keys_b": len(keys_b),
        "common": len(common_keys),
        "only_a": len(only_a),
        "only_b": len(only_b),
        "ok_results": ok_results,
        "mismatch_results": mismatch_results,
    }


# ============================================================================
# 2. Runtime-level comparison
# ============================================================================


def build_obs_for_policy(sample, policy_name: str, config_name: str) -> dict:
    """Build observation dict that works for both JAX and PyTorch policies."""
    # Include all possible key variants so both transforms can find what they need
    obs = {
        # SRB-style flat keys (for srb_policy transforms)
        "proprio": np.asarray(sample["observation.state"]).astype(np.float32),
        "state": np.asarray(sample["observation.state"]).astype(np.float32),
        "image_base": np.asarray(sample["observation.images.image_base"]).astype(np.float32),
        "image_wrist": np.asarray(sample["observation.images.image_wrist"]).astype(np.float32),
        "prompt": sample["task"],
        # LeRobot-style nested keys (for srb_train transforms)
        "observation.state": np.asarray(sample["observation.state"]).astype(np.float32),
        "observation.images.image_base": np.asarray(sample["observation.images.image_base"]).astype(np.float32),
        "observation.images.image_wrist": np.asarray(sample["observation.images.image_wrist"]).astype(np.float32),
    }
    return obs


def compare_runtime_outputs(
    name_a: str,
    policy_a,
    name_b: str,
    policy_b,
    samples: list,
    num_steps: int = 2,
) -> dict:
    """Compare runtime outputs of two policies on the same inputs."""
    print(f"\n{'='*80}")
    print(f"Runtime comparison: {name_a} vs {name_b}")
    print(f"{'='*80}")

    stats = []
    for idx, sample in enumerate(samples):
        obs_a = build_obs_for_policy(sample, name_a, "pi05_srb")
        obs_b = build_obs_for_policy(sample, name_b, "srb_train")

        # Use identical zero noise
        noise = np.zeros((1, 16, 32), dtype=np.float32)

        out_a = policy_a.infer(obs_a, noise=noise)
        out_b = policy_b.infer(obs_b, noise=noise)

        actions_a = np.asarray(out_a["actions"], dtype=np.float32)
        actions_b = np.asarray(out_b["actions"], dtype=np.float32)

        diff = actions_a - actions_b
        abs_diff = np.abs(diff)

        # Per-dimension stats
        per_dim_mean = abs_diff.mean(axis=0)
        per_dim_max = abs_diff.max(axis=0)

        # Sample printing: handle both (H, D) and (B, H, D) shapes
        if actions_a.ndim == 3:
            a_sample = actions_a[0, :3, :5].tolist()
            b_sample = actions_b[0, :3, :5].tolist()
        elif actions_a.ndim == 2:
            a_sample = actions_a[:3, :5].tolist()
            b_sample = actions_b[:3, :5].tolist()
        else:
            a_sample = actions_a.ravel()[:5].tolist()
            b_sample = actions_b.ravel()[:5].tolist()

        md = {
            "idx": idx,
            "shape": actions_a.shape,
            "mean_abs": float(abs_diff.mean()),
            "max_abs": float(abs_diff.max()),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "l2": float(np.linalg.norm(diff.ravel())),
            "per_dim_mean": per_dim_mean.tolist(),
            "per_dim_max": per_dim_max.tolist(),
            "actions_a_sample": a_sample,
            "actions_b_sample": b_sample,
        }
        stats.append(md)

        print(f"\nSample {idx}:")
        print(f"  Shape: {md['shape']}")
        print(f"  Mean abs diff: {md['mean_abs']:.6f}")
        print(f"  Max abs diff:  {md['max_abs']:.6f}")
        print(f"  RMSE:          {md['rmse']:.6f}")
        print(f"  L2 norm:       {md['l2']:.6f}")
        print(f"  {name_a} actions[0,:3,:5]: {md['actions_a_sample']}")
        print(f"  {name_b} actions[0,:3,:5]: {md['actions_b_sample']}")

    # Aggregate
    agg = {
        "mean_abs": float(np.mean([s["mean_abs"] for s in stats])),
        "max_abs": float(np.mean([s["max_abs"] for s in stats])),
        "rmse": float(np.mean([s["rmse"] for s in stats])),
        "l2": float(np.mean([s["l2"] for s in stats])),
        "max_over_all_samples": float(max(s["max_abs"] for s in stats)),
    }

    print(f"\nAggregate over {len(stats)} samples:")
    for k, v in agg.items():
        print(f"  {k}: {v:.6f}")

    return {"per_sample": stats, "aggregate": agg}


# ============================================================================
# 3. Component-level comparison (intermediate outputs)
# ============================================================================


def compare_component_outputs(
    name_a: str,
    policy_a,
    name_b: str,
    policy_b,
    sample: dict,
) -> None:
    """Compare intermediate outputs (after normalization, after image embedding, etc.)."""
    print(f"\n{'='*80}")
    print(f"Component comparison: {name_a} vs {name_b}")
    print(f"{'='*80}")

    obs_a = build_obs_for_policy(sample, name_a, "pi05_srb")
    obs_b = build_obs_for_policy(sample, name_b, "srb_train")

    # Compare state after transform
    inputs_a = dict(obs_a)
    inputs_b = dict(obs_b)

    inputs_a = policy_a._input_transform(inputs_a)
    inputs_b = policy_b._input_transform(inputs_b)

    # Compare state
    if "state" in inputs_a and "state" in inputs_b:
        state_a = np.asarray(inputs_a["state"], dtype=np.float32).ravel()
        state_b = np.asarray(inputs_b["state"], dtype=np.float32).ravel()
        min_len = min(len(state_a), len(state_b))
        diff = state_a[:min_len] - state_b[:min_len]
        print(f"\nState after transform:")
        print(f"  Shape A: {state_a.shape}, Shape B: {state_b.shape}")
        print(f"  Values A: {state_a[:8]}")
        print(f"  Values B: {state_b[:8]}")
        print(f"  Mean abs diff: {np.abs(diff).mean():.6f}")
        print(f"  Max abs diff: {np.abs(diff).max():.6f}")

    # Compare images
    for img_key in ["base_0_rgb", "left_wrist_0_rgb"]:
        if img_key in inputs_a.get("image", {}) and img_key in inputs_b.get("image", {}):
            img_a = np.asarray(inputs_a["image"][img_key], dtype=np.float32)
            img_b = np.asarray(inputs_b["image"][img_key], dtype=np.float32)
            diff = np.abs(img_a - img_b)
            print(f"\nImage '{img_key}' after transform:")
            print(f"  Shape A: {img_a.shape}, Shape B: {img_b.shape}")
            print(f"  Dtype A: {img_a.dtype}, Dtype B: {img_b.dtype}")
            print(f"  Range A: [{img_a.min():.4f}, {img_a.max():.4f}]")
            print(f"  Range B: [{img_b.min():.4f}, {img_b.max():.4f}]")
            print(f"  Mean abs diff: {diff.mean():.6f}")
            print(f"  Max abs diff: {diff.max():.6f}")

    # Compare prompt
    if "prompt" in inputs_a and "prompt" in inputs_b:
        print(f"\nPrompt A: '{inputs_a['prompt']}'")
        print(f"Prompt B: '{inputs_b['prompt']}'")
        print(f"Match: {inputs_a['prompt'] == inputs_b['prompt']}")


# ============================================================================
# Main
# ============================================================================


def main():
    print("=" * 80)
    print("JAX vs PyTorch Model Comparison")
    print("=" * 80)

    # --- Paths ---
    jax_checkpoint = ".cache/openpi/openpi-assets/checkpoints/pi05_base"
    converted_checkpoint = "checkpoints/pi05_srb_pytorch/model.safetensors"
    srb_train_checkpoint = "checkpoints/srb_train/model.safetensors"

    # ---------------------------------------------------------------
    # Part 1: Weight-level comparison
    # ---------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# PART 1: Weight-level comparison")
    print("#" * 80)

    if os.path.exists(converted_checkpoint) and os.path.exists(srb_train_checkpoint):
        w_converted = load_safetensors_weights(converted_checkpoint)
        w_srb_train = load_safetensors_weights(srb_train_checkpoint)

        compare_weight_dicts(
            "Converted (pi05_srb_pytorch)",
            w_converted,
            "Trained (srb_train)",
            w_srb_train,
        )
    else:
        print("Skipping weight comparison: checkpoints not found")
        if not os.path.exists(converted_checkpoint):
            print(f"  Missing: {converted_checkpoint}")
        if not os.path.exists(srb_train_checkpoint):
            print(f"  Missing: {srb_train_checkpoint}")

    # ---------------------------------------------------------------
    # Part 2: Runtime-level comparison
    # ---------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# PART 2: Runtime-level comparison")
    print("#" * 80)

    # Load dataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("srb_dataset")
    samples = [ds[i] for i in range(min(5, len(ds)))]

    # Create policies
    jax_cfg = _config.get_config("pi05_srb")
    jax_policy = policy_config.create_trained_policy(
        jax_cfg,
        jax_checkpoint,
        sample_kwargs={"num_steps": 10},
        pytorch_device="cpu",
    )

    # Converted PyTorch policy (same JAX weights, PyTorch implementation)
    converted_dir = os.path.dirname(converted_checkpoint) if "model.safetensors" in converted_checkpoint else converted_checkpoint
    converted_cfg = _config.get_config("pi05_srb")
    converted_policy = policy_config.create_trained_policy(
        converted_cfg,
        converted_dir,
        sample_kwargs={"num_steps": 10},
        pytorch_device="cpu",
    )

    # Compare JAX vs converted PyTorch (same weights, different implementation)
    compare_runtime_outputs(
        "JAX base",
        jax_policy,
        "Converted PyTorch",
        converted_policy,
        samples,
    )

    # Trained PyTorch policy (LoRA-trained, different weights)
    # srb_train uses LoRA variants which may not be available in PyTorch
    try:
        srb_train_dir = os.path.dirname(srb_train_checkpoint) if "model.safetensors" in srb_train_checkpoint else srb_train_checkpoint
        srb_train_cfg = _config.get_config("srb_train")
        srb_train_policy = policy_config.create_trained_policy(
            srb_train_cfg,
            srb_train_dir,
            sample_kwargs={"num_steps": 10},
            pytorch_device="cpu",
        )

        # Compare JAX vs trained PyTorch (different weights + different implementation)
        compare_runtime_outputs(
            "JAX base",
            jax_policy,
            "Trained srb_train",
            srb_train_policy,
            samples,
        )

        # Compare converted vs trained PyTorch (different weights, same implementation)
        compare_runtime_outputs(
            "Converted PyTorch",
            converted_policy,
            "Trained srb_train",
            srb_train_policy,
            samples,
        )
    except Exception as e:
        print(f"\nSkipping srb_train policy (LoRA variants not available): {e}")

    # ---------------------------------------------------------------
    # Part 3: Component-level comparison
    # ---------------------------------------------------------------
    print("\n" + "#" * 80)
    print("# PART 3: Component-level comparison (preprocessing)")
    print("#" * 80)

    compare_component_outputs(
        "JAX base",
        jax_policy,
        "Converted PyTorch",
        converted_policy,
        samples[0],
    )

    compare_component_outputs(
        "JAX base",
        jax_policy,
        "Trained srb_train",
        srb_train_policy,
        samples[0],
    )

    print("\n" + "=" * 80)
    print("Comparison complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
