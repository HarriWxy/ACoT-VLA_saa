#!/usr/bin/env python3
"""
Load HuggingFace Gemma 4 pretrained weights into PI0Pytorch model.

This script loads pretrained Gemma 4 language model weights from a local
or HuggingFace checkpoint into the PI0Pytorch model. The vision tower
and multimodal embedder are loaded automatically by Gemma4WithExpertModel
from the same checkpoint during __init__.

Usage:
    # Load Gemma 4 E2B VLM + random action expert, save as PyTorch checkpoint
    python examples/load_gemma4_hf_weights.py \
        --config_name pi0_aloha_sim_gemma4 \
        --output_path ./checkpoints/gemma4_base_pytorch

    # Load with PI05 mode (adarms)
    python examples/load_gemma4_hf_weights.py \
        --config_name pi05_libero_gemma4 \
        --output_path ./checkpoints/gemma4_pi05_pytorch

Prerequisites:
    pip install transformers>=5.10.1 safetensors huggingface_hub

Note:
    - The action expert (Gemma4 300M) is randomly initialized since there is
      no pretrained action expert available. Fine-tuning will train it from scratch.
    - The VLM + vision tower + multimodal embedder are loaded from the
      checkpoint specified by `gemma4_model_path` in the training config.
    - After running this script, use the output path as `pytorch_weight_path`
      in your training config.
"""

import argparse
import logging
import os
import pathlib

import safetensors.torch
import torch
from transformers import AutoModelForCausalLM

import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
import openpi.training.config as _config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# os.environ["HF_ENDPOINT"]= "https://hf-mirror.com"

def load_vlm_weights_into_model(model: pi0_pytorch.PI0Pytorch, vlm_model_id: str):
    """Load pretrained VLM language model weights into the PI0Pytorch model.

    The vision tower and multimodal embedder are already loaded by
    Gemma4WithExpertModel.__init__ from the checkpoint. This function
    only loads the text decoder weights into gemma4_vlm.

    Args:
        model: The PI0Pytorch model to load weights into.
        vlm_model_id: Path or HuggingFace model ID for Gemma 4.
    """
    gemma4_with_expert = model.paligemma_with_expert
    assert hasattr(gemma4_with_expert, "gemma4_vlm"), "Model must be a Gemma 4 variant"

    # --- Load Gemma 4 language model weights ---
    logger.info(f"Loading Gemma 4 language model from {vlm_model_id}...")
    hf_gemma4 = AutoModelForCausalLM.from_pretrained(
        vlm_model_id,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # Map HuggingFace Gemma4 text model weights to our gemma4_vlm
    # HF keys: "model.layers.0.self_attn.q_proj.weight"
    # Ours:    "gemma4_vlm.model.layers.0.self_attn.q_proj.weight"
    vlm_state_dict = {}
    hf_state = hf_gemma4.state_dict()
    for key, value in hf_state.items():
        new_key = f"gemma4_vlm.{key}"
        vlm_state_dict[new_key] = value

    # Load into model (strict=False because we only load VLM part, not expert)
    missing, unexpected = gemma4_with_expert.load_state_dict(vlm_state_dict, strict=False)
    logger.info(f"Loaded Gemma 4 VLM weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        logger.info(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        logger.info(f"  Unexpected keys (first 10): {unexpected[:10]}")

    # Clean up
    del hf_gemma4
    torch.cuda.empty_cache()

    return model


def main():
    parser = argparse.ArgumentParser(description="Load HuggingFace Gemma 4 weights into PI0Pytorch")
    parser.add_argument(
        "--config_name",
        type=str,
        # required=True,
        default="pi05_libero_gemma4",
        help="Training config name (e.g., pi0_aloha_sim_gemma4). Must exist in _CONFIGS.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        # required=True,
        default="checkpoints/gemma4",
        help="Output path for the PyTorch checkpoint.",
    )
    parser.add_argument(
        "--vlm_model_id",
        type=str,
        default="./models/gemma-4-E2B",
        help="HuggingFace model ID for Gemma 4 language model.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Precision for the saved model.",
    )
    args = parser.parse_args()

    # Get the training config
    train_config = _config.get_config(args.config_name)
    model_cfg = train_config.model

    logger.info(f"Creating PI0Pytorch model with config: {model_cfg}")
    model = pi0_pytorch.PI0Pytorch(model_cfg)

    # Load pretrained weights (vision tower is loaded automatically in __init__)
    load_vlm_weights_into_model(model, args.vlm_model_id)

    # Convert to target precision
    if args.precision == "bfloat16":
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model.to(torch.float32)

    # Save checkpoint
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.safetensors"
    safetensors.torch.save_model(model, str(model_path))
    logger.info(f"Saved model to {model_path}")

    # Copy norm stats from the config's assets if available
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is not None:
        logger.info(f"Note: You'll also need norm_stats for asset_id={data_config.asset_id}")
        logger.info("These should be placed in the checkpoint's assets/ directory.")

    # Print summary
    total_params = sum(p.numel() for p in model.parameters())
    vlm_params = sum(p.numel() for n, p in model.named_parameters() if "gemma4_vlm" in n or "paligemma" in n)
    expert_params = sum(p.numel() for n, p in model.named_parameters() if "gemma4_expert" in n)
    other_params = total_params - vlm_params - expert_params

    logger.info("=" * 60)
    logger.info("Parameter Summary:")
    logger.info(f"  VLM (Gemma4 + SigLIP): {vlm_params / 1e6:.1f}M")
    logger.info(f"  Action Expert (Gemma4): {expert_params / 1e6:.1f}M")
    logger.info(f"  Other (projections etc): {other_params / 1e6:.1f}M")
    logger.info(f"  Total: {total_params / 1e6:.1f}M")
    logger.info("=" * 60)
    logger.info(f"\nTo use this checkpoint, set in your training config:")
    logger.info(f'  pytorch_weight_path="{output_path}"')


if __name__ == "__main__":
    main()
