# AGENTS.md — ACoT-VLA (OpenPI fork)

## Quick Commands

```bash
# Install (must use GIT_LFS_SKIP_SMUDGE to avoid downloading large model files)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# Lint + format
uv run ruff check --fix src/ scripts/ packages/
uv run ruff format src/ scripts/ packages/

# Test (skips @manual tests; conftest.py auto-falls back to CPU JAX if no GPU)
uv run pytest --strict-markers -m "not manual"

# Training pipeline (order matters)
uv run python scripts/compute_norm_stats.py --config-name <CONFIG_NAME>
bash scripts/train.sh <CONFIG_NAME> <EXP_NAME>
bash scripts/server.sh <GPU_ID> <PORT>
```

## Architecture

- **Framework**: JAX/Flax (primary), PyTorch used for checkpoint loading only
- **Package manager**: uv (Python 3.12)
- **Build**: hatchling
- **Source layout**: `src/openpi/` is the installable package; `packages/openpi-client/` is a separate lightweight client package
- **Models**: `src/openpi/models/` — `pi0.py`, `pi0_fast.py` (from OpenPI upstream), `acot_vla.py` (ACoT-VLA extension with EAR/IAR reasoning modules)
- **Configs**: `src/openpi/training/config.py` — all `TrainConfig` objects live in `_CONFIGS` list, keyed by `name` field
- **Key model types**: PI0, PI05, PI0_FAST, and ACoT-VLA variants (ACOT_VLA_PI0, ACOT_VLA_PI05)
- **Data transforms**: `src/openpi/transforms.py` — shared transform pipeline; robot-specific transforms in `src/openpi/policies/<robot>_policy.py`
- **Third-party submodules**: `third_party/aloha`, `third_party/libero` (init with `git submodule update --init --recursive`)

## Config Names Worth Knowing

| Config | Purpose |
|---|---|
| `srb_train` | SRB LoRA training (gemma_2b_lora + gemma_300m_lora), used by `compute_norm_stats.py` default |
| `pi05_srb` / `pi0_fast_srb` | SRB full-finetune templates |
| `debug` / `debug_pi05` | Minimal fake-data configs for quick iteration |
| `acot_icra_simulation_challenge_reasoning_to_action` | AgiBot World Challenge baseline (referenced in `serve_policy.py` and `openloop.py`) |

Add new configs to `_CONFIGS` in `config.py`; they auto-register and become usable by name.

## Gotchas

- **`GIT_LFS_SKIP_SMUDGE=1`** is mandatory for `uv sync` and `uv pip install` — without it uv tries to download large model checkpoint files via LFS
- **RLDS DataLoader** (DROID configs) requires `num_workers=0` — it handles multiprocessing internally
- **`conftest.py`** at `src/openpi/conftest.py` sets `JAX_PLATFORMS=cpu` when no GPU is detected; tests run fine on CPU-only machines
- **ruff excludes** `docker/` and `third_party/` — don't lint inside those directories
- **`force-single-line` isort** — imports are sorted one-per-line; `collections.abc`, `typing`, `typing_extensions` are exceptions
- **`assets/`, `checkpoints/`, `data/`** are all gitignored — norm stats and checkpoints live outside version control
- **SRB state dimension**: model internal state dim is 32 for pi05; if your concatenated SRB state exceeds 32 dims it gets truncated (set `strict_state_dim=True` in `SRBDataConfig` to error instead)
- **`scripts/compute_norm_stats.py`** has a hardcoded `config_name="srb_train"` at the bottom — override via CLI arg `--config-name` for other configs
- **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`** is set by `train.sh`; `server.sh` uses 0.9 and also sets `XLA_PYTHON_CLIENT_ALLOCATOR=platform`

## Testing Notes

- Tests live alongside source: `src/openpi/**/*_test.py`
- Test paths configured in `pyproject.toml`: `testpaths = ["src", "scripts", "packages"]`
- The `@manual` pytest marker gates expensive/integration tests; CI runs `-m "not manual"`
- CI installs ffmpeg (`sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev`) before running tests

## Pre-commit

Configured in `.pre-commit-config.yaml`:
1. `uv-lock` — keeps `uv.lock` in sync
2. `ruff --fix` — lints with auto-fix
3. `ruff-format` — formats

Excludes `third_party/`.
