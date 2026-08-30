"""Checkpoint helpers for the low-dimensional SRB world model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from flax import serialization

from openpi.control.srb_state import AffineStats
from openpi.models.srb_world_model import SRBWorldModel
from openpi.models.srb_world_model import SRBWorldModelConfig


@dataclass(frozen=True)
class WorldModelBundle:
    model: SRBWorldModel
    parameters: Any
    stats: dict[str, AffineStats]
    metadata: dict[str, Any]


def save_world_model(
    output_dir: str | Path,
    model_config: SRBWorldModelConfig,
    parameters: Any,
    stats: dict[str, AffineStats],
    metadata: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "params.msgpack").write_bytes(serialization.to_bytes(parameters))
    (output_dir / "model_config.json").write_text(json.dumps(model_config.to_dict(), indent=2, sort_keys=True))
    (output_dir / "stats.json").write_text(
        json.dumps({name: value.to_dict() for name, value in stats.items()}, indent=2, sort_keys=True)
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_world_model(output_dir: str | Path) -> WorldModelBundle:
    output_dir = Path(output_dir)
    config = SRBWorldModelConfig.from_dict(json.loads((output_dir / "model_config.json").read_text()))
    model = SRBWorldModel(config)
    # ``to_bytes`` serializes a plain parameter pytree, so the low-level restore
    # function is the appropriate inverse (``from_bytes`` requires a template
    # object, which is unavailable until after initialization).
    parameters = serialization.msgpack_restore((output_dir / "params.msgpack").read_bytes())
    stats_data = json.loads((output_dir / "stats.json").read_text())
    stats = {name: AffineStats.from_dict(value) for name, value in stats_data.items()}
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return WorldModelBundle(model=model, parameters=parameters, stats=stats, metadata=metadata)
