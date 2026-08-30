"""Lightweight control utilities for physics-aware SRB experiments.

Imports are lazy on purpose.  The SRB collector must initialize Isaac Sim before
loading JAX, and importing ``openpi.control.exploration`` should not allocate a
JAX backend just because this package initializer ran.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openpi.control.cem_planner import CEMConfig
    from openpi.control.cem_planner import CEMPlanner
    from openpi.control.exploration import SmoothExplorationConfig
    from openpi.control.exploration import SmoothRandomExplorer
    from openpi.control.srb_state import AffineStats
    from openpi.control.srb_state import FieldSpec
    from openpi.control.srb_state import ObservationSchema
    from openpi.control.srb_state import PhysicsNormalizer
    from openpi.control.transition_dataset import TransitionDataset

__all__ = [
    "AffineStats",
    "CEMConfig",
    "CEMPlanner",
    "FieldSpec",
    "ObservationSchema",
    "PhysicsNormalizer",
    "SmoothExplorationConfig",
    "SmoothRandomExplorer",
    "TransitionDataset",
]


_LAZY_IMPORTS = {
    "AffineStats": ("openpi.control.srb_state", "AffineStats"),
    "CEMConfig": ("openpi.control.cem_planner", "CEMConfig"),
    "CEMPlanner": ("openpi.control.cem_planner", "CEMPlanner"),
    "FieldSpec": ("openpi.control.srb_state", "FieldSpec"),
    "ObservationSchema": ("openpi.control.srb_state", "ObservationSchema"),
    "PhysicsNormalizer": ("openpi.control.srb_state", "PhysicsNormalizer"),
    "SmoothExplorationConfig": ("openpi.control.exploration", "SmoothExplorationConfig"),
    "SmoothRandomExplorer": ("openpi.control.exploration", "SmoothRandomExplorer"),
    "TransitionDataset": ("openpi.control.transition_dataset", "TransitionDataset"),
}


def __getattr__(name: str):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    attribute = getattr(import_module(module_name), attribute_name)
    globals()[name] = attribute
    return attribute
