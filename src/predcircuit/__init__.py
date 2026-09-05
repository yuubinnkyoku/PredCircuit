"""PredCircuit: predictive coding on arbitrary neural circuit graphs."""

from .model import PredictiveCodingGraph
from .topology import CircuitGraph

__all__ = ["CircuitGraph", "PredictiveCodingGraph"]
__version__ = "0.1.0"
