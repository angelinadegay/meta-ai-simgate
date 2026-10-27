"""The interface every simulator plugs into.

To add a simulator (Gazebo, Genesis, a CFD or FEM tool...), subclass
Simulator, set `domain` and `name`, and register it in backends/__init__.py.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from ..spec import SceneSpec


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class BuildResult:
    code: str                    # simulation source in the simulator's own language
    language: str                # e.g. "mjcf-xml", "python-openseespy"
    scene: dict                  # static description for the browser viewer
    notes: list[str] = field(default_factory=list)


class Simulator(ABC):
    domain: str
    name: str
    language: str

    @abstractmethod
    def build(self, spec: SceneSpec, code: str | None = None) -> BuildResult:
        """Create the simulation from a spec (or from code an agent wrote)."""

    @abstractmethod
    def validate(self, spec: SceneSpec, build: BuildResult) -> list[Check]:
        """Physics and sanity checks before results are shown."""

    @abstractmethod
    def run(self, spec: SceneSpec, build: BuildResult, chunk_frames: int = 30) -> Iterator[list[dict]]:
        """Run and yield chunks of frames for live streaming."""

    @abstractmethod
    def summary(self) -> dict:
        """Metrics from the last run, for the explainer agent."""
