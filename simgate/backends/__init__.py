"""Simulator registry: the router picks a backend by the spec's domain."""
from .base import Simulator
from .mujoco_backend import MuJoCoSimulator
from .opensees_backend import OpenSeesSimulator

REGISTRY: dict[str, type[Simulator]] = {
    MuJoCoSimulator.domain: MuJoCoSimulator,
    OpenSeesSimulator.domain: OpenSeesSimulator,
}


def describe() -> str:
    """Short catalog the planner sees, so it knows what each simulator is for."""
    return "\n".join(
        [
            "- robot_terrain → MuJoCo: robots/vehicles moving over terrain; motion, contact, traction, tipping, gravity on other worlds.",
            "- structure → OpenSees: buildings/frames under earthquake shaking; drift, period, resonance, damage level.",
        ]
    )
