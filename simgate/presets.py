"""Reference physical values the checker validates against.

Surface gravity values (m/s^2) are from NASA planetary and satellite fact
sheets. Friction coefficients are rough engineering estimates for a rubber or
metal wheel on each surface; treat them as defaults, not measurements.
"""

GRAVITY = {
    "earth": 9.81,
    "moon": 1.62,
    "mars": 3.71,
    "titan": 1.352,
    "europa": 1.315,
    "enceladus": 0.113,
}

SURFACE_FRICTION = {
    "rock": 0.9,
    "regolith": 0.7,
    "gravel": 0.6,
    "sand": 0.5,
    "mud": 0.3,
    "ice": 0.1,
}

# Plausible ranges used by the checker.
FRICTION_RANGE = (0.02, 2.0)
SLOPE_RANGE_DEG = (0.0, 45.0)


def gravity_for(body: str) -> float | None:
    return GRAVITY.get(body.lower())


def friction_for(surface: str) -> float | None:
    return SURFACE_FRICTION.get(surface.lower())
