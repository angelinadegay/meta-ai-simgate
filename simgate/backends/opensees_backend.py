"""OpenSees backend: simple 2D steel moment frames under earthquake shaking.

Linear-elastic model (no yielding), lumped floor masses, 5% Rayleigh damping
and a synthetic sinusoidal ground motion. Good for comparing designs and
teaching; not a substitute for a real structural analysis.
"""
import math
from typing import Iterator

import numpy as np
import openseespy.opensees as ops

from ..spec import SceneSpec
from .base import BuildResult, Check, Simulator

E_STEEL = 200e9                        # Pa
SECTIONS = {                           # rough W-shape properties: area m², inertia m⁴
    "light": (0.0065, 1.2e-4),
    "medium": (0.0120, 3.0e-4),
    "heavy": (0.0200, 7.0e-4),
}
FPS = 30
SUBSTEPS = 4                           # analysis steps per recorded frame
G = 9.81                               # ground acceleration is given in Earth g


def frame_geometry(spec: SceneSpec):
    s = spec.structure
    nodes, columns, beams = {}, [], []
    tag = 1
    grid = {}
    for floor in range(s.stories + 1):
        for col in range(s.bays + 1):
            nodes[tag] = (col * s.bay_width_m, floor * s.story_height_m)
            grid[(floor, col)] = tag
            tag += 1
    for floor in range(s.stories):
        for col in range(s.bays + 1):
            columns.append((grid[(floor, col)], grid[(floor + 1, col)]))
    for floor in range(1, s.stories + 1):
        for col in range(s.bays):
            beams.append((grid[(floor, col)], grid[(floor, col + 1)]))
    return nodes, columns, beams, grid


def ground_motion(load, dt: float) -> np.ndarray:
    n = int(load.duration_s / dt)
    t = np.arange(n) * dt
    envelope = np.clip(t / 1.0, 0, 1) * np.clip((load.duration_s - t) / 2.0, 0, 1)
    return load.pga_g * np.sin(2 * math.pi * load.freq_hz * t) * envelope


def opensees_script(spec: SceneSpec) -> str:
    """Readable OpenSeesPy source for the build, shown to curious users."""
    s, ld = spec.structure, spec.load
    A, I = SECTIONS[s.section]
    return f'''# Generated OpenSeesPy model: {spec.title}
import openseespy.opensees as ops
ops.wipe(); ops.model("basic", "-ndm", 2, "-ndf", 3)
# {s.stories}-story, {s.bays}-bay steel moment frame
# story height {s.story_height_m} m, bay width {s.bay_width_m} m
# section "{s.section}": A={A} m^2, I={I} m^4, E={E_STEEL:.0e} Pa
# floor mass {s.floor_mass_t} t, 5% Rayleigh damping
# ground motion: {ld.pga_g} g peak at {ld.freq_hz} Hz for {ld.duration_s} s
'''


class OpenSeesSimulator(Simulator):
    domain = "structure"
    name = "OpenSees"
    language = "python-openseespy"

    def __init__(self) -> None:
        self._summary: dict = {}

    # ── model ─────────────────────────────────────────────────────────────
    def _model(self, spec: SceneSpec) -> tuple[dict, dict]:
        s = spec.structure
        A, I = SECTIONS[s.section]
        nodes, columns, beams, grid = frame_geometry(spec)
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        for tag, (x, y) in nodes.items():
            ops.node(tag, x, y)
            if y == 0:
                ops.fix(tag, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        etag = 1
        for i, j in columns:
            ops.element("elasticBeamColumn", etag, i, j, A, E_STEEL, I, 1)
            etag += 1
        for i, j in beams:
            ops.element("elasticBeamColumn", etag, i, j, A, E_STEEL, 1.2 * I, 1)
            etag += 1
        m_node = s.floor_mass_t * 1000 / (s.bays + 1)
        for floor in range(1, s.stories + 1):
            for col in range(s.bays + 1):
                ops.mass(grid[(floor, col)], m_node, m_node, 0.0)
        return nodes, grid

    def _periods(self, n: int) -> list[float]:
        try:
            eig = ops.eigen(n)                       # default sparse solver; fast
            if not eig or any(w2 <= 0 for w2 in eig):
                raise RuntimeError
        except Exception:  # noqa: BLE001 - tiny models can trip the sparse solver
            eig = ops.eigen("-fullGenLapack", n)
        return [2 * math.pi / math.sqrt(w2) for w2 in eig if w2 > 0]

    # ── interface ─────────────────────────────────────────────────────────
    def build(self, spec: SceneSpec, code: str | None = None, view: bool = True) -> BuildResult:
        nodes, _ = self._model(spec)
        _, columns, beams, _ = frame_geometry(spec)
        scene = {
            "backend": self.name,
            "kind": "frame",
            "nodes": {str(k): v for k, v in nodes.items()},
            "members": [[i, j, "column"] for i, j in columns] + [[i, j, "beam"] for i, j in beams],
        }
        return BuildResult(code=code or opensees_script(spec), language=self.language, scene=scene)

    def validate(self, spec: SceneSpec, build: BuildResult) -> list[Check]:
        checks = []
        s = spec.structure
        A, I = SECTIONS[s.section]

        # 1) Textbook check: cantilever tip deflection = P L^3 / (3 E I)
        L, P = s.story_height_m, 10_000.0
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0, 0); ops.node(2, 0, L); ops.fix(1, 1, 1, 1)
        ops.geomTransf("Linear", 1)
        ops.element("elasticBeamColumn", 1, 1, 2, A, E_STEEL, I, 1)
        ops.timeSeries("Linear", 1); ops.pattern("Plain", 1, 1); ops.load(2, P, 0, 0)
        ops.system("BandGeneral"); ops.numberer("RCM"); ops.constraints("Plain")
        ops.integrator("LoadControl", 1.0); ops.algorithm("Linear"); ops.analysis("Static")
        ops.analyze(1)
        got = ops.nodeDisp(2, 1)
        expected = P * L**3 / (3 * E_STEEL * I)
        err = abs(got - expected) / expected
        checks.append(Check("Solver matches textbook", err < 0.01,
                            f"Cantilever deflection {got*1000:.3f} mm vs PL³/3EI = {expected*1000:.3f} mm ({err*100:.2f}% off)."))

        # 2) Period sanity check vs building-code estimate (ASCE 7: Ta = 0.0724 h^0.8, steel moment frame)
        self._model(spec)
        T1 = self._periods(1)[0]
        h = s.stories * s.story_height_m
        Ta = 0.0724 * h**0.8
        ok = 0.4 * Ta <= T1 <= 3.0 * Ta
        checks.append(Check("Period is realistic", ok,
                            f"First-mode period {T1:.2f} s vs code estimate {Ta:.2f} s for a {h:.0f} m steel frame."))
        self._T1, self._Ta = T1, Ta

        ld = spec.load
        checks.append(Check("Shaking is plausible", 0 < ld.pga_g <= 2.0 and 0.1 <= ld.freq_hz <= 20,
                            f"Peak ground acceleration {ld.pga_g} g at {ld.freq_hz} Hz."))
        return checks

    def run(self, spec: SceneSpec, build: BuildResult, chunk_frames: int = 30, record: bool = True) -> Iterator[list[dict]]:
        s, ld = spec.structure, spec.load
        nodes, grid = self._model(spec)
        periods = self._periods(min(2, s.stories * (s.bays + 1)))
        w1 = 2 * math.pi / periods[0]
        w2 = 2 * math.pi / periods[1] if len(periods) > 1 else 3 * w1
        zeta = 0.05
        a0, a1 = zeta * 2 * w1 * w2 / (w1 + w2), zeta * 2 / (w1 + w2)
        ops.rayleigh(a0, 0.0, a1, 0.0)

        dt = 1.0 / (FPS * SUBSTEPS)
        acc = ground_motion(ld, dt)
        ops.timeSeries("Path", 2, "-dt", dt, "-values", *acc.tolist(), "-factor", G)
        ops.pattern("UniformExcitation", 2, 1, "-accel", 2)
        ops.constraints("Plain"); ops.numberer("RCM"); ops.system("BandGeneral")
        ops.test("NormDispIncr", 1e-8, 20); ops.algorithm("Newton")
        ops.integrator("Newmark", 0.5, 0.25); ops.analysis("Transient")

        roof = [grid[(s.stories, c)] for c in range(s.bays + 1)]
        max_roof, max_drift, frames = 0.0, 0.0, []
        n_frames = len(acc) // SUBSTEPS
        for f in range(n_frames):
            for _ in range(SUBSTEPS):
                if ops.analyze(1, dt) != 0:
                    raise RuntimeError("OpenSees failed to converge.")
            disp = {tag: (ops.nodeDisp(tag, 1), ops.nodeDisp(tag, 2)) for tag in nodes}
            max_roof = max(max_roof, max(abs(disp[t][0]) for t in roof))
            for floor in range(1, s.stories + 1):
                ux = disp[grid[(floor, 0)]][0]
                ux_below = disp[grid[(floor - 1, 0)]][0]
                max_drift = max(max_drift, abs(ux - ux_below) / s.story_height_m)
            frames.append({"t": round(f / FPS, 3),
                           "ground": round(float(acc[f * SUBSTEPS]), 4),
                           "disp": {str(k): [round(v[0], 6), round(v[1], 6)] for k, v in disp.items()}})
            if len(frames) >= chunk_frames:
                yield frames
                frames = []
        if frames:
            yield frames

        drift_pct = max_drift * 100
        if drift_pct < 0.5:
            damage = "little to no damage"
        elif drift_pct < 1.5:
            damage = "moderate damage (cracked finishes, some repairs)"
        elif drift_pct < 2.5:
            damage = "extensive damage"
        else:
            damage = "severe damage, possible collapse in a real building"
        self._summary = {
            "simulator": self.name,
            "stories": s.stories,
            "height_m": s.stories * s.story_height_m,
            "period_s": round(periods[0], 3),
            "code_period_estimate_s": round(0.0724 * (s.stories * s.story_height_m) ** 0.8, 3),
            "shaking_freq_hz": ld.freq_hz,
            "building_freq_hz": round(1 / periods[0], 3),
            "pga_g": ld.pga_g,
            "max_roof_displacement_cm": round(max_roof * 100, 2),
            "max_story_drift_pct": round(drift_pct, 3),
            "damage_estimate": damage,
            "model_limits": "Linear-elastic steel frame, no yielding or gravity (P-delta) effects; damage levels are approximate drift-based estimates.",
        }

    def summary(self) -> dict:
        return self._summary
