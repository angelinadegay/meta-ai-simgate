"""Runs one session: plan → route → build → check → (checkpoint) → run → explain.

Every step emits events, so the browser can show the agents working live.
"""
import asyncio
import uuid
from typing import Awaitable, Callable

from . import agents
from .backends.base import BuildResult, Check, Simulator, blocking_failures
from .batch import crash_report, measured
from .director import Director
from .workflows import SessionState
from .spec import SceneSpec

Emit = Callable[[dict], Awaitable[None]]


class Session:
    def __init__(self, emit: Emit) -> None:
        self.emit = emit
        self.spec: SceneSpec | None = None
        self.sim: Simulator | None = None
        self.build: BuildResult | None = None
        self.checks: list[Check] = []
        self.awaiting_approval = False
        self.library_selection: tuple[str, str, str] | None = None
        # The last run of the current scene and what was learned from it; cleared when the scene changes.
        self.last_summary: dict | None = None
        self.last_diagnosis: dict | None = None
        self.last_redesign: dict | None = None
        self._view_shows_other_scene = False   # set after "Show runs" loads the winning design's scene

    async def agent(self, name: str, status: str, message: str) -> None:
        await self.emit({"type": "agent", "agent": name, "status": status, "message": message})

    async def handle(self, text: str) -> None:
        try:
            if self.build is not None and agents.is_run_anyway(text):
                self.awaiting_approval = False
                await self.run()
            elif self.build is not None and blocking_failures(self.checks) and (
                    agents.is_rerun(text) or agents.is_approval(text)):
                failing = "; ".join(f"{c.name} ({c.detail})" for c in blocking_failures(self.checks))
                await self.emit({"type": "question",
                                 "message": f"This scene still fails: {failing}. Describe a change, or say 'run anyway' to run it.",
                                 "stress_test_available": False})
            elif self.build is not None and agents.is_rerun(text):
                self.awaiting_approval = False
                await self.run()
            elif self.build is not None and agents.is_approval(text):
                if self.awaiting_approval:
                    self.awaiting_approval = False
                    await self.run()
                else:   # e.g. a second click on Run: don't silently run the same scene twice
                    await self.emit({"type": "question",
                                     "message": "This scene already ran. Say \"run again\" to repeat it, or describe a change.",
                                     "stress_test_available": bool(self.spec and self.spec.domain == "robot_terrain")})
            elif (self.spec is not None and self.spec.domain == "robot_terrain"
                  and agents.is_stress_test(text)):
                await self.run_stress(text)
            elif self.spec is not None:
                await self.make(text, editing=True)
            else:
                await self.make(text, editing=False)
        except Exception as e:  # noqa: BLE001 - show any failure to the user
            await self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})

    async def make(self, text: str, editing: bool) -> None:
        await self.agent("Director", "working", "Planning the world operations…")
        director = Director(self.emit)
        state = SessionState(simulator=self.sim, build=self.build, checks=list(self.checks), summary=self.last_summary,
                             diagnosis=self.last_diagnosis, redesign=self.last_redesign)
        world, message = await director.execute_plan(text, self.spec if editing else None, state)
        if not editing and world.build is None:   # an answer with no scene yet: don't keep an empty one
            await self.agent("Director", "done", "Answered without building a scene.")
            await self.emit({"type": "question", "message": message, "stress_test_available": False})
            return
        spec = world.spec
        unchanged = editing and world.build is not None and world.build is self.build   # answer or analysis
        if self.library_selection and spec.domain == "robot_terrain" and not unchanged:
            self._apply_library_selection(spec)
            world.spec = spec
        self.spec, self.sim, self.build, self.checks = spec, world.simulator, world.build, list(world.checks)
        if not unchanged:
            self._forget_runs()
        if world.report is not None:
            if world.report["type"] == "diagnosis":
                self.last_diagnosis = world.report
            else:
                self.last_redesign = world.report
            await self.emit({"type": world.report["type"], "report": world.report})
        else:
            await self.emit({"type": "spec", "spec": spec.model_dump()})
        if world.build is not None and not unchanged:
            await self.emit({
                "type": "scene",
                "scene": world.build.scene,
                "code": world.build.code,
                "language": world.build.language,
            })
            await self.emit({"type": "checks", "checks": [check.__dict__ for check in world.checks]})
        if world.results is not None:
            await self.emit({"type": "metrics", "metrics": world.results})
        if not unchanged:   # an answer or analysis leaves a pending "Run it?" as it was
            self.awaiting_approval = world.question is not None and not blocking_failures(self.checks)
        await self.agent("Director", "done", spec.title + (" — " + " ".join(spec.assumptions) if spec.assumptions else ""))
        if world.run_requested and self.build is not None:
            self.awaiting_approval = False
            await self.emit({"type": "explanation", "message": message})   # a chat line without a Run chip
            await self.run()
            return
        await self.emit({
            "type": "question",
            "message": message,
            "chips": self._next_chips(world.report),
            "stress_test_available": spec.domain == "robot_terrain",
        })

    def _next_chips(self, report: dict | None) -> list[str] | None:
        """After a diagnosis or redesign, offer the next step instead of "Run"."""
        if report is None:
            return None
        if report["type"] == "diagnosis":
            if report.get("reproduced") is None:   # the robot asked about didn't fail: offer the one that did
                return self._failure_chips(self.last_summary)
            return [self._fix_chip(report.get("robot_id"))] if report.get("reproduced") else ["Run again"]
        return ["Use this design"] if report.get("winner") else ["Run again"]

    def _racer_label(self, robot_id: str | None) -> str | None:
        """A race robot's label, for chips that name it; None in a single-robot scene."""
        if self.spec is None or len(self.spec.robots) < 2:
            return None
        return next((entry.label for entry in self.spec.robots if entry.id == robot_id), None)

    def _fix_chip(self, robot_id: str | None) -> str:
        label = self._racer_label(robot_id)
        return f"Make the {label} pass" if label else "Make it pass"

    def _failure_chips(self, summary: dict | None) -> list[str] | None:
        """After a run in which a robot failed: find out why, or make it pass (in a race, naming that robot)."""
        robot_id = SessionState(summary=summary).failed_robot()
        if robot_id is None:
            return None
        label = self._racer_label(robot_id)
        return [f"Why did the {label} fail?" if label else "Why did it fail?", self._fix_chip(robot_id)]

    async def show_design(self) -> None:
        """Replay the last redesign's winning design with streamed frames, without applying it."""
        try:
            winner = (self.last_redesign or {}).get("winner")
            if self.spec is None or winner is None or self.last_summary is None:
                await self.emit({"type": "question", "message": "There's no passing design to show yet. Ask me to make it pass first."})
                return
            from .batch import _validate_variation
            from .diagnose import pin_task

            variation = {"robot": winner["changes"]}
            if self.last_redesign.get("robot_id"):
                variation["robot_id"] = self.last_redesign["robot_id"]
            _, spec, _ = _validate_variation(pin_task(self.spec, self.last_summary), variation)
            sim = agents.route(spec)
            build = await asyncio.to_thread(sim.build, spec)
            await self.emit({"type": "scene", "scene": build.scene, "code": build.code, "language": build.language})
            self._view_shows_other_scene = True
            await self.agent("Simulator", "working", f"Replaying the winning design: {winner['description']}…")
            await self._stream(spec, sim, build)
            await self.agent("Simulator", "done", "Replay finished; the scene itself is unchanged.")
            await self.emit({"type": "explanation",
                             "message": f"That was the winning design ({winner['description']}), not applied yet. "
                                        "Say \"use this design\" to apply it."})
        except Exception as e:  # noqa: BLE001
            await self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})

    async def _stream(self, spec: SceneSpec, sim: Simulator, build: BuildResult) -> dict:
        """Run a scene, streaming run_start, frames and metrics to the browser; returns the metrics."""
        await self.emit({"type": "run_start", "duration_s": spec.duration_s})
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            for chunk in sim.run(spec, build):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, produce)
        while (chunk := await queue.get()) is not None:
            await self.emit({"type": "frames", "frames": chunk})
        await task
        metrics = sim.summary()
        await self.emit({"type": "metrics", "metrics": metrics})
        return metrics

    def _forget_runs(self) -> None:
        self.last_summary = self.last_diagnosis = self.last_redesign = None

    def _apply_library_selection(self, spec: SceneSpec) -> None:
        from .library import LibraryRegistry
        from .spec import Robot, Terrain

        assert self.library_selection is not None
        robot_id, policy_id, world_id = self.library_selection
        registry = LibraryRegistry()
        robot_card = registry.robot(robot_id)
        policy = registry.policy(policy_id)
        world = next((item for item in registry.worlds() if item.id == world_id), None)
        if world is None:
            raise ValueError(f"Unknown world library id: {world_id}")
        if policy not in registry.compatible_policies(robot_id):
            raise ValueError(f"Policy {policy_id!r} is not compatible with robot {robot_id!r}.")

        if robot_card.source == "builtin":
            kind = "rover" if robot_card.id == "rover" else "walker"
            spec.robot = Robot(kind=kind, mass_kg=robot_card.mass_kg)
        else:
            spec.robot = Robot(
                kind="library",
                robot_id=robot_card.id,
                policy_id=policy.id,
                mass_kg=robot_card.mass_kg,
            )
        if spec.robots:
            spec.robots[0].robot = spec.robot.model_copy(deep=True)
        spec.environment.body = world.body
        spec.environment.gravity = world.gravity_m_s2
        spec.environment.surface = world.surface
        spec.environment.friction = world.friction
        if world.terrain_kind == "real":
            spec.terrain = Terrain(kind="real", preset=world.terrain_preset)
        spec.title = f"{robot_card.name} on {spec.terrain.kind if spec.terrain else 'generated'} terrain, {world.body.title()}"
        spec.fill_defaults()

    async def select_library(self, robot_id: str, policy_id: str, world_id: str) -> None:
        from .library import LibraryRegistry

        registry = LibraryRegistry()
        robot = registry.robot(robot_id)
        policy = registry.policy(policy_id)
        world = next((item for item in registry.worlds() if item.id == world_id), None)
        if world is None:
            raise ValueError(f"Unknown world library id: {world_id}")
        if policy not in registry.compatible_policies(robot_id):
            raise ValueError(f"Policy {policy_id!r} is not compatible with robot {robot_id!r}.")
        self.library_selection = (robot.id, policy.id, world.id)
        await self.emit({
            "type": "library_selected",
            "robot": robot.name,
            "policy": policy.name,
            "world": world.name,
        })
        if self.spec is not None and self.spec.domain == "robot_terrain":
            spec = self.spec.model_copy(deep=True)
            spec.assumptions = []
            self._apply_library_selection(spec)
            await self.build_and_check(spec)

    async def set_point(self, which: str, x: float, y: float) -> None:
        """Start or goal picked by clicking on the terrain: changes the spec directly, no model call needed."""
        try:
            if self.spec is None or self.spec.domain != "robot_terrain":
                await self.emit({"type": "question", "message": "Describe a robot scene first, then click the terrain to set its start or goal."})
                return
            from .tools import World

            world = World(self.spec.model_copy(deep=True))
            director = Director(self.emit)
            director._root_id = director._id()
            await self.emit({
                "type": "step", "id": director._root_id, "parent_id": None,
                "agent": "Director", "model": "rules", "tool": None, "args": {},
                "thought": f"Applying the map click as a {which} position.",
                "result": "", "status": "working", "ms": 0,
            })
            if which == "start":
                name = "set_start"
                args = {"robot_id": world.spec.robots[0].id, "xy": [x, y]}
            else:
                name = "set_goal"
                args = {"robot_id": "all", "xy": [x, y]}
            await director._mock_call(world, name, args, f"Setting the {which} from the map.")
            await director._mock_call(world, "build_and_check", {}, "Rebuilding and checking the clicked position.")
            await director._finish_root(f"{which.capitalize()} set to ({x:.1f}, {y:.1f}) m.")
            self.spec, self.sim, self.build, self.checks = world.spec, world.simulator, world.build, list(world.checks)
            self._forget_runs()
            await self.emit({"type": "spec", "spec": self.spec.model_dump()})
            if self.build is not None:
                await self.emit({"type": "scene", "scene": self.build.scene, "code": self.build.code, "language": self.build.language})
                await self.emit({"type": "checks", "checks": [check.__dict__ for check in world.checks]})
            self.awaiting_approval = not blocking_failures(self.checks)
            await self.emit({"type": "question", "message": "Does this look right? Say \"run\" to start, or tell me what to change.",
                             "stress_test_available": True})
        except Exception as e:  # noqa: BLE001
            await self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})

    async def build_and_check(self, spec: SceneSpec) -> None:
        self.spec = spec
        self._forget_runs()
        await self.emit({"type": "spec", "spec": spec.model_dump()})
        sim = agents.route(spec)
        if self.sim is None or self.sim.name != sim.name:
            self.sim = sim
        await self.agent("Router", "done", f"This is a {spec.domain.replace('_', ' ')} question, so I'm using {sim.name}.")

        await self.agent("Builder", "working", f"Writing {sim.language} for {sim.name}…")
        logs: list[str] = []
        self.build = await asyncio.to_thread(agents.build, self.sim, spec, logs.append)
        for line in logs:
            await self.agent("Builder", "working", line)
        await self.emit({"type": "scene", "scene": self.build.scene, "code": self.build.code, "language": self.build.language})
        await self.agent("Builder", "done", f"Scene built ({len(self.build.code.splitlines())} lines of {self.build.language}).")

        await self.agent("Checker", "working", "Verifying physics against reference values…")
        checks = await asyncio.to_thread(self.sim.validate, spec, self.build)
        await self.emit({"type": "checks", "checks": [c.__dict__ for c in checks]})
        failed = [c for c in checks if not c.passed]
        if failed:
            await self.agent("Checker", "warning", "Some checks failed: " + "; ".join(f"{c.name} ({c.detail})" for c in failed))
        else:
            await self.agent("Checker", "done", f"All {len(checks)} checks passed.")

        self.checks = checks
        self.awaiting_approval = not blocking_failures(checks)
        await self.emit({"type": "question", "message": "Does this look right? Say \"run\" to start, or tell me what to change.",
                         "stress_test_available": spec.domain == "robot_terrain"})

    async def run_stress(self, text: str) -> None:
        assert self.spec is not None and self.spec.domain == "robot_terrain"
        from .batch import run_batch

        spec = self.spec.model_copy(deep=True)
        await self.agent("Stress planner", "working", "Choosing conditions for this robot scene…")
        variations = await asyncio.to_thread(agents.plan_stress, spec, text)
        used_fallback = bool(variations) and all(
            variation.get("label", "").startswith("Fallback: ") for variation in variations
        )
        labels = ["Baseline"] + [
            f"{variation.get('label', 'Variation')}: "
            + ", ".join(f"{key}={value}" for key, value in variation.items() if key != "label")
            for variation in variations
        ]
        notice = ("The model suggested conditions that don't fit this scene; using compatible defaults. "
                  if used_fallback else "")
        await self.agent("Stress planner", "done", notice + "Planned " + "; ".join(labels) + ".")

        total = min(len(variations) + 1, 24)
        await self.emit({"type": "batch_start", "total": total})
        await self.agent("Batch runner", "working", f"0/{total} done")
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_result(result: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, result)

        def execute() -> list[dict]:
            try:
                return run_batch(spec, variations, on_result)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(execute))
        done = 0
        while (result := await queue.get()) is not None:
            done += 1
            await self.emit({"type": "batch_result", "result": result})
            await self.agent("Batch runner", "working", f"{done}/{total} done")
        results = await task
        passed = sum(bool(result["passed"]) for result in results)
        await self.emit({"type": "batch_done", "total": len(results), "passed": passed,
                         "failed": len(results) - passed})
        await self.agent("Batch runner", "done", f"{len(results)}/{len(results)} done.")
        rows = []
        for result in results:
            summary = result.get("summary") or {}
            row = {"id": result["id"], "label": result["label"], "passed": result["passed"],
                   "reason": result["reason"], "metrics": self._row_metrics(summary),
                   "ghost_status": result.get("ghost_status")}
            if len(spec.robots) > 1:   # a race: one column per robot
                row["robots"] = {robot_id: {**robot, "metrics": self._row_metrics((summary.get("robots") or {}).get(robot_id) or {})}
                                 for robot_id, robot in (result.get("robots") or {}).items()}
            rows.append(row)
        columns = [{"id": entry.id, "label": entry.label, "kind": entry.robot.kind} for entry in spec.robots]
        await self.emit({"type": "report", "rows": rows, "passed": passed, "total": len(results),
                         **({"robots": columns} if len(spec.robots) > 1 else {})})
        crashed = crash_report(results)
        if crashed:   # a system problem, not physics: shown as a warning and kept out of the analysis
            await self.emit({"type": "step", "id": f"crash-{uuid.uuid4().hex[:8]}", "parent_id": None,
                             "agent": "Batch runner", "model": "ProcessPoolExecutor", "tool": "crashed", "args": {},
                             "thought": "Checking every run completed.", "result": crashed, "status": "warning", "ms": 0})
            if not measured(results):
                await self.emit({"type": "explanation", "message": agents.explain_stress(spec, results)})
                return
        await self.agent("Explainer", "working", "Finding the pattern across the stress runs…")
        explanation = await asyncio.to_thread(agents.explain_stress, spec, results)
        await self.agent("Explainer", "done", "Stress-test pattern summarized.")
        await self.emit({"type": "explanation", "message": explanation})

    @staticmethod
    def _row_metrics(summary: dict) -> dict:
        """The key numbers for one stress-test cell (a single robot's summary or one racer's metrics)."""
        metrics = {}
        if summary.get("time_to_goal_s") is not None:
            metrics["time_to_goal_s"] = summary["time_to_goal_s"]
        elif summary.get("final_distance_m") is not None:
            metrics["final_distance_m"] = summary["final_distance_m"]
        elif summary.get("forward_progress_m") is not None:
            metrics["forward_progress_m"] = summary["forward_progress_m"]
        for key in ("max_tilt_deg", "slip_ratio"):
            if summary.get(key) is not None:
                metrics[key] = summary[key]
        return metrics

    async def run(self) -> None:
        assert self.spec and self.sim and self.build
        if self._view_shows_other_scene:   # "Show runs" left another design in the view
            await self.emit({"type": "scene", "scene": self.build.scene, "code": self.build.code, "language": self.build.language})
            self._view_shows_other_scene = False
        await self.agent("Simulator", "working", f"Running {self.spec.duration_s:.0f} s in {self.sim.name}…")
        metrics = await self._stream(self.spec, self.sim, self.build)
        self.last_summary, self.last_diagnosis, self.last_redesign = metrics, None, None
        await self.agent("Simulator", "done", "Run finished.")
        await self.agent("Explainer", "working", "Explaining what happened…")
        text = await asyncio.to_thread(agents.explain, self.spec, metrics)
        await self.agent("Explainer", "done", "Done.")
        chips = self._failure_chips(metrics) if self.spec.domain == "robot_terrain" else None
        await self.emit({"type": "explanation", "message": text, **({"chips": chips} if chips else {})})
