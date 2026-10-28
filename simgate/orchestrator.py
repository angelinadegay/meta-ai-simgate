"""Runs one session: plan → route → build → check → (checkpoint) → run → explain.

Every step emits events, so the browser can show the agents working live.
"""
import asyncio
from typing import Awaitable, Callable

from . import agents
from .backends.base import BuildResult, Simulator
from .director import Director
from .spec import SceneSpec

Emit = Callable[[dict], Awaitable[None]]


class Session:
    def __init__(self, emit: Emit) -> None:
        self.emit = emit
        self.spec: SceneSpec | None = None
        self.sim: Simulator | None = None
        self.build: BuildResult | None = None
        self.awaiting_approval = False
        self.library_selection: tuple[str, str, str] | None = None

    async def agent(self, name: str, status: str, message: str) -> None:
        await self.emit({"type": "agent", "agent": name, "status": status, "message": message})

    async def handle(self, text: str) -> None:
        try:
            if self.build is not None and agents.is_rerun(text):
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
        world, message = await director.execute_plan(text, self.spec if editing else None)
        if not editing and world.build is None:   # an answer with no scene yet: don't keep an empty one
            await self.agent("Director", "done", "Answered without building a scene.")
            await self.emit({"type": "question", "message": message, "stress_test_available": False})
            return
        spec = world.spec
        if self.library_selection and spec.domain == "robot_terrain":
            self._apply_library_selection(spec)
            world.spec = spec
        self.spec, self.sim, self.build = spec, world.simulator, world.build
        await self.emit({"type": "spec", "spec": spec.model_dump()})
        if world.build is not None:
            await self.emit({
                "type": "scene",
                "scene": world.build.scene,
                "code": world.build.code,
                "language": world.build.language,
            })
            await self.emit({"type": "checks", "checks": [check.__dict__ for check in world.checks]})
        if world.results is not None:
            await self.emit({"type": "metrics", "metrics": world.results})
        self.awaiting_approval = world.question is not None
        await self.agent("Director", "done", spec.title + (" — " + " ".join(spec.assumptions) if spec.assumptions else ""))
        await self.emit({
            "type": "question",
            "message": message,
            "stress_test_available": spec.domain == "robot_terrain",
        })

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
            self.spec, self.sim, self.build = world.spec, world.simulator, world.build
            await self.emit({"type": "spec", "spec": self.spec.model_dump()})
            if self.build is not None:
                await self.emit({"type": "scene", "scene": self.build.scene, "code": self.build.code, "language": self.build.language})
                await self.emit({"type": "checks", "checks": [check.__dict__ for check in world.checks]})
            self.awaiting_approval = True
            await self.emit({"type": "question", "message": "Does this look right? Say \"run\" to start, or tell me what to change.",
                             "stress_test_available": True})
        except Exception as e:  # noqa: BLE001
            await self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})

    async def build_and_check(self, spec: SceneSpec) -> None:
        self.spec = spec
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

        self.awaiting_approval = True
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
            summary = result.get("summary", {})
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
            rows.append({"id": result["id"], "label": result["label"], "passed": result["passed"],
                         "reason": result["reason"], "metrics": metrics,
                         "ghost_status": result.get("ghost_status")})
        await self.emit({"type": "report", "rows": rows, "passed": passed, "total": len(results)})
        await self.agent("Explainer", "working", "Finding the pattern across the stress runs…")
        explanation = await asyncio.to_thread(agents.explain_stress, spec, results)
        await self.agent("Explainer", "done", "Stress-test pattern summarized.")
        await self.emit({"type": "explanation", "message": explanation})

    async def run(self) -> None:
        assert self.spec and self.sim and self.build
        await self.agent("Simulator", "working", f"Running {self.spec.duration_s:.0f} s in {self.sim.name}…")
        await self.emit({"type": "run_start", "duration_s": self.spec.duration_s})
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            for chunk in self.sim.run(self.spec, self.build):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        task = loop.run_in_executor(None, produce)
        while (chunk := await queue.get()) is not None:
            await self.emit({"type": "frames", "frames": chunk})
        await task

        metrics = self.sim.summary()
        await self.emit({"type": "metrics", "metrics": metrics})
        await self.agent("Simulator", "done", "Run finished.")
        await self.agent("Explainer", "working", "Explaining what happened…")
        text = await asyncio.to_thread(agents.explain, self.spec, metrics)
        await self.agent("Explainer", "done", "Done.")
        await self.emit({"type": "explanation", "message": text})
