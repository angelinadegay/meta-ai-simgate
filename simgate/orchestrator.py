"""Runs one session: plan → route → build → check → (checkpoint) → run → explain.

Every step emits events, so the browser can show the agents working live.
"""
import asyncio
import re
import uuid
from typing import Awaitable, Callable

from . import agents, cost
from .backends.base import BuildResult, Check, Simulator, blocking_failures
from .batch import crash_report, measured
from .director import FOLLOW_UP_WORDS, Director, follow_up_plan, real_terrain_body_change
from .facts import facts as scene_facts
from .next_actions import next_actions
from .plan import Plan
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
        self.last_finder_report: dict | None = None
        self._finder_source_spec: SceneSpec | None = None
        self._view_shows_other_scene = False   # set after "Show runs" loads the winning design's scene
        self.pending_real_terrain_change: dict[str, str] | None = None

    async def agent(self, name: str, status: str, message: str) -> None:
        await self.emit({"type": "agent", "agent": name, "status": status, "message": message})

    async def handle(self, text: str) -> None:
        """One chat request, with what it cost (runs, wall time, model calls) sent after its result."""
        with cost.track() as spent:
            await self._handle(text)
            if spent.runs or spent.model_calls:
                await self.emit({"type": "cost", "cost": spent.snapshot()})

    async def _handle(self, text: str) -> None:
        try:
            if self.pending_real_terrain_change is not None:
                await self._handle_real_terrain_choice(text)
                return
            elif (change := real_terrain_body_change(text, self.spec)) is not None:
                self.pending_real_terrain_change = change
                await self._ask_real_terrain_choice()
            elif self.spec is not None and re.search(r"\bstart on flatter ground\b", text, re.I):
                await self._start_on_flatter_ground()
            elif self.spec is not None and re.search(r"\bfind a gentler route\b", text, re.I):
                await self._find_a_gentler_route()
            elif (self.spec is not None and self.spec.domain == "robot_terrain"
                    and agents.is_stress_repeat(text)):   # "repeat ×3" is a repeated stress test, not a rerun
                await self.run_stress(text)
            elif self.build is not None and agents.is_run_anyway(text):
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

    async def _ask_real_terrain_choice(self) -> None:
        assert self.pending_real_terrain_change is not None and self.spec is not None
        pending = self.pending_real_terrain_change
        reason = (f"{pending['label']} is real {pending['current_body'].title()} terrain. Do you want "
                  f"{pending['target_body'].title()} gravity on this same terrain as a what-if, "
                  f"{pending['target_body'].title()} terrain instead, or Keep {pending['current_body'].title()}?")
        target = pending["target_body"].title()
        current = pending["current_body"].title()
        options = [
            {"label": f"{target} gravity on this same terrain (a what-if)",
             "message": f"{target} gravity on this same terrain"},
            {"label": f"{target} terrain instead", "message": f"{target} terrain instead"},
            {"label": f"Keep {current}", "message": f"Keep {current}"},
        ]
        guidance = next_actions("ambiguous", {"reason": reason, "options": options},
                                scene_facts(self.spec, self.last_summary or {}), self.spec)
        await self.emit({"type": "question", "message": guidance["reason"], "actions": guidance["actions"],
                         "stress_test_available": False})

    async def _handle_real_terrain_choice(self, text: str) -> None:
        assert self.pending_real_terrain_change is not None and self.spec is not None
        source = " ".join(text.lower().split())
        pending = self.pending_real_terrain_change
        target = pending["target_body"]
        if source == f"keep {pending['current_body']}" or source == f"keep {pending['current_body']} terrain":
            self.pending_real_terrain_change = None
            await self.emit({"type": "explanation",
                             "message": f"Kept {pending['label']} as real {pending['current_body'].title()} terrain."})
            guidance = next_actions("scene", {"kind": "ready"}, scene_facts(self.spec, {}), self.spec)
            await self.emit({"type": "question", "message": guidance["reason"], "actions": guidance["actions"],
                             "stress_test_available": self.spec.domain == "robot_terrain"})
            return

        spec = self.spec.model_copy(deep=True)
        spec.assumptions = []
        spec.environment.body = target
        if source == f"{target} gravity on this same terrain":
            from . import presets

            spec.environment.gravity = presets.gravity_for(target)
            note = (f"Used {target.title()} gravity on {pending['label']}'s real "
                    f"{pending['current_body'].title()} terrain as a what-if.")
        elif source == f"{target} terrain instead":
            from . import presets
            from .spec import Terrain
            from .terrain_data import list_presets

            earth_presets = [(name, meta) for name, meta in list_presets().items()
                             if meta.get("body") == target and not meta.get("uploaded")]
            if earth_presets:
                name, meta = earth_presets[0]
                spec.terrain = Terrain(kind="real", preset=name, size_m=spec.terrain.size_m,
                                       seed=spec.terrain.seed)
                spec.environment.surface = meta.get("surface", "rock")
                spec.environment.friction = presets.friction_for(spec.environment.surface) or 0.9
                note = f"Replaced {pending['label']}'s real elevation with the installed Earth terrain {meta['label']}."
            else:
                spec.terrain = Terrain(kind="hills", size_m=spec.terrain.size_m, roughness=0.3,
                                       seed=spec.terrain.seed)
                spec.environment.surface = "rock"
                spec.environment.friction = presets.friction_for("rock")
                note = f"Replaced {pending['label']}'s real elevation with generated Earth hills."
            spec.environment.gravity = presets.gravity_for(target)
        else:
            await self._ask_real_terrain_choice()
            return

        spec.fill_defaults()
        if note not in spec.assumptions:
            spec.assumptions.append(note)
        self.pending_real_terrain_change = None
        await self.agent("Director", "working", "Applying your terrain choice and rebuilding the scene…")
        await self.build_and_check(spec)
        await self.agent("Director", "done", note)

    async def make(self, text: str, editing: bool, plan=None) -> None:
        """Plan and carry out one request (``plan``: a follow-up already planned, e.g. after a build)."""
        await self.agent("Director", "working", "Planning the world operations…")
        director = Director(self.emit)
        state = SessionState(simulator=self.sim, build=self.build, checks=list(self.checks), summary=self.last_summary,
                             diagnosis=self.last_diagnosis, redesign=self.last_redesign)
        world, message = await director.execute_plan(text, self.spec if editing else None, state, plan)
        follow = director.last_plan.then if director.last_plan is not None and plan is None else None
        if not editing and world.build is None:   # an answer with no scene yet: don't keep an empty one
            await self.agent("Director", "done", "Answered without building a scene.")
            await self.emit({"type": "question", "message": message, "stress_test_available": False})
            return
        spec = world.spec
        unchanged = editing and world.build is not None and world.build is self.build   # answer or analysis
        if self.library_selection and spec.domain == "robot_terrain" and not unchanged:
            self._apply_library_selection(spec)
            self.library_selection = None   # it sets up one scene; later requests are planned from their own words
            world.spec = spec
            if world.build is not None:   # it was built before the selection changed the robot: build and check that
                world.simulator = agents.route(spec)
                world.build = await asyncio.to_thread(agents.build, world.simulator, spec, lambda line: None)
                world.checks = await asyncio.to_thread(world.simulator.validate, spec, world.build)
                world.run_requested = world.run_requested and not blocking_failures(world.checks)
        self.spec, self.sim, self.build, self.checks = spec, world.simulator, world.build, list(world.checks)
        if not unchanged:
            self._forget_runs()
        if world.report is not None:
            if world.report["type"] == "diagnosis":
                self.last_diagnosis = world.report
            elif world.report["type"] == "redesign":
                self.last_redesign = world.report
            elif world.report["type"] == "envelope":
                self.last_finder_report = world.report
                self._finder_source_spec = spec.model_copy(deep=True)
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
        if follow is not None and self.build is not None:
            if not blocking_failures(self.checks):   # build → (run if needed) → the analysis the request asked for
                self.awaiting_approval = False
                await self.emit({"type": "explanation", "message": message})
                await self._follow_up(follow, text)
                return
            message += f" It didn't pass its checks, so I didn't go on to {FOLLOW_UP_WORDS[follow]}."
        if world.run_requested and self.build is not None:
            self.awaiting_approval = False
            await self.emit({"type": "explanation", "message": message})   # a chat line without a Run chip
            await self.run()
            return
        if unchanged:
            if world.report is not None:
                report_type = world.report.get("type", "")
                workflow = "finder" if report_type == "envelope" else report_type
                guidance = next_actions(workflow, {"report": world.report}, scene_facts(spec, self.last_summary or {}), spec)
                event = {
                    "type": "question",
                    "message": guidance["reason"],
                    "actions": guidance["actions"],
                    "stress_test_available": spec.domain == "robot_terrain",
                }
                if report_type == "envelope" and world.report.get("honesty"):
                    event["note"] = world.report["honesty"]
                await self.emit(event)
                return
            await self.emit({
                "type": "question",
                "message": message,
                "chips": self._next_chips(world.report),
                "stress_test_available": spec.domain == "robot_terrain",
            })
            return
        if world.report is not None:
            report_type = world.report.get("type", "")
            workflow = "finder" if report_type == "envelope" else report_type
            guidance = next_actions(workflow, {"report": world.report}, scene_facts(spec, world.results or {}), spec)
        elif blocking_failures(self.checks):
            guidance = next_actions("checks", {
                "kind": "check_failed", "checks": [check.__dict__ for check in self.checks],
                "structures": self.build.scene.get("structures", []) if self.build else [],
            }, scene_facts(spec, world.results or {}), spec)
        else:
            guidance = next_actions("scene", {"kind": "ready", "structures": self.build.scene.get("structures", [])
                                    if self.build else []}, scene_facts(spec, world.results or {}), spec)
        await self.emit({
            "type": "question",
            "message": guidance["reason"],
            "actions": guidance["actions"],
            "stress_test_available": spec.domain == "robot_terrain",
            **({"note": world.report["honesty"]}
               if world.report is not None and world.report.get("type") == "envelope" and world.report.get("honesty")
               else {}),
        })

    async def _follow_up(self, workflow: str, text: str) -> None:
        """The analysis a build request also asked for: a stress test, where it breaks, or why it fails (which needs
        a run of the scene first)."""
        if workflow == "stress_test":
            await self.run_stress(text)
            return
        if workflow == "diagnose":
            await self.run()
            if SessionState(summary=self.last_summary).failed_robot() is None:
                await self.emit({"type": "question", "stress_test_available": True,
                                 "message": "It reached the goal, so there is no failure to explain."})
                return
        await self.make(text, editing=True, plan=follow_up_plan(workflow, text, self.spec))

    def _next_chips(self, report: dict | None) -> list[str] | None:
        """After a diagnosis or redesign, offer the next step instead of "Run"."""
        if report is None or report["type"] == "envelope":   # finding where it breaks leaves the scene ready to run
            return None
        if report["type"] == "diagnosis":
            if report.get("reproduced") is None:   # the robot asked about didn't fail: offer the one that did
                return self._failure_chips(self.last_summary)
            return [self._fix_chip(report.get("robot_id"))] if report.get("reproduced") else ["Run again"]
        if report.get("winner"):
            return ["Use this design"]
        return ["Use fragile design"] if report.get("fragile") else ["Run again"]

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

    def _checked_action_robot(self, robot_id: str | None) -> str:
        if self.spec is None or self.spec.domain != "robot_terrain" or not self.spec.robots:
            raise ValueError("This action needs an existing robot scene.")
        selected = robot_id or (self.spec.robots[0].id if len(self.spec.robots) == 1 else None)
        if selected is None or not any(entry.id == selected for entry in self.spec.robots):
            raise ValueError("Choose a robot from this scene before running that action.")
        return selected

    async def _structure_marsquake(self) -> None:
        from concurrent.futures import ProcessPoolExecutor
        from .structural.shaking import synthetic_marsquake

        await self.agent("Checker", "working", "Running a synthetic marsquake what-if, not measured seismic data.")
        records = self.build.scene.get("structures", [])
        if not records:
            raise ValueError("No placed structures are available for a shaking check.")
        with ProcessPoolExecutor(max_workers=1) as executor:
            report = await asyncio.get_running_loop().run_in_executor(
                executor, synthetic_marsquake, records, self.spec.environment.gravity)
        await self.emit({"type": "structure_shaking", "report": report})
        await self.emit({"type": "explanation", "message": report["honesty"]})
        await self.agent("Checker", "done", "Synthetic structural shaking complete; the baseline scene is unchanged.")

    async def handle_structure_action(self, operation: str, structure_index: int | None = None) -> None:
        """Apply one code-selected structural change and rebuild/re-check, never replan it."""
        try:
            if self.spec is None or self.build is None or not self.spec.structures:
                raise ValueError("Build a structure scene before using a structure action.")
            if operation == "marsquake":
                await self._structure_marsquake()
                return
            if isinstance(structure_index, bool) or not isinstance(structure_index, int) \
                    or not 0 <= structure_index < len(self.spec.structures):
                raise ValueError("That structure action has an invalid or stale target.")
            record = next((item for item in self.build.scene.get("structures", [])
                           if item.get("request_index") == structure_index), None)
            if record is None:
                raise ValueError("The selected structure was not placed in this scene.")
            updated = self.spec.model_copy(deep=True)
            structure = updated.structures[structure_index]
            if operation == "stronger_members":
                from .structural.checks import stronger_section

                section = stronger_section(record)
                if section is None:
                    raise ValueError("This structure already uses the largest installed catalogue section.")
                structure.section = section
                note = f"Changed only {structure.kind} {structure_index + 1} members to {section}; re-checking."
            elif operation == "mid_support" and structure.kind == "bridge":
                if structure.mid_support:
                    raise ValueError("This bridge already has a mid support.")
                structure.mid_support = True
                note = "Added one central bridge support; re-checking. Support foundation capacity is not analysed."
            elif operation == "thinner_shielding" and structure.kind == "habitat":
                if structure.shielding_thickness_m <= 0:
                    raise ValueError("This habitat has no shielding to reduce.")
                previous = structure.shielding_thickness_m
                structure.shielding_thickness_m = previous / 2
                note = (f"Reduced only habitat shielding from {previous:g} m to {structure.shielding_thickness_m:g} m; "
                        "lower structural load, but reduced radiation shielding. Radiation protection is not analysed.")
            else:
                raise ValueError(f"Unsupported structure action {operation!r} for {structure.kind}.")
            updated.assumptions.append(note)
            await self.agent("Director", "working", note)
            await self.build_and_check(updated)
            await self.emit({"type": "explanation", "message": note})
            await self.agent("Director", "done", "The selected structure change has been re-checked.")
        except Exception as error:
            await self.emit({"type": "error", "message": f"{type(error).__name__}: {error}"})

    async def handle_workflow_action(self, workflow: str, robot_id: str | None) -> None:
        """Run an explicit result-card action against its selected scene robot, without replanning its target."""
        try:
            target = self._checked_action_robot(robot_id)
            if workflow == "replay_weakest_spot":
                await self.replay_weakest_spot(target)
                return
            if workflow == "show_design":
                await self.show_design(target)
                return
            if workflow == "run":
                await self.run()
                return
            if workflow == "apply_design":
                report = self.last_redesign or {}
                report_target = report.get("robot_id") or (
                    self.spec.robots[0].id if len(self.spec.robots) == 1 else None
                )
                if target != report_target:
                    raise ValueError("That design belongs to a different robot.")
                winner = report.get("winner") or report.get("fragile")
                if winner is None:
                    raise ValueError("There is no design to apply for that robot.")
                plan = Plan(workflow="edit", summary="Apply the selected robot design",
                            changes={"robot": winner["changes"], "robot_id": target})
                await self.make("apply the selected robot design", editing=True, plan=plan)
                return
            if workflow not in {"diagnose", "redesign", "find_failures"}:
                raise ValueError(f"Unsupported result action: {workflow}.")
            plan = Plan(
                workflow=workflow,
                summary={"diagnose": "Why the selected robot failed", "redesign": "Make the selected robot pass",
                         "find_failures": "Find the selected robot's limits"}[workflow],
                robot_id=target,
                constraints="Make the selected robot pass" if workflow == "redesign" else None,
            )
            await self.make(plan.summary, editing=True, plan=plan)
        except Exception as error:  # noqa: BLE001 - malformed or stale action is shown without closing the session
            await self.emit({"type": "error", "message": f"{type(error).__name__}: {error}"})

    @staticmethod
    def _weakest_confirmed_evidence(report: dict, robot_id: str) -> dict | None:
        failures = [failure for failure in report.get("failures", [])
                    if failure.get("confirm", {}).get("label") in ("reliable fail", "borderline")]
        failures.sort(key=lambda failure: (
            failure.get("kind") != "single",
            len(failure.get("values") or {}),
            failure.get("distance", float("inf")),
            failure.get("run", ""),
        ))
        evidence = report.get("evidence", [])
        for failure in failures:
            confirmed_runs = [row for row in evidence
                              if row.get("confirms") == failure.get("run") and row.get("broke") is True
                              and (row.get("change") or {}).get("robot_id") in (None, robot_id)]
            if confirmed_runs:
                return min(confirmed_runs, key=lambda row: (row.get("seed", float("inf")), row.get("id", "")))
        return None

    async def replay_weakest_spot(self, robot_id: str | None) -> None:
        """Re-simulate the simplest confirmed Finder failure with its exact seed and stream frames."""
        try:
            target = self._checked_action_robot(robot_id)
            report = self.last_finder_report or {}
            if report.get("robot_id") != target or self._finder_source_spec is None:
                raise ValueError("There is no Finder result for that robot in this scene.")
            evidence = self._weakest_confirmed_evidence(report, target)
            if evidence is None:
                await self.emit({"type": "question", "message": "There is no confirmed failure to replay for that robot."})
                return

            from .batch import _validate_variation
            from .finder import prepare

            source = prepare(self._finder_source_spec, report.get("knobs") or [])
            change = dict(evidence.get("change") or {})
            if "seed" not in change and evidence.get("seed") is not None:
                base_seed = source.terrain.seed if source.terrain is not None else 0
                change["seed"] = base_seed + int(evidence["seed"])
                change["perturb_seed"] = int(evidence["seed"])
            _, spec, _ = _validate_variation(source, change)
            sim = agents.route(spec)
            build = await asyncio.to_thread(agents.build, sim, spec, lambda _: None)
            await self.emit({"type": "scene", "scene": build.scene, "code": build.code, "language": build.language})
            self._view_shows_other_scene = True
            await self.agent("Simulator", "working",
                             f"Replaying confirmed run {evidence.get('confirms')} for {report.get('robot_label', target)} "
                             f"with seed {evidence.get('seed', change.get('seed'))}…")
            await self._stream(spec, sim, build)
            await self.agent("Simulator", "done", "Evidence replay finished; the Finder result is unchanged.")
            await self.emit({"type": "explanation",
                             "message": f"Replayed the confirmed {evidence.get('failure') or 'failure'} for "
                                        f"{report.get('robot_label', target)} using the recorded seed."})
        except Exception as error:  # noqa: BLE001
            await self.emit({"type": "error", "message": f"Could not replay the weakest spot: {error}"})

    async def show_design(self, robot_id: str | None = None) -> None:
        """Replay the last redesign's winning design with streamed frames, without applying it."""
        try:
            if self.last_redesign:
                report_target = self.last_redesign.get("robot_id") or (
                    self.spec.robots[0].id if self.spec and len(self.spec.robots) == 1 else None
                )
                if robot_id is not None and robot_id != report_target:
                    raise ValueError("That design belongs to a different robot.")
            winner = (self.last_redesign or {}).get("winner") or (self.last_redesign or {}).get("fragile")
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
                             "message": (f"That was the fragile design ({winner['description']}), not applied yet. "
                                         "Say \"use fragile design\" to apply it anyway.") if winner.get("fragile") else
                                        (f"That was the winning design ({winner['description']}), not applied yet. "
                                         "Say \"use this design\" to apply it.")})
        except Exception as e:  # noqa: BLE001
            await self.emit({"type": "error", "message": f"{type(e).__name__}: {e}"})

    async def _stream(self, spec: SceneSpec, sim: Simulator, build: BuildResult) -> dict:
        """Run a scene, streaming run_start, frames and metrics to the browser; returns the metrics."""
        cost.add_runs(1)
        await self.emit({"type": "run_start", "duration_s": spec.duration_s})
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            try:
                for chunk in sim.run(spec, build):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:   # also when the run raises, so the wait below ends and `await task` reports the error
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
        self.last_finder_report = None
        self._finder_source_spec = None

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

    async def select_library(self, robot_id: str, policy_id: str, world_id: str, *, start_scene: bool = False,
                             terrain_kind: str = "hills", title: str = "", obstacles: list[dict] | None = None,
                             scene_spec: dict | None = None, robot_settings: dict | None = None) -> None:
        from .library import LibraryRegistry

        registry = LibraryRegistry()
        robot = registry.robot(robot_id)
        policy = registry.policy(policy_id)
        world = next((item for item in registry.worlds() if item.id == world_id), None)
        if world is None:
            raise ValueError(f"Unknown world library id: {world_id}")
        if policy not in registry.compatible_policies(robot_id):
            raise ValueError(f"Policy {policy_id!r} is not compatible with robot {robot_id!r}.")
        if start_scene and self.spec is None:
            from .spec import Environment, Mission, Obstacle, Robot, SceneSpec, Terrain

            if scene_spec is not None:
                spec = SceneSpec.model_validate(scene_spec).fill_defaults()
                if spec.domain != "robot_terrain":
                    raise ValueError("A saved Library world must be a robot terrain scene.")
                if title:
                    spec.title = title
                await self.build_and_check(spec)
                return
            kind = world.terrain_kind
            if kind != "real":
                kind = terrain_kind
            terrain = Terrain(kind=kind, preset=world.terrain_preset if kind == "real" else None)
            goal = "highest" if terrain.kind in {"hills", "rocky", "craters"} else "east"
            spec = SceneSpec(
                domain="robot_terrain",
                title=title or f"{robot.name} · {world.name}",
                environment=Environment(body=world.body, gravity=world.gravity_m_s2, surface=world.surface,
                                        friction=world.friction),
                terrain=terrain,
                robot=Robot(kind="rover"),
                mission=Mission(goal_place=goal),
                obstacles=[Obstacle.model_validate(item) for item in (obstacles or [])],
            )
            self.library_selection = (robot.id, policy.id, world.id)
            self._apply_library_selection(spec)
            if robot.source == "builtin" and robot_settings:
                allowed = {"wheels"} if robot.id == "rover" else {"legs"} if robot.id == "walker" else set()
                if set(robot_settings) - allowed:
                    raise ValueError(f"Unsupported setup options for {robot.name}.")
                updated = Robot.model_validate({**spec.robot.model_dump(), **robot_settings})
                spec.robot = updated
                spec.robots[0].robot = updated.model_copy(deep=True)
                spec.fill_defaults()
            if title:
                spec.title = title
            self.library_selection = None
            await self.agent("Director", "done", f"Using the selected {robot.name} and {world.name}; no planning call needed.")
            await self.build_and_check(spec)
            return
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
            self.library_selection = None
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
        blocked = blocking_failures(checks)
        guidance = next_actions("checks" if blocked else "scene", {
            "kind": "check_failed" if blocked else "ready", "checks": [check.__dict__ for check in checks],
            "structures": self.build.scene.get("structures", []),
        }, scene_facts(spec, {}), spec)
        await self.emit({"type": "question", "message": guidance["reason"], "actions": guidance["actions"],
                         "stress_test_available": spec.domain == "robot_terrain"})

    async def _start_on_flatter_ground(self) -> None:
        if self.spec is None or self.spec.domain != "robot_terrain":
            await self.emit({"type": "error", "message": "Start on flatter ground needs a robot scene."})
            return
        from .backends.mujoco_backend import move_start_to_flatter

        spec = self.spec.model_copy(deep=True)
        note = move_start_to_flatter(spec, search_radius_m=15.0)
        if note is None:
            await self.emit({"type": "explanation", "message": "No flatter starting spot was found within 15 m; the start was left unchanged."})
            return
        await self.agent("Director", "working", "Looking for flatter ground near the start…")
        await self.build_and_check(spec)
        await self.agent("Director", "done", note)

    async def _find_a_gentler_route(self) -> None:
        if self.spec is None or self.spec.domain != "robot_terrain":
            await self.emit({"type": "error", "message": "Finding a gentler route needs a robot scene."})
            return
        from .backends.mujoco_backend import apply_gentler_route

        spec = self.spec.model_copy(deep=True)
        note = apply_gentler_route(spec)
        if note is None:
            await self.emit({"type": "explanation", "message": "I couldn't find a gentler parallel route on this terrain; the world was left unchanged."})
            return
        await self.agent("Director", "working", "Finding flatter start and goal locations along the same route…")
        await self.build_and_check(spec)
        await self.agent("Director", "done", note)

    async def restore_scene(self, scene_data: dict) -> None:
        """Rebuild a locally saved, validated scene without planning a new one."""
        try:
            spec = SceneSpec.model_validate(scene_data).fill_defaults()
            self.library_selection = None
            await self.build_and_check(spec)
        except Exception as error:  # noqa: BLE001 - malformed local storage is shown, never run
            await self.emit({"type": "error", "message": f"Could not reopen this saved scene: {error}"})

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
        repeats = agents.STRESS_REPEATS if agents.is_stress_repeat(text) else 1
        repeated = await self._repeat_rows(spec, variations, results, repeats) if repeats > 1 else {}
        rows = []
        for result in results:
            summary = result.get("summary") or {}
            row = {"id": result["id"], "label": result["label"], "passed": result["passed"],
                   "reason": result["reason"], "metrics": self._row_metrics(summary),
                   "ghost_status": result.get("ghost_status")}
            if len(spec.robots) > 1:   # a race: one cell per robot, and no single pass/fail for the row
                row["robots"] = {robot_id: {**robot, "metrics": self._row_metrics((summary.get("robots") or {}).get(robot_id) or {})}
                                 for robot_id, robot in (result.get("robots") or {}).items()}
                row["passed"] = None
            if result["id"] in repeated:
                row["reliability"] = repeated[result["id"]]
            rows.append(row)
        columns = [{"id": entry.id, "label": entry.label, "kind": entry.robot.kind} for entry in spec.robots]
        race = {}
        if len(spec.robots) > 1:   # "Rover: 8/8 · Walker: 0/8" over the runs that ran
            ran = [row for row in rows if row.get("robots")]
            counts = {column["id"]: sum(bool(row["robots"].get(column["id"], {}).get("passed")) for row in ran) for column in columns}
            race = {"robots": columns, "robot_passed": counts,
                    "overall": " · ".join(f"{column['label']}: {counts[column['id']]}/{len(ran)}" for column in columns)}
        spent = cost.current()
        await self.emit({"type": "report", "rows": rows, "passed": passed, "total": len(results), "repeats": repeats,
                         **race, **({"cost": spent.snapshot()} if spent else {})})
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
        if repeated:   # how sure each row is, counted in code
            labels = [summary["label"] for summary in repeated.values()]
            explanation += (f" Repeated on {repeats} seeds: " + ", ".join(
                f"{labels.count(name)} {name}" for name in ("reliable pass", "borderline", "reliable fail")
                if labels.count(name)) + ".")
        await self.emit({"type": "explanation", "message": explanation,
                         **({} if repeated else {"chips": ["Repeat ×3"]})})

    async def _repeat_rows(self, spec: SceneSpec, variations: list[dict], results: list[dict],
                           repeats: int) -> dict[str, dict]:
        """Every row that ran, on repeats - 1 more seeds: its pass count and rate, Wilson 95% interval, label
        (reliable pass, borderline, reliable fail) and the key metric's mean ± spread, by row id."""
        from . import reliability

        ran = [(result, None if result["id"] == "run-0" else variations[int(result["id"].removeprefix("run-")) - 1])
               for result in results if not result.get("error")]
        await self.agent("Batch runner", "working", f"Repeating {len(ran)} rows on {repeats - 1} more seeds…")
        more = await asyncio.to_thread(reliability.run_seeded, spec, [variation for _, variation in ran], repeats, None, 1)
        repeated = {}
        for (result, _), extra in zip(ran, more):
            runs = [result, *extra]
            repeated[result["id"]] = reliability.summarize(runs)
            if len(spec.robots) > 1:   # a race: each robot's own rate too
                repeated[result["id"]]["robots"] = {entry.id: reliability.summarize(runs, entry.id) for entry in spec.robots}
        await self.agent("Batch runner", "done", f"Repeated {len(ran)} rows on {repeats} seeds in total.")
        return repeated

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
        if self.spec.domain == "robot_terrain":
            from .facts import facts as build_facts

            outcome = "run_passed" if metrics.get("goal_reached") else "run_failed"
            guidance = next_actions(outcome, {"kind": outcome, "passed": bool(metrics.get("goal_reached")),
                                              "metrics": metrics}, build_facts(self.spec, metrics), self.spec)
        else:
            guidance = None
        await self.emit({"type": "explanation", "message": text,
                         **({"reason": guidance["reason"], "actions": guidance["actions"]} if guidance else {})})
