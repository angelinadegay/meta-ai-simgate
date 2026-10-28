"""Tool-using Director loop for constructing and operating simulation worlds."""

import asyncio
import re
import time
import uuid
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import agents, config, llm
from .spec import Robot, SceneSpec
from .tools import ToolResult, World, invoke_tool, tool_catalog

StepCallback = Callable[[dict[str, Any]], Awaitable[None]]


class DirectorTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thought: str = Field(min_length=1, max_length=400)
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    final: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "DirectorTurn":
        if (self.tool is None) == (self.final is None):
            raise ValueError("A Director reply must contain exactly one of tool or final.")
        if self.tool is None and self.args:
            raise ValueError("A final Director reply cannot include tool arguments.")
        return self


DIRECTOR_SYSTEM = """You are the Director of a physics simulation sandbox.
Use the available tools to change the current world, build and check it, run requested
simulations, compare conditions, and explain measured results. Respond with exactly one
JSON object each turn:
{"thought":"one short sentence shown to the user","tool":"registered_name","args":{}}
or {"thought":"one short sentence shown to the user","final":"message"}.

Rules:
- Use only registered tools and library/preset ids listed in the world context. Never invent ids.
- Build and check before running. If a check fails, repair it with tools or build_custom, then check again.
- Run only when the user explicitly asks to run, compare, race, test, or has approved a pending run.
  Otherwise end by calling ask_user with "Run it?".
- Physical values and placements must be selected by tools, not guessed in prose.
- Prefer short, clear actions; maximum 12 tool calls per user message.
- A tool error is returned once for correction; if the next attempt is invalid, stop with a clear final message.

Registered tools and purpose:
{tools}
"""


class Director:
    def __init__(self, step_callback: StepCallback | None = None) -> None:
        self.step_callback = step_callback
        self._counter = 0
        self._root_id: str | None = None

    async def execute(self, text: str, current: SceneSpec | None = None) -> tuple[World, str]:
        if config.MOCK_MODE:
            return await self._mock_execute(text, current)
        world = World(current.model_copy(deep=True) if current is not None else SceneSpec(domain="robot_terrain"))
        prompt = DIRECTOR_SYSTEM.replace(
            "{tools}",
            "\n".join(f"- {name}: {detail}" for name, detail in tool_catalog().items()),
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Current world state:\n{world.state()}\n\nUser request:\n{text}"},
        ]
        recovery_used = False
        root_id = self._id()
        self._root_id = root_id
        await self._emit_step(
            id=root_id, parent_id=None, agent="Director", model=config.PLANNER_MODEL,
            tool=None, args={}, thought="Planning the next simulation actions.", result="",
            status="working", ms=0,
        )
        for _ in range(12):
            reply = await asyncio.to_thread(llm.chat, messages, config.PLANNER_MODEL)
            messages.append({"role": "assistant", "content": reply})
            try:
                turn = DirectorTurn.model_validate(llm.extract_json(reply))
                if turn.tool is not None and turn.tool not in tool_catalog():
                    raise ValueError(f"Unknown tool: {turn.tool}")
            except (ValueError, TypeError) as exc:
                await self._emit_step(
                    id=self._id(), parent_id=root_id, agent="Director", model=config.PLANNER_MODEL,
                    tool=None, args={}, thought="Checking the response format.", result=str(exc),
                    status="warning", ms=0,
                )
                if recovery_used:
                    message = f"I couldn't continue because the Director returned invalid instructions: {exc}"
                    await self._finish_root(message)
                    return world, message
                recovery_used = True
                messages.append({"role": "user", "content": f"Invalid Director reply: {exc}. Return one corrected JSON object."})
                continue
            if turn.final is not None:
                await self._emit_step(
                    id=self._id(), parent_id=root_id, agent="Director", model=config.PLANNER_MODEL,
                    tool=None, args={}, thought=turn.thought, result=turn.final, status="done", ms=0,
                )
                await self._finish_root(turn.final)
                return world, turn.final
            assert turn.tool is not None
            tool_step = self._id()
            started = time.perf_counter()
            await self._emit_step(
                id=tool_step, parent_id=root_id, agent="Director", model=config.PLANNER_MODEL,
                tool=turn.tool, args=turn.args, thought=turn.thought, result="",
                status="working", ms=0,
            )
            try:
                result = await asyncio.to_thread(invoke_tool, world, turn.tool, turn.args)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                await self._emit_step(
                    id=tool_step, parent_id=root_id, agent="Director", model=config.PLANNER_MODEL,
                    tool=turn.tool, args=turn.args, thought=turn.thought, result=message,
                    status="warning", ms=0,
                )
                if recovery_used:
                    message = f"I stopped after the tool failed: {message}"
                    await self._finish_root(message)
                    return world, message
                recovery_used = True
                schema = tool_catalog()[turn.tool]
                messages.append({
                    "role": "user",
                    "content": (
                        f"Tool {turn.tool} failed: {message}. The arguments you sent were "
                        f"{turn.args!r}. Correct the arguments using this tool definition and schema:\n"
                        f"{schema}\nRetry the action once; do not omit required choices."
                    ),
                })
                continue
            await self._emit_handoffs(turn.tool, result, tool_step)
            await self._emit_step(
                id=tool_step, parent_id=root_id, agent="Director", model=config.PLANNER_MODEL,
                tool=turn.tool, args=turn.args, thought=turn.thought, result=result.text,
                status="done", ms=round((time.perf_counter() - started) * 1000),
            )
            messages.append({
                "role": "user",
                "content": f"Tool result: {result.text}\nStructured data: {result.data}\nUpdated world: {world.state()}",
            })
            if turn.tool == "ask_user":
                await self._finish_root(result.text)
                return world, result.text
        message = "The Director reached the 12-tool limit before finishing. Please narrow the request."
        await self._finish_root(message)
        return world, message

    async def _mock_execute(self, text: str, current: SceneSpec | None) -> tuple[World, str]:
        planned = await asyncio.to_thread(agents.edit, current, text) if current is not None else await asyncio.to_thread(agents.plan, text)
        self._root_id = self._id()
        await self._emit_step(
            id=self._root_id, parent_id=None, agent="Director", model="rules",
            tool=None, args={}, thought="Turning the request into safe world operations.",
            result="", status="working", ms=0,
        )
        if planned.domain != "robot_terrain":
            world = World(planned)
            await self._mock_call(world, "build_and_check", {}, "Building and checking the planned structure.")
            await self._mock_call(world, "ask_user", {"question": "Run it?"}, "Asking before starting the simulation.")
            message = world.question or "Run it?"
            await self._finish_root(message)
            return world, message
        if current is not None:
            world = World(planned)
            terrain = planned.terrain
            await self._mock_call(world, "set_world", {
                "body": planned.environment.body,
                "surface": planned.environment.surface,
                "terrain_kind": terrain.kind if terrain else None,
                "preset": terrain.preset if terrain and terrain.kind == "real" else None,
                "slope_deg": terrain.slope_deg if terrain and terrain.kind == "slope" else None,
                "roughness": terrain.roughness if terrain else None,
            }, "Applying the requested changes to the existing world.")
            await self._mock_call(world, "build_and_check", {}, "Building and checking the updated scene.")
            await self._mock_call(world, "ask_user", {"question": "Run it?"}, "Waiting for approval to run.")
            message = world.question or "Run it?"
            await self._finish_root(message)
            return world, message

        requests = _requested_robots(text, planned.robot)
        world_spec = planned.model_copy(deep=True)
        world_spec.robots = []
        world_spec.robot = None
        world_spec.mission = None
        world_spec.obstacles = []
        world = World(world_spec)
        terrain = planned.terrain
        world_args = {
            "body": planned.environment.body,
            "surface": planned.environment.surface,
            "terrain_kind": terrain.kind if terrain else None,
            "preset": terrain.preset if terrain and terrain.kind == "real" else None,
            "slope_deg": terrain.slope_deg if terrain and terrain.kind == "slope" else None,
            "roughness": terrain.roughness if terrain else None,
        }
        await self._mock_call(world, "set_world", world_args, "Selecting the shared world conditions.")
        for index, robot_args in enumerate(requests):
            result = await self._mock_call(world, "spawn_robot", robot_args, f"Adding robot {index + 1}.")
            if index == 0 and planned.robot is not None:
                world.spec.robots[-1].robot = planned.robot.model_copy(deep=True)
                world.spec.robot = planned.robot.model_copy(deep=True)
                world.spec.fill_defaults()
        if planned.obstacles:
            await self._mock_call(
                world,
                "add_obstacles",
                {"obstacles": [obstacle.model_dump(mode="json") for obstacle in planned.obstacles]},
                "Placing the requested obstacles.",
            )
        if planned.mission is not None and planned.mission.has_goal():
            goal = planned.mission.goal_xy if planned.mission.goal_xy is not None else planned.mission.goal_place
            goal_args = {"robot_id": "all", "xy": goal} if isinstance(goal, list) else {"robot_id": "all", "place": goal}
            await self._mock_call(world, "set_goal", goal_args, "Assigning the goal.")
        await self._mock_call(world, "build_and_check", {}, "Building and checking the robot scene.")
        await self._mock_call(world, "ask_user", {"question": "Run it?"}, "Waiting for approval to run.")
        message = world.question or "Run it?"
        await self._finish_root(message)
        return world, message

    async def _mock_call(self, world: World, name: str, args: dict[str, Any], thought: str) -> ToolResult:
        step_id = self._id()
        started = time.perf_counter()
        await self._emit_step(
            id=step_id, parent_id=self._root_id, agent="Director", model="rules",
            tool=name, args=args, thought=thought, result="", status="working", ms=0,
        )
        result = await asyncio.to_thread(invoke_tool, world, name, args)
        await self._emit_handoffs(name, result, step_id)
        await self._emit_step(
            id=step_id, parent_id=self._root_id, agent="Director", model="rules",
            tool=name, args=args, thought=thought, result=result.text, status="done",
            ms=round((time.perf_counter() - started) * 1000),
        )
        return result

    async def _emit_handoffs(self, tool: str, result: ToolResult, parent_id: str) -> None:
        handoffs = []
        if tool == "build_and_check":
            failed = [
                check for check in result.data.get("checks", [])
                if not check.get("passed", False)
            ]
            handoffs = [
                ("Builder", config.BUILDER_MODEL if config.BUILDER_MODE == "llm" else "template", "Build finished."),
                (
                    "Checker",
                    "rules",
                    "All checks passed." if result.data.get("passed") else "; ".join(
                        f"{check.get('name')}: {check.get('detail')}" for check in failed
                    ) or "Some physics checks need attention.",
                ),
            ]
        elif tool == "build_custom":
            errors = result.data.get("errors", [])
            attempts = result.data.get("attempts", 1)
            for attempt in range(1, attempts + 1):
                detail = errors[attempt - 1] if attempt <= len(errors) else result.text
                status = "warning" if attempt <= len(errors) else "done"
                await self._emit_agent_step(
                    "Builder",
                    config.BUILDER_MODEL,
                    tool,
                    parent_id,
                    f"Custom fragment attempt {attempt}.",
                    detail,
                    status,
                )
            return
        elif tool == "run":
            handoffs = [("Simulator", "MuJoCo", "Simulation completed and metrics were measured.")]
        elif tool == "explain":
            handoffs = [("Explainer", config.PLANNER_MODEL if not config.MOCK_MODE else "rules", result.text)]
        elif tool == "stress_test":
            handoffs = [
                ("Stress planner", config.PLANNER_MODEL if not config.MOCK_MODE else "rules", "Selected condition variations."),
                ("Batch runner", "ProcessPoolExecutor", result.text),
                ("Explainer", config.PLANNER_MODEL if not config.MOCK_MODE else "rules", "Summarized the measured pattern."),
            ]
        for agent, model, message in handoffs:
            status = "warning" if agent == "Checker" and not result.data.get("passed") else "done"
            await self._emit_agent_step(
                agent, model, tool, parent_id, f"{agent} hand-off.", message, status,
            )

    async def _emit_agent_step(
        self,
        agent: str,
        model: str,
        tool: str,
        parent_id: str,
        thought: str,
        result: str,
        status: str,
    ) -> None:
        step_id = self._id()
        await self._emit_step(
            id=step_id, parent_id=parent_id, agent=agent, model=model, tool=tool,
            args={}, thought=thought, result="", status="working", ms=0,
        )
        await self._emit_step(
            id=step_id, parent_id=parent_id, agent=agent, model=model, tool=tool,
            args={}, thought=thought, result=result, status=status, ms=0,
        )

    async def _emit_step(self, **payload: Any) -> None:
        if self.step_callback is not None:
            await self.step_callback({"type": "step", **payload})

    async def _finish_root(self, message: str) -> None:
        if self._root_id is not None:
            await self._emit_step(
                id=self._root_id, parent_id=None, agent="Director",
                model=config.PLANNER_MODEL if not config.MOCK_MODE else "rules",
                tool=None, args={}, thought="Director finished this request.",
                result=message, status="done", ms=0,
            )

    def _id(self) -> str:
        self._counter += 1
        return f"step-{self._counter}-{uuid.uuid4().hex[:8]}"


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}


def _requested_robots(text: str, fallback: Robot | None) -> list[dict[str, Any]]:
    source = text.lower()
    pattern = re.compile(r"(?:(\d+|one|two|three|four|five|six)\s+)?(g1s?|humanoids?|rovers?|walkers?|hexapods?)")
    matches = list(pattern.finditer(source))
    if re.search(r"\bg1\s+humanoid\b", source):
        matches = [match for match in matches if not match.group(2).startswith("humanoid")]
    parsed: list[dict[str, Any]] = []
    for match in matches:
        amount = match.group(1)
        count = int(amount) if amount and amount.isdigit() else _NUMBER_WORDS.get(amount, 1)
        name = match.group(2)
        if name.startswith(("g1", "humanoid")):
            item = {"robot_id": "g1", "policy_id": "g1_walking", "label": "G1"}
        elif name.startswith("walker") or name.startswith("hexapod"):
            item = {"kind": "walker", "label": "Walker"}
            if re.search(r"\b(?:6|six)[\s-]*leg", source):
                item["legs"] = 6
            if re.search(r"\b(?:4|four)[\s-]*leg", source):
                item["legs"] = 4
        else:
            item = {"kind": "rover", "wheels": 6 if re.search(r"\b(?:6|six)[\s-]*wheel", source) else 4, "label": "Rover"}
        parsed.extend([item.copy() for _ in range(count)])
    if not parsed:
        robot = fallback or Robot()
        if robot.kind == "library":
            return [{"robot_id": robot.robot_id, "policy_id": robot.policy_id, "label": "G1"}]
        item = {"kind": robot.kind, "label": robot.kind.title()}
        if robot.kind == "walker":
            item["legs"] = robot.legs
        else:
            item["wheels"] = robot.wheels
        return [item]
    if len(parsed) > 6:
        raise ValueError("The Director can spawn at most six robots.")
    return parsed
