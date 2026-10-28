"""Web server: serves the sandbox UI and streams agent events over a WebSocket.

Run:  uvicorn server:app --reload   then open http://localhost:8000
"""
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from simgate import config
from simgate.library import LibraryRegistry
from simgate.orchestrator import Session
from simgate.terrain_data import list_presets

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="SimGate")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "vendor" / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "mock_mode": config.MOCK_MODE, "planner_model": config.PLANNER_MODEL, "builder_mode": config.BUILDER_MODE}


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()

    async def emit(event: dict) -> None:
        await socket.send_text(json.dumps(event))

    session = Session(emit)
    library = LibraryRegistry()
    terrains = [{"name": n, "label": m["label"], "body": m.get("body")} for n, m in list_presets().items()]
    await emit({"type": "hello", "mock_mode": config.MOCK_MODE, "model": None if config.MOCK_MODE else config.PLANNER_MODEL,
                "terrains": terrains,
                "library": {
                    "robots": [card.model_dump(mode="json") for card in library.robots()],
                    "policies": [card.model_dump(mode="json") for card in library.policies()],
                    "worlds": [card.model_dump(mode="json") for card in library.worlds()],
                }})
    try:
        while True:
            msg = json.loads(await socket.receive_text())
            if msg.get("type") == "reset":
                session = Session(emit)
                await emit({"type": "reset"})
            elif msg.get("type") in ("set_start", "set_goal"):
                await session.set_point(msg["type"][4:], float(msg["x"]), float(msg["y"]))
            elif msg.get("type") == "show_design":
                await session.show_design()
            elif msg.get("type") == "library_select":
                await session.select_library(msg["robot_id"], msg["policy_id"], msg["world_id"])
            elif text := (msg.get("text") or "").strip():
                await session.handle(text)
    except WebSocketDisconnect:
        pass
