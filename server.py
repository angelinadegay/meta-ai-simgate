"""Web server: serves the sandbox UI and streams agent events over a WebSocket.

Run:  uvicorn server:app --reload   then open http://localhost:8000
"""
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from simgate import config
from simgate.library import LibraryRegistry, UploadError, save_upload
from simgate.library.terrain_upload import NeedsAnswers, prepare_terrain
from simgate.library.uploads import MAX_UPLOAD_BYTES, discard_upload
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


@app.post("/upload")
async def upload(kind: str = Form(..., alias="type"), name: str = Form(""), credit: str = Form(""),
                 license: str = Form(""), files: list[UploadFile] = File(...), body: str = Form(""),
                 surface: str = Form(""), size_m: str = Form(""), height_range_m: str = Form("")):
    """Store an uploaded terrain, robot or policy (files or one .zip) after the checks in library/uploads.py.
    A terrain also becomes a preset (body and surface, and for a heightmap image its size and height range, are
    the user's); missing answers come back as questions with status 422."""
    received, total = [], 0
    for item in files:
        content = await item.read(MAX_UPLOAD_BYTES - total + 1)
        total += len(content)
        if total > MAX_UPLOAD_BYTES:
            return JSONResponse({"ok": False, "error": f"The upload is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
                                status_code=413)
        received.append((item.filename or "", content))
    try:
        record = await asyncio.to_thread(save_upload, kind, received, name, credit, license)
    except UploadError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    if kind != "terrain":
        return {"ok": True, "upload": record.card()}
    try:
        terrain = await asyncio.to_thread(prepare_terrain, record, body or None, surface or None,
                                          _number(size_m), _number(height_range_m))
    except UploadError as exc:
        discard_upload(record)   # nothing half-made stays in the Library; the user uploads again with answers
        questions = getattr(exc, "questions", None)
        return JSONResponse({"ok": False, "error": str(exc), **({"questions": questions} if questions else {})},
                            status_code=422 if isinstance(exc, NeedsAnswers) else 400)
    return {"ok": True, "upload": record.card(), "terrain": terrain}


def _number(text: str) -> float | None:
    try:
        return float(text) if text.strip() else None
    except ValueError:
        return None


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
