"""Web server: serves the sandbox UI and streams agent events over a WebSocket.

Run:  python server.py   (or uvicorn server:app --reload)   then open http://localhost:8000
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from simgate import config, llm
from simgate.library import LibraryRegistry, UploadError, save_upload
from simgate.library.inspector import answer_robot, inspect_robot
from simgate.library.terrain_upload import NeedsAnswers, prepare_terrain
from simgate.library.uploads import MAX_UPLOAD_BYTES, discard_upload, get_upload, list_uploads, upload_status
from simgate.demos import DEMOS_DIR, list_demos
from simgate.orchestrator import Session
from simgate.terrain_data import list_presets, load_ortho_image, load_preview_image
from simgate.thumbnails import CARD_TITLES, shaded_relief, thumbnail_url, webp_bytes

STATIC = Path(__file__).parent / "static"
INSIGHT_GLB = Path(__file__).parent / "assets" / "nasa" / "insight_lander.glb"
log = logging.getLogger("uvicorn.error")


def startup_checks() -> str:
    """Check the Meta key once (switching to mock mode if it's missing or invalid) and note missing optional assets.
    Never raises: a bad key or a missing asset is a status line, not a crash."""
    status = llm.check_key()
    (log.info if status.startswith("Connected") else log.warning)("SimGate: %s", status)
    log.info("SimGate: %s", config.mode_label())
    if not INSIGHT_GLB.is_file():
        log.warning("SimGate: NASA InSight model not found (%s); Mars landers use the simple fallback shape. "
                    "Run `python scripts/fetch_assets.py` to download it.", INSIGHT_GLB.relative_to(Path(__file__).parent))
    return status


@asynccontextmanager
async def lifespan(_app):
    await asyncio.to_thread(startup_checks)
    yield


app = FastAPI(title="SimGate", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class _WebSocketEmitter:
    def __init__(self, socket: WebSocket):
        self.socket = socket
        self.disconnected = False

    async def __call__(self, event: dict) -> None:
        if self.disconnected:
            raise WebSocketDisconnect(code=1006)
        payload = json.dumps(event)
        try:
            await self.socket.send_text(payload)
        except Exception as error:
            self.disconnected = True
            raise WebSocketDisconnect(code=1006) from error


@app.get("/")
def index():
    return FileResponse(STATIC / "vendor" / "index.html")


@app.get("/assets/nasa/insight_lander.glb")
def insight_visual():
    if not INSIGHT_GLB.is_file():
        return JSONResponse({"detail": "Local NASA model is not installed."}, status_code=404)
    return FileResponse(INSIGHT_GLB, media_type="model/gltf-binary")


def library_view() -> dict:
    """Every Library card, plus each upload's state (listed, needs answers, not ready with reasons)."""
    library = LibraryRegistry()
    presets = list_presets()
    worlds = []
    for card in library.worlds():
        world = card.model_dump(mode="json")
        if card.terrain_kind == "indoor":
            from simgate.indoor import get_layout

            world["indoor_layout"] = get_layout(card.terrain_preset)
        if card.terrain_preset and (load_ortho_image(card.terrain_preset) is not None
                       or load_preview_image(card.terrain_preset) is not None):
            world["preview_url"] = f"/terrain-preview/{card.terrain_preset}"
        if card.terrain_kind == "real":   # the home card: a plain title, its notes kept as details, and a picture
            meta = presets.get(card.terrain_preset, {})
            world["card_title"] = CARD_TITLES.get(card.terrain_preset) or meta.get("site") or meta.get("label")
            world["accuracy_note"] = meta.get("accuracy_note") or ""
            world["thumbnail_url"] = thumbnail_url(card.id) or f"/terrain-thumbnail/{card.terrain_preset}"
        elif card.terrain_kind == "indoor":
            world["thumbnail_url"] = thumbnail_url(card.id)
        worlds.append(world)
    return {"robots": [card.model_dump(mode="json") for card in library.robots()],
            "controllers": [card.model_dump(mode="json") for card in library.controllers()],
            "worlds": worlds,
            "uploads": [upload_status(record) for record in list_uploads()]}


@app.get("/terrain-preview/{name}")
def terrain_preview(name: str):
    image = load_ortho_image(name) or load_preview_image(name)
    if image is None:
        return JSONResponse({"ok": False, "error": "No terrain preview is available."}, status_code=404)
    data, _ = image
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/terrain-thumbnail/{name}")
def terrain_thumbnail(name: str):
    """A smooth shaded-relief card picture for a terrain without a shipped 3D thumbnail (an uploaded one)."""
    if name not in list_presets():
        return JSONResponse({"ok": False, "error": "No such terrain."}, status_code=404)
    return Response(_relief_thumbnail(name), media_type="image/webp", headers={"Cache-Control": "public, max-age=3600"})


@lru_cache(maxsize=32)
def _relief_thumbnail(name: str) -> bytes:
    from simgate.terrain_data import load_preset

    heights, meta = load_preset(name)
    return webp_bytes(shaded_relief(heights, float(meta["size_m"]), meta.get("body", "earth")))


@app.get("/demos")
def demo_list():
    return list_demos()


@app.get("/demos/{demo_id}")
def demo(demo_id: str):
    """One saved demo's recording (for the browser-side evolution replay), sent as the gzipped file it is."""
    entry = next((demo for demo in list_demos() if demo["id"] == demo_id), None)
    if entry is None:
        return JSONResponse({"detail": f"There is no saved demo called {demo_id!r}."}, status_code=404)
    return Response((DEMOS_DIR / entry["file"]).read_bytes(), media_type="application/json",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=3600"})


@app.get("/library")
def library():
    return library_view()


@app.get("/health")
def health():
    return {"ok": True, "mock_mode": config.MOCK_MODE, "mode": config.mode_label(), "mode_status": config.MODE_STATUS,
            "planner_model": config.PLANNER_MODEL, "builder_model": config.BUILDER_MODEL, "builder_mode": config.BUILDER_MODE}


@app.post("/upload")
async def upload(kind: str = Form(..., alias="type"), name: str = Form(""), credit: str = Form(""),
                 license: str = Form(""), files: list[UploadFile] = File(...), body: str = Form(""),
                 surface: str = Form(""), size_m: str = Form(""), height_range_m: str = Form("")):
    """Store an uploaded terrain or robot (files or one .zip) after the checks in library/uploads.py.
    A terrain also becomes a preset (body and surface, and for a heightmap image its size and height range, are
    the user's); missing answers come back as questions with status 422. A robot is inspected: its card draft,
    questions and checks come back, and it is listed once nothing is open and the checks pass. Trained
    controllers (.onnx, .pt, ...) are rejected: SimGate doesn't run uploaded AI controllers."""
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
    if kind == "robot":
        try:
            robot = await asyncio.to_thread(inspect_robot, record)
        except UploadError as exc:
            discard_upload(record)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return {"ok": True, "upload": record.card(), "robot": robot}
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


@app.post("/upload/{kind}/{upload_id}/answers")
async def upload_answers(kind: str, upload_id: str, payload: dict = Body(...)):
    """Answers to the Inspector's questions about a robot, e.g. {"answers": {"forward_axis": "-x"}}:
    the card is updated, re-checked, and listed when nothing is open. No re-upload."""
    answer = {"robot": answer_robot}.get(kind)
    answers = payload.get("answers")
    if answer is None:
        return JSONResponse({"ok": False, "error": "Only robot uploads take answers."}, status_code=404)
    if not isinstance(answers, dict) or not answers:
        return JSONResponse({"ok": False, "error": "Send {\"answers\": {field: choice}}."}, status_code=400)
    try:
        record = get_upload(kind, upload_id)
        result = await asyncio.to_thread(answer, record, answers)
    except UploadError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=404 if str(exc).startswith("No ") else 400)
    return {"ok": True, kind: result}


def _number(text: str) -> float | None:
    try:
        return float(text) if text.strip() else None
    except ValueError:
        return None


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    emit = _WebSocketEmitter(socket)
    session = Session(emit)
    terrains = [{"name": n, "label": m["label"], "body": m.get("body")} for n, m in list_presets().items()]
    await emit({"type": "hello", "mock_mode": config.MOCK_MODE, "model": None if config.MOCK_MODE else config.PLANNER_MODEL,
                "mode": config.mode_label(), "mode_status": config.MODE_STATUS, "demos": list_demos(),
                "terrains": terrains, "library": library_view()})
    try:
        while True:
            msg = json.loads(await socket.receive_text())
            if msg.get("type") == "reset":
                session = Session(emit)
                await emit({"type": "reset"})
            elif msg.get("type") == "workflow_action":
                await session.handle_workflow_action(msg.get("workflow", ""), msg.get("robot_id"))
            elif msg.get("type") == "play_demo":
                await session.play_demo(msg.get("demo_id", ""))
            elif msg.get("type") == "replay_map_run":
                await session.replay_map_run(msg.get("run_id", ""))
            elif msg.get("type") == "structure_action":
                await session.handle_structure_action(msg.get("operation", ""), msg.get("structure_index"))
            elif msg.get("type") in ("set_start", "set_goal"):
                await session.set_point(msg["type"][4:], float(msg["x"]), float(msg["y"]), msg.get("robot_id"))
            elif msg.get("type") == "show_design":
                await session.show_design(msg.get("robot_id"))
            elif msg.get("type") == "restore_scene":
                await session.restore_scene(msg.get("scene_spec") or {})
            elif msg.get("type") == "library_select":
                await session.select_library(msg["robot_id"], msg["controller_id"], msg["world_id"],
                                             start_scene=bool(msg.get("start_scene")),
                                             terrain_kind=msg.get("terrain_kind", "hills"),
                                             title=msg.get("title", ""), obstacles=msg.get("obstacles"),
                                             scene_spec=msg.get("scene_spec"), robot_settings=msg.get("robot_settings"))
            elif text := (msg.get("text") or "").strip():
                await session.handle(text)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("SIMGATE_HOST", "127.0.0.1"), port=int(os.getenv("SIMGATE_PORT", "8000")))
