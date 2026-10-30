"""Web server: serves the sandbox UI and streams agent events over a WebSocket.

Run:  uvicorn server:app --reload   then open http://localhost:8000
"""
import asyncio
import json
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from simgate import config
from simgate.library import LibraryRegistry, UploadError, save_upload
from simgate.library.adapter import adapt_policy, answer_policy
from simgate.library.inspector import answer_robot, inspect_robot
from simgate.library.terrain_upload import NeedsAnswers, prepare_terrain
from simgate.library.uploads import MAX_UPLOAD_BYTES, discard_upload, get_upload, list_uploads, upload_status
from simgate.orchestrator import Session
from simgate.terrain_data import list_presets

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="SimGate")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "vendor" / "index.html")


def library_view() -> dict:
    """Every Library card, plus each upload's state (listed, needs answers, not ready with reasons)."""
    library = LibraryRegistry()
    return {"robots": [card.model_dump(mode="json") for card in library.robots()],
            "policies": [card.model_dump(mode="json") for card in library.policies()],
            "worlds": [card.model_dump(mode="json") for card in library.worlds()],
            "uploads": [upload_status(record) for record in list_uploads()]}


@app.get("/library")
def library():
    return library_view()


@app.get("/health")
def health():
    return {"ok": True, "mock_mode": config.MOCK_MODE, "planner_model": config.PLANNER_MODEL, "builder_mode": config.BUILDER_MODE}


@app.post("/upload")
async def upload(kind: str = Form(..., alias="type"), name: str = Form(""), credit: str = Form(""),
                 license: str = Form(""), files: list[UploadFile] = File(...), body: str = Form(""),
                 surface: str = Form(""), size_m: str = Form(""), height_range_m: str = Form(""),
                 robot_id: str = Form("")):
    """Store an uploaded terrain, robot or policy (files or one .zip) after the checks in library/uploads.py.
    A terrain also becomes a preset (body and surface, and for a heightmap image its size and height range, are
    the user's); missing answers come back as questions with status 422. A robot is inspected: its card draft,
    questions and checks come back, and it is listed once nothing is open and the checks pass. A policy (ONNX)
    is adapted to robot_id, a Library robot, in the same way."""
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
    if kind == "policy":
        robots = [card for card in LibraryRegistry().robots() if card.joint_names]
        if robot_id not in {card.id for card in robots}:
            discard_upload(record)
            question = {"field": "robot_id", "question": "Which Library robot is this policy for?",
                        "choices": [card.id for card in robots]}
            return JSONResponse({"ok": False, "error": question["question"], "questions": [question]}, status_code=422)
        try:
            policy = await asyncio.to_thread(adapt_policy, record, robot_id)
        except UploadError as exc:
            discard_upload(record)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return {"ok": True, "upload": record.card(), "policy": policy}
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
    """Answers to the Inspector's (robot) or Adapter's (policy) questions, e.g. {"answers": {"forward_axis": "-x"}}:
    the card is updated, re-checked, and listed when nothing is open. No re-upload."""
    answer = {"robot": answer_robot, "policy": answer_policy}.get(kind)
    answers = payload.get("answers")
    if answer is None:
        return JSONResponse({"ok": False, "error": "Only robot and policy uploads take answers."}, status_code=404)
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

    async def emit(event: dict) -> None:
        await socket.send_text(json.dumps(event))

    session = Session(emit)
    terrains = [{"name": n, "label": m["label"], "body": m.get("body")} for n, m in list_presets().items()]
    await emit({"type": "hello", "mock_mode": config.MOCK_MODE, "model": None if config.MOCK_MODE else config.PLANNER_MODEL,
                "terrains": terrains, "library": library_view()})
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
