"""The Inspector: turns an uploaded robot (MJCF or URDF, with meshes) into a Library card.

1. Load it with MuJoCo using only files from the upload (a URDF is converted to MJCF with mj_saveLastXML).
2. Normalize it: one top body "simgate_root" with a free joint, the robot turned so its forward is +x,
   every joint named, motors added if it has no actuators, meshes copied into the upload's assets/ folder.
3. FACTS, measured in code and never guessed: bodies, joints, actuators, mass, bounding box, feet and wheel
   candidates, name hints.
4. A draft card from the Inspector agent (PLANNER_MODEL; rules in mock mode), checked against FACTS, with
   questions for anything uncertain. Code measures the rest: wheel sides, radii and spin, leg phases.
5. Checks: compiles, plausible mass and size, stands for 1 s, a wheeled robot drives 0.5 m, a legged robot's
   baseline gait is tried. With no open questions and passing checks the robot is listed: robot_card.json
   and policy_card.json (its built-in controller) are written to the upload folder.
"""
import json
import math
import posixpath
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Literal

import mujoco
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .. import config, llm
from .cards import CommandSpec, LegSpec, PolicyCard, RobotCard, WheelSpec
from .uploads import UploadError, UploadRecord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_BODY = "simgate_root"
FORWARD_YAW_DEG = {"+x": 0.0, "-x": 180.0, "+y": -90.0, "-y": 90.0}   # turn that brings that axis to +x
NAME_HINTS = ("hip", "thigh", "knee", "calf", "shin", "ankle", "foot", "toe", "wheel", "arm", "shoulder", "elbow",
              "wrist", "front", "rear", "back", "hind", "left", "right")
MASS_RANGE_KG, HEIGHT_RANGE_M = (0.1, 500.0), (0.05, 5.0)
STAND_S, DRIVE_S, GAIT_S, DRIVE_GOAL_M = 1.0, 4.0, 5.0, 0.5
QUESTION_CHOICES = {"forward_axis": list(FORWARD_YAW_DEG), "kind": ["wheeled", "legged", "humanoid", "other"],
                    "controller": ["wheels", "gait", "needs_policy"]}


# ── Draft (what the Inspector agent returns) ────────────────────────────────

class LegDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=40)
    hip: str
    knee: str | None = None


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: Literal["forward_axis", "kind", "controller"]
    question: str = Field(min_length=1, max_length=200)
    choices: list[str] = Field(min_length=2, max_length=6)


class RobotDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["wheeled", "legged", "humanoid", "other"]
    forward_axis: Literal["+x", "-x", "+y", "-y"] = "+x"
    wheels: list[str] = []          # wheel joint names
    legs: list[LegDraft] = []
    feet: list[str] = []            # foot body names
    standing_height_m: float = Field(gt=0)
    controller: Literal["wheels", "gait", "needs_policy"]
    questions: list[Question] = []
    notes: str = Field(default="", max_length=600)


INSPECTOR_SYSTEM = """You are the Inspector in a robot simulation sandbox. A user uploaded a robot model. Read the
FACTS (measured in code from the compiled model; they are always right), the file names and any README, and
write a draft Library card as one JSON object matching this schema:
{schema}
Rules:
- Use only joint and body names that appear in FACTS. wheels lists JOINT names (FACTS.candidate_wheels[].joint),
  feet lists BODY names, and a leg's hip and knee are JOINT names.
- kind: wheeled (driven by wheels), legged (4 or 6 legs), humanoid (2 legs, upright), other.
- controller: "wheels" for a wheeled robot (needs 2+ wheel joints), "gait" for a legged robot (needs 2+ legs,
  each with a hip joint), otherwise "needs_policy".
- A leg's hip is the joint that swings it forward and back; knee is a joint further down that bends it.
- forward_axis is the robot's forward direction in its own frame. If FACTS and names don't settle it, keep
  "+x" and add a question for field "forward_axis" with choices ["+x", "-x"]. Ask a question only when FACTS,
  names and the README leave it open (fields: forward_axis, kind, controller); never guess silently.
- standing_height_m: copy FACTS.standing_height_m.
Return only the JSON object."""


# ── Loading and normalizing ─────────────────────────────────────────────────

def _main_file(files: list[str], read) -> str:
    urdf = [path for path in files if path.lower().endswith(".urdf")]
    if urdf:
        return urdf[0]
    mjcf = [path for path in files if path.lower().endswith((".xml", ".mjcf"))
            and ET.fromstring(read(path)).tag == "mujoco" and b"<body" in read(path)]
    if not mjcf:
        raise UploadError("No robot model found: upload an MJCF (.xml/.mjcf with <mujoco> and bodies) or a URDF.")
    return sorted(mjcf, key=lambda path: (path.count("/"), len(path)))[0]


def _resolve(reference: str, base_dir: str, files: set[str]) -> str | None:
    """The upload file a model references: relative to the model's folder (URDF package:// paths from the
    upload's root), else a unique file with that name."""
    reference = re.sub(r"^package://[^/]+/", "", reference.replace("\\", "/"))
    for candidate in (posixpath.join(base_dir, reference) if base_dir else reference, reference):
        normal = posixpath.normpath(candidate) if candidate else ""   # "urdf/../meshes/a.stl" -> "meshes/a.stl"
        if normal in files:
            return normal
    named = [path for path in files if PurePosixPath(path).name == PurePosixPath(reference).name]
    return named[0] if len(named) == 1 else None


def load_robot(folder: Path, files: list[str]) -> tuple[mujoco.MjModel, str, dict[str, str]]:
    """Compile the robot from upload files only. Returns (model, the MuJoCo-saved MJCF, mesh file name in that
    MJCF -> upload path)."""
    read = lambda path: (folder / path).read_bytes()   # noqa: E731
    main = _main_file(files, read)
    base_dir = str(PurePosixPath(main).parent) if "/" in main else ""
    root = ET.fromstring(read(main))
    file_set = set(files)
    references: dict[str, str] = {}
    if main.lower().endswith(".urdf"):
        links = [link.get("name") for link in root.findall("link")]
        children = {joint.find("child").get("link") for joint in root.findall("joint") if joint.find("child") is not None}
        roots = [name for name in links if name not in children]
        if len(roots) != 1:
            raise UploadError(f"The URDF must have exactly one root link; it has {len(roots)}.")
        for mesh in root.iter("mesh"):
            target = _resolve(mesh.get("filename", ""), base_dir, file_set)
            if target is None:
                raise UploadError(f"{main} references {mesh.get('filename')!r}, which isn't in the upload.")
            mesh.set("filename", target)
        # keep the root link a body of its own (MuJoCo would merge it into the world) and keep visual meshes
        ET.SubElement(root, "link", name="simgate_anchor")
        joint = ET.SubElement(root, "joint", name="simgate_anchor_joint", type="fixed")
        ET.SubElement(joint, "parent", link="simgate_anchor")
        ET.SubElement(joint, "child", link=roots[0])
        extension = root.find("mujoco")   # many URDFs carry their own <mujoco> block; there may be only one
        if extension is None:
            extension = ET.SubElement(root, "mujoco")
        compiler = extension.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(extension, "compiler")
        for folder_attribute in ("meshdir", "assetdir", "texturedir"):   # mesh filenames now name upload files exactly
            compiler.attrib.pop(folder_attribute, None)
        compiler.attrib.update(discardvisual="false", fusestatic="false", strippath="false")
        text, assets = ET.tostring(root, encoding="unicode"), {path: read(path) for path in files}
    else:
        compiler = root.find("compiler")
        dirs = {key: (compiler.get(key, "") if compiler is not None else "") for key in ("meshdir", "texturedir", "assetdir")}
        for tag, key in (("mesh", "meshdir"), ("skin", "meshdir"), ("hfield", "assetdir"), ("texture", "texturedir")):
            for node in root.iter(tag):
                if node.get("file"):
                    folder_name = dirs[key] or dirs["assetdir"]
                    target = _resolve(str(PurePosixPath(folder_name) / node.get("file")) if folder_name else node.get("file"),
                                      base_dir, file_set)
                    if target is None:
                        raise UploadError(f"{main} references {node.get('file')!r}, which isn't in the upload.")
        text = read(main).decode("utf-8")
        assets = {}
        for path in files:   # every upload file, by its path from the model's folder
            relative = str(PurePosixPath(path).relative_to(base_dir)) if base_dir and path.startswith(base_dir + "/") else path
            assets[relative] = read(path)
    try:
        model = mujoco.MjModel.from_xml_string(text, assets)
    except ValueError as exc:
        raise UploadError(f"MuJoCo can't compile {main}: {exc}") from exc
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp) / "saved.xml"
        mujoco.mj_saveLastXML(str(saved), model)
        saved_text = saved.read_text(encoding="utf-8")
    for node in ET.fromstring(saved_text).iter("mesh"):
        name = node.get("file")
        if name:
            references[name] = _resolve(name, base_dir, file_set) or _resolve(PurePosixPath(name).name, "", file_set)
    return model, saved_text, references


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def normalize(saved_text: str, model: mujoco.MjModel, references: dict[str, str], forward_axis: str,
              folder: Path) -> tuple[str, list[str]]:
    """The robot as SimGate stores it (see the module docstring). Returns (MJCF text, notes for the user)."""
    root = ET.fromstring(saved_text)
    notes = []
    worldbody = root.find("worldbody")
    tops = worldbody.findall("body") if worldbody is not None else []
    if len(tops) != 1:
        raise UploadError(f"Upload one robot: the model has {len(tops)} top-level bodies.")
    dropped = [child.tag for child in list(worldbody) if child.tag != "body"]
    for child in list(worldbody):
        if child.tag != "body":
            worldbody.remove(child)
    if dropped:
        notes.append(f"Left out the scene parts around the robot ({', '.join(sorted(set(dropped)))}).")
    top = tops[0]
    top_name = top.get("name") or "simgate_body"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, top.get("name")) if top.get("name") else 1
    top.set("name", top_name)
    for joint in [child for child in list(top) if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free")]:
        top.remove(joint)
    for attribute in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
        top.attrib.pop(attribute, None)
    yaw = math.radians(FORWARD_YAW_DEG[forward_axis])
    turned = _quat_mul([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], [float(v) for v in model.body_quat[body_id]])
    top.set("quat", " ".join(f"{value:.6g}" for value in turned))
    worldbody.remove(top)
    wrapper = ET.SubElement(worldbody, "body", name=ROOT_BODY)
    ET.SubElement(wrapper, "freejoint", name=f"{ROOT_BODY}_free")
    wrapper.append(top)

    count = 0
    for joint in wrapper.iter("joint"):
        if not joint.get("name"):
            count += 1
            joint.set("name", f"joint_{count}")
    actuator = root.find("actuator")
    movable = [joint for joint in wrapper.iter("joint") if joint.get("type", "hinge") in ("hinge", "slide")]
    if actuator is None or not len(actuator):
        mass = float(model.body_mass.sum())
        actuator = actuator if actuator is not None else ET.SubElement(root, "actuator")
        for joint in movable:
            limit = joint.get("actuatorfrcrange")
            limit = max(abs(float(v)) for v in limit.split()) if limit else max(1.0, mass * 9.81 * 0.5)
            ET.SubElement(actuator, "motor", name=joint.get("name"), joint=joint.get("name"), gear="1",
                          ctrllimited="true", ctrlrange=f"{-limit:g} {limit:g}")
        if movable:
            notes.append(f"It had no actuators, so each of its {len(movable)} joints got a motor"
                         " (torque limits from the model's effort limits where given).")

    asset = root.find("asset")
    assets_dir = folder / "assets"
    shutil.rmtree(assets_dir, ignore_errors=True)
    if asset is not None:
        for index, mesh in enumerate(asset.findall("mesh")):
            if mesh.get("file"):
                source = references.get(mesh.get("file"))
                if source is None:
                    raise UploadError(f"Mesh {mesh.get('file')!r} isn't in the upload.")
                stored = f"{index:03d}_{PurePosixPath(source).name}"
                assets_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(folder / source, assets_dir / stored)
                mesh.set("file", stored)
                mesh.attrib.pop("content_type", None)
        for tag in ("texture", "hfield", "skin"):
            for node in [node for node in asset.findall(tag) if node.get("file")]:
                asset.remove(node)
                notes.append(f"Left out a {tag} image file (only meshes are loaded from uploads).")
        textures = {node.get("name") for node in asset.findall("texture")}
        for material in asset.findall("material"):
            if material.get("texture") and material.get("texture") not in textures:
                material.attrib.pop("texture")
    for extra in ("equality", "tendon", "contact", "sensor", "keyframe"):
        if root.find(extra) is not None and len(root.find(extra)):
            notes.append(f"Its <{extra}> section isn't used in SimGate scenes.")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib.clear()
    compiler.set("angle", "radian")
    compiler.set("meshdir", "assets")
    return ET.tostring(root, encoding="unicode"), notes


def _compile(text: str, folder: Path) -> mujoco.MjModel:
    assets_dir = folder / "assets"
    assets = {f"assets/{path.name}": path.read_bytes() for path in assets_dir.iterdir()} if assets_dir.is_dir() else {}
    return mujoco.MjModel.from_xml_string(text, assets)


# ── FACTS ───────────────────────────────────────────────────────────────────

def _name(model, kind, index) -> str:
    return mujoco.mj_id2name(model, kind, index) or ""


def _geom_bounds(model, data, geom) -> tuple[np.ndarray, np.ndarray]:
    center, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + center
    world = corners @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return world.min(axis=0), world.max(axis=0)


def _hints(name: str) -> list[str]:
    tokens = set(re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()))
    return [hint for hint in NAME_HINTS if hint in tokens or any(token.startswith(hint) for token in tokens)]


def facts(model: mujoco.MjModel) -> dict:
    """What the compiled, normalized robot is, measured in its default pose (root at the origin)."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    colliding = [g for g in range(model.ngeom) if model.geom_contype[g] or model.geom_conaffinity[g]]
    bounds = {g: _geom_bounds(model, data, g) for g in range(model.ngeom) if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE)}
    if not bounds or not colliding:
        raise UploadError("The robot has no collision geometry, so it can't touch the ground.")
    low = np.min([bounds[g][0] for g in bounds], axis=0)
    high = np.max([bounds[g][1] for g in bounds], axis=0)
    ground = min(bounds[g][0][2] for g in colliding)
    height = float(high[2] - low[2])
    joint_types = {int(mujoco.mjtJoint.mjJNT_FREE): "free", int(mujoco.mjtJoint.mjJNT_BALL): "ball",
                   int(mujoco.mjtJoint.mjJNT_SLIDE): "slide", int(mujoco.mjtJoint.mjJNT_HINGE): "hinge"}
    joints = []
    for j in range(model.njnt):
        joints.append({"name": _name(model, mujoco.mjtObj.mjOBJ_JOINT, j), "type": joint_types[int(model.jnt_type[j])],
                       "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])),
                       "range": [round(float(v), 4) for v in model.jnt_range[j]] if model.jnt_limited[j] else None,
                       "axis": [round(float(v), 3) for v in data.xaxis[j]], "hints": _hints(_name(model, mujoco.mjtObj.mjOBJ_JOINT, j))})
    actuators = []
    for a in range(model.nu):
        bias = model.actuator_biasprm[a]
        affine = int(model.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        kind = "position" if affine and bias[1] < 0 else "velocity" if affine and bias[2] < 0 else "motor"
        actuators.append({"name": _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a), "type": kind,
                          "joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0])),
                          "gear": round(float(model.actuator_gear[a, 0]), 4),
                          "ctrlrange": [round(float(v), 4) for v in model.actuator_ctrlrange[a]] if model.actuator_ctrllimited[a] else None,
                          "forcerange": [round(float(v), 4) for v in model.actuator_forcerange[a]] if model.actuator_forcelimited[a] else None})
    near_ground = max(0.01, 0.02 * height)
    feet = sorted({_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) for g in colliding
                   if bounds[g][0][2] <= ground + near_ground})
    wheels = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE) or model.jnt_limited[j]:
            continue
        body = int(model.jnt_bodyid[j])
        for g in colliding:
            if int(model.geom_bodyid[g]) != body or int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                continue
            geom_axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
            if abs(float(geom_axis @ data.xaxis[j])) > 0.9 and bounds[g][0][2] <= ground + near_ground:
                wheels.append({"joint": _name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
                               "body": _name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                               "radius_m": round(float(model.geom_size[g][0]), 4),
                               "position": [round(float(v), 3) for v in data.xpos[body]]})
                break
    bodies = [{"name": _name(model, mujoco.mjtObj.mjOBJ_BODY, b),
               "parent": _name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[b])),
               "mass_kg": round(float(model.body_mass[b]), 4),
               "position": [round(float(v), 3) for v in data.xpos[b]], "hints": _hints(_name(model, mujoco.mjtObj.mjOBJ_BODY, b))}
              for b in range(1, model.nbody)]
    return {"bodies": bodies, "joints": joints, "actuators": actuators,
            "total_mass_kg": round(float(model.body_mass.sum()), 4),
            "bounding_box_m": {"min": [round(float(v), 4) for v in low], "max": [round(float(v), 4) for v in high],
                               "size": [round(float(v), 4) for v in high - low]},
            "height_m": round(height, 4), "standing_height_m": round(max(-ground, 0.0) + 0.01, 4),
            "units_warning": (f"It is {height:.3f} m tall: if it should be bigger or smaller, the model may be in mm or "
                              "inches instead of metres.") if not HEIGHT_RANGE_M[0] <= height <= HEIGHT_RANGE_M[1] else None,
            "mesh_count": int(model.nmesh), "candidate_feet": feet, "candidate_wheels": wheels}


# ── The draft: rules, the agent, and the check against FACTS ──────────────────

def _chain(model, body_name: str) -> list[str]:
    """Hinge joints from the root down to this body."""
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joints = []
    while body > 0:
        joints = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.body_jntadr[body], model.body_jntadr[body] + model.body_jntnum[body])
                  if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE)] + joints
        body = int(model.body_parentid[body])
    return joints


def _pitch_joints(found: dict, chain: list[str]) -> list[str]:
    axes = {joint["name"]: joint["axis"] for joint in found["joints"]}
    return [name for name in chain if abs(axes[name][1]) > 0.7]


def rules_draft(model: mujoco.MjModel, found: dict) -> RobotDraft:
    wheel_joints = [wheel["joint"] for wheel in found["candidate_wheels"]]
    wheel_bodies = {wheel["body"] for wheel in found["candidate_wheels"]}
    legs, feet = {}, []
    for foot in found["candidate_feet"]:
        if foot in wheel_bodies:
            continue
        pitch = _pitch_joints(found, _chain(model, foot))
        if pitch and pitch[0] not in legs:
            legs[pitch[0]] = LegDraft(name=foot, hip=pitch[0], knee=pitch[1] if len(pitch) > 1 else None)
            feet.append(foot)
    hints = {hint for item in found["bodies"] + found["joints"] for hint in item["hints"]}
    if len(wheel_joints) >= 2:
        kind, controller = "wheeled", "wheels"
    elif len(legs) in (4, 6):
        kind, controller = "legged", "gait"
    elif len(legs) == 2:
        kind, controller = ("humanoid" if found["height_m"] > 0.4 or "arm" in hints else "legged"), "needs_policy"
    else:
        kind, controller = "other", "needs_policy"
    forward, sure = _forward_from_names(found) if kind != "humanoid" else _forward_from_feet(model, feet)
    questions = []
    if kind != "other" and not sure:
        questions.append(Question(field="forward_axis", question="Which way is forward: +x or -x?", choices=["+x", "-x"]))
    if len(wheel_joints) >= 2 and len(legs) >= 2:
        questions.append(Question(field="kind", question="It has wheels and legs. Should it drive or walk?",
                                  choices=["wheeled", "legged"]))
    return RobotDraft(kind=kind, forward_axis=forward, wheels=wheel_joints if kind == "wheeled" else [],
                      legs=list(legs.values()) if kind != "wheeled" else [], feet=feet,
                      standing_height_m=found["standing_height_m"], controller=controller, questions=questions,
                      notes="Drafted by the Inspector's rules.")


def _forward_from_names(found: dict) -> tuple[str, bool]:
    """+x or -x from where bodies named front/rear (fl/fr/rl/rr, ...) are; unsure when the names don't say."""
    xs = {"front": [], "rear": []}
    for body in found["bodies"]:
        tokens = set(re.findall(r"[a-z]+", body["name"].lower()))
        if tokens & {"front", "fl", "fr", "lf", "rf"}:
            xs["front"].append(body["position"][0])
        if tokens & {"rear", "back", "hind", "rl", "rr", "hl", "hr", "lh", "rh"}:
            xs["rear"].append(body["position"][0])
    if xs["front"] and xs["rear"] and abs(np.mean(xs["front"]) - np.mean(xs["rear"])) > 0.01:
        return ("+x" if np.mean(xs["front"]) > np.mean(xs["rear"]) else "-x"), True
    return "+x", False


def _forward_from_feet(model, feet: list[str]) -> tuple[str, bool]:
    """A humanoid's toes point forward: its foot geometry reaches further ahead of the ankle than behind it."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    offsets = []
    for foot in feet:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot)
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == body and (model.geom_contype[g] or model.geom_conaffinity[g]):
                low, high = _geom_bounds(model, data, g)
                offsets.append((low[0] + high[0]) / 2 - data.xpos[body][0])
    if offsets and min(offsets) > 0.01:
        return "+x", True
    if offsets and max(offsets) < -0.01:
        return "-x", True
    return "+x", False


def check_draft(draft: RobotDraft, found: dict, model) -> list[str]:
    """Where the draft disagrees with FACTS; empty when it can be used."""
    joints = {joint["name"]: joint for joint in found["joints"]}
    bodies = {body["name"] for body in found["bodies"]}
    errors = []
    for name in draft.wheels:
        if name not in joints or joints[name]["type"] != "hinge":
            errors.append(f"wheel {name!r} is not a hinge joint in FACTS.joints")
    for leg in draft.legs:
        for role, name in (("hip", leg.hip), ("knee", leg.knee)):
            if name is not None and (name not in joints or joints[name]["type"] != "hinge"):
                errors.append(f"leg {leg.name!r}: {role} {name!r} is not a hinge joint in FACTS.joints")
        if leg.knee and leg.hip in joints and leg.knee in joints and leg.hip not in _chain(model, joints[leg.knee]["body"]):
            errors.append(f"leg {leg.name!r}: knee {leg.knee!r} is not below hip {leg.hip!r}")
    for foot in draft.feet:
        if foot not in bodies:
            errors.append(f"foot {foot!r} is not a body in FACTS.bodies")
    if draft.controller == "wheels" and len(draft.wheels) < 2:
        errors.append("controller 'wheels' needs at least 2 wheel joints")
    if draft.controller == "gait" and len(draft.legs) < 2:
        errors.append("controller 'gait' needs at least 2 legs with a hip joint")
    measured = found["standing_height_m"]
    if not 0.5 * measured <= draft.standing_height_m <= 1.5 * measured:
        errors.append(f"standing_height_m {draft.standing_height_m} is far from FACTS.standing_height_m {measured}")
    for question in draft.questions:
        bad = [choice for choice in question.choices if choice not in QUESTION_CHOICES[question.field]]
        if bad:
            errors.append(f"question about {question.field} has invalid choices {bad}; "
                          f"allowed: {QUESTION_CHOICES[question.field]}")
    return errors


def draft_card(model: mujoco.MjModel, found: dict, files: list[str], readme: str) -> tuple[RobotDraft, str, list[dict]]:
    """(draft, who wrote it: "model" or "rules", the model calls). The model's draft must pass check_draft,
    with one retry that shows it the exact errors; otherwise the rules draft is used."""
    rules = rules_draft(model, found)
    if config.MOCK_MODE:
        return rules, "rules", []

    def parse(reply: dict) -> RobotDraft:
        draft = RobotDraft.model_validate(reply)
        errors = check_draft(draft, found, model)
        if errors:
            raise ValueError("The draft disagrees with FACTS: " + "; ".join(errors))
        return draft

    messages = [{"role": "system", "content": INSPECTOR_SYSTEM.replace("{schema}", json.dumps(RobotDraft.model_json_schema()))},
                {"role": "user", "content": f"FACTS:\n{json.dumps(found)}\n\nFiles: {', '.join(files)}\n\nREADME:\n{readme[:4000] or '(none)'}"}]
    try:
        draft, calls = llm.chat_json(messages, config.PLANNER_MODEL, parse)
    except Exception as exc:  # noqa: BLE001 - a model outage falls back to the rules
        return rules, "rules", [{"error": f"{type(exc).__name__}: {exc}"[:300]}]
    if draft is None:
        return rules, "rules", calls
    # a question the measurements already answer: the rules are sure of that field and the model agrees with them
    open_fields = {question.field for question in rules.questions}
    settled = [q.field for q in draft.questions if q.field not in open_fields and getattr(draft, q.field) == getattr(rules, q.field)]
    if settled:
        draft.questions = [q for q in draft.questions if q.field not in settled]
        calls.append({"dropped_questions": settled, "why": "FACTS and names already settle them"})
    return draft, "model", calls


# ── Card details measured in code, checks, listing ───────────────────────────

def _specs(model: mujoco.MjModel, found: dict, draft: RobotDraft) -> tuple[list[WheelSpec], float | None, list[LegSpec]]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    candidates = {wheel["joint"]: wheel for wheel in found["candidate_wheels"]}
    wheels = []
    for name in draft.wheels:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        body = int(model.jnt_bodyid[joint])
        radius = candidates[name]["radius_m"] if name in candidates else 0.1
        spin = 1 if float(data.xaxis[joint][1]) >= 0 else -1   # rolling forward (+x) spins about +y
        wheels.append(WheelSpec(joint=name, side="left" if data.xpos[body][1] > 0 else "right", radius_m=radius, spin_sign=spin))
    lefts = [float(data.xpos[int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, w.joint)])][1]) for w in wheels if w.side == "left"]
    rights = [float(data.xpos[int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, w.joint)])][1]) for w in wheels if w.side == "right"]
    track = float(np.mean(lefts) - np.mean(rights)) if lefts and rights else None

    legs = []
    feet = {leg.name: (draft.feet[i] if i < len(draft.feet) else None) for i, leg in enumerate(draft.legs)}
    for leg in draft.legs:
        foot = feet.get(leg.name) or leg.name
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, foot)
        if foot_id < 0:   # the leg's lowest body: the knee's (or hip's) body
            foot_id = int(model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, leg.knee or leg.hip)])
        legs.append((leg, foot_id, float(data.xpos[foot_id][0]), float(data.xpos[foot_id][1])))
    specs = []
    for side in ("left", "right"):
        same_side = sorted([item for item in legs if (item[3] > 0) == (side == "left")], key=lambda item: -item[2])
        for rank, (leg, foot_id, _, _) in enumerate(same_side):   # diagonal pairs / tripods: alternate along each side
            phase = 0.0 if (rank + (side == "right")) % 2 == 0 else math.pi
            specs.append(LegSpec(name=leg.name, hip=leg.hip, knee=leg.knee, foot=_name(model, mujoco.mjtObj.mjOBJ_BODY, foot_id),
                                 side=side, phase=phase, hip_sign=_moves(model, leg.hip, foot_id, axis=0, sign=-1),
                                 knee_sign=_moves(model, leg.knee, foot_id, axis=2, sign=1) if leg.knee else 1))
    return wheels, track, specs


def _moves(model, joint_name: str, body: int, axis: int, sign: int) -> int:
    """+1 if turning the joint positively moves the body along `sign` on `axis` (x: backward is -1, z: up is +1)."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    before = float(data.xpos[body][axis])
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[model.jnt_qposadr[joint]] += 0.05
    mujoco.mj_kinematics(model, data)
    return 1 if (float(data.xpos[body][axis]) - before) * sign > 0 else -1


def _cards(record: UploadRecord, model, found: dict, draft: RobotDraft, mjcf_path: str) -> tuple[RobotCard, PolicyCard]:
    wheels, track, legs = _specs(model, found, draft)
    controller = {"wheels": "wheels", "gait": "gait"}.get(draft.controller, "hold")
    actuated = [a["joint"] for a in found["actuators"] if a["joint"]]
    joint_order = list(dict.fromkeys(actuated))
    defaults = {_name(model, mujoco.mjtObj.mjOBJ_JOINT, j): float(model.qpos0[model.jnt_qposadr[j]]) for j in range(model.njnt)}
    mass, height = found["total_mass_kg"], max(found["height_m"], 0.05)
    kp = max(1.0, 0.5 * mass * 9.81 * height)          # holds the robot's weight over its height
    wheel_names = {wheel.joint for wheel in wheels}
    wheel_kd = max(0.05, 2.0 * mass * (wheels[0].radius_m if wheels else 0.1))   # torque per rad/s of speed error
    kds = [wheel_kd if name in wheel_names else kp / 20 for name in joint_order]
    types = {a["type"] for a in found["actuators"]}
    actuator_type = "position" if types == {"position"} else "velocity" if types == {"velocity"} else "torque"
    robot = RobotCard(id=record.id, name=record.name, license=record.license, credit=record.credit, kind=draft.kind,
                      source="upload", mjcf_path=mjcf_path, root_body=ROOT_BODY, joint_names=joint_order,
                      actuator_type=actuator_type, standing_height_m=found["standing_height_m"], mass_kg=mass,
                      controller=controller, wheels=wheels, track_m=track, legs=legs, feet=draft.feet)
    label = {"wheels": "Wheel controller", "gait": "Baseline gait",
             "hold": "Holds its pose (needs a trained policy to move)"}[controller]
    policy = PolicyCard(id=f"{record.id}-{controller}", name=label, robot_id=record.id, format="builtin",
                        control_dt=0.02, joint_order=joint_order, default_angles=[defaults.get(n, 0.0) for n in joint_order],
                        kps=[kp] * len(joint_order), kds=kds,
                        command=CommandSpec(names=["vx", "vy", "yaw_rate"], ranges=[(-1.0, 1.0)] * 3,
                                            default=[0.5, 0.0, 0.0]),   # with no goal it moves, like the G1
                        training_notes=["A simple built-in controller, not a trained policy."
                                        + (" The baseline gait may walk badly." if controller == "gait" else "")],
                        license="SimGate built-in controller", credit="SimGate")
    return robot, policy


def run_checks(text: str, folder: Path, robot: RobotCard, policy: PolicyCard, found: dict) -> list[dict]:
    """Compiles; plausible mass and size; stands 1 s on flat ground; drives 0.5 m (wheels); baseline gait."""
    from ..simple_control import SimpleController

    checks = []
    mass, height = found["total_mass_kg"], found["height_m"]
    checks.append({"name": "Mass and size are plausible",
                   "passed": MASS_RANGE_KG[0] <= mass <= MASS_RANGE_KG[1] and HEIGHT_RANGE_M[0] <= height <= HEIGHT_RANGE_M[1],
                   "detail": f"{mass:.2f} kg, {height:.3f} m tall (expected {MASS_RANGE_KG[0]:g}-{MASS_RANGE_KG[1]:g} kg, "
                             f"{HEIGHT_RANGE_M[0]:g}-{HEIGHT_RANGE_M[1]:g} m)." + (f" {found['units_warning']}" if found["units_warning"] else "")})
    root = ET.fromstring(text)
    worldbody = root.find("worldbody")
    worldbody.insert(0, ET.Element("geom", name="simgate_floor", type="plane", size="20 20 0.1"))
    worldbody.find(f"body[@name='{ROOT_BODY}']").set("pos", f"0 0 {robot.standing_height_m:.4f}")
    for option in root.findall("option"):
        root.remove(option)
    root.insert(1, ET.Element("option", timestep="0.002"))
    try:
        model = _compile(ET.tostring(root, encoding="unicode"), folder)
    except ValueError as exc:
        return checks + [{"name": "Model compiles", "passed": False, "detail": str(exc)[:300]}]
    checks.append({"name": "Model compiles", "passed": True, "detail": "MuJoCo compiled it with a floor."})

    def simulate(seconds: float, command: dict | None) -> tuple[mujoco.MjData, float, float]:
        data = mujoco.MjData(model)
        controller = SimpleController(robot, policy, model, data)
        mujoco.mj_forward(model, data)
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        for _ in range(int(0.3 / model.opt.timestep)):    # settle onto the floor in the default pose
            controller.hold_default()
            mujoco.mj_step(model, data)
        start, peak = data.xpos[body].copy(), 0.0
        if command:
            controller.set_command(command)
        for step in range(int(seconds / model.opt.timestep)):
            controller.step(step * model.opt.timestep) if command else controller.hold_default()
            mujoco.mj_step(model, data)
            peak = max(peak, math.degrees(math.acos(float(np.clip(data.xmat[body][8], -1, 1)))))
        return data, float(data.xpos[body][0] - start[0]), peak if np.isfinite(data.qpos).all() else float("nan")

    data, _, tilt = simulate(STAND_S, None)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    drop = robot.standing_height_m - float(data.xpos[body][2])
    stands = math.isfinite(tilt) and tilt < 30 and drop < 0.3 * robot.standing_height_m + 0.02
    checks.append({"name": "Stands for 1 s on flat ground", "passed": bool(stands),
                   "detail": f"Holding its default pose: peak tilt {tilt:.0f}°, root {drop:+.3f} m lower than placed."})
    if robot.controller == "wheels":
        _, progress, tilt = simulate(DRIVE_S, {"vx": 0.5})
        checks.append({"name": "Drives forward 0.5 m with the wheel controller", "passed": progress >= DRIVE_GOAL_M,
                       "detail": f"Moved {progress:+.2f} m along +x in {DRIVE_S:g} s at 0.5 m/s (peak tilt {tilt:.0f}°)."})
    elif robot.controller == "gait":
        _, progress, tilt = simulate(GAIT_S, {"vx": 0.6})
        fell = not math.isfinite(tilt) or tilt > 60
        checks.append({"name": "Baseline gait tried", "passed": True, "advisory": True,
                       "detail": f"Baseline gait: moved {progress:+.2f} m in {GAIT_S:g} s"
                                 + (", and fell over." if fell else f", peak tilt {tilt:.0f}°.")
                                 + " It is a simple gait, not a trained policy, so it may walk badly."})
    return checks


def _upload_facts(found: dict, source_model: mujoco.MjModel, references: dict) -> dict:
    """What the uploaded model itself had, before SimGate added its root free joint and copied meshes."""
    found["free_joint_in_upload"] = any(int(t) == int(mujoco.mjtJoint.mjJNT_FREE) for t in source_model.jnt_type)
    found["mesh_files"] = len({path for path in references.values() if path})
    return found


def _readme(folder: Path, files: list[str]) -> str:
    texts = [(folder / path).read_text(encoding="utf-8", errors="replace") for path in files if path.lower().endswith((".md", ".txt"))]
    return "\n\n".join(texts)


def _state_path(record: UploadRecord) -> Path:
    return Path(record.folder) / "inspection.json"


def inspect_robot(record: UploadRecord) -> dict:
    """Load, normalize, measure, draft, check and (if nothing is open) list an uploaded robot."""
    folder = Path(record.folder)
    model, saved, references = load_robot(folder, record.files)
    (folder / "converted.xml").write_text(saved, encoding="utf-8")
    (folder / "references.json").write_text(json.dumps(references), encoding="utf-8")
    text, notes = normalize(saved, model, references, "+x", folder)
    found = _upload_facts(facts(_compile(text, folder)), model, references)
    draft, author, calls = draft_card(_compile(text, folder), found, record.files, _readme(folder, record.files))
    state = {"upload_id": record.id, "inspector": author, "model_calls": calls, "draft": draft.model_dump(),
             "questions": [question.model_dump() for question in draft.questions], "answers": {},
             "conversion_notes": notes}
    return _finish(record, state)


def answer_robot(record: UploadRecord, answers: dict) -> dict:
    """Apply the user's answers (no re-upload), re-check, and list the robot when nothing is open."""
    path = _state_path(record)
    if not path.is_file():
        raise UploadError(f"No inspection found for upload {record.id}.")
    state = json.loads(path.read_text(encoding="utf-8"))
    draft = state["draft"]
    for field, value in answers.items():
        if field not in QUESTION_CHOICES or value not in QUESTION_CHOICES[field]:
            raise UploadError(f"Can't set {field!r} to {value!r}.")
        state["answers"][field] = value
        draft[field] = value
        if field == "kind":   # a changed kind takes the controller that fits it
            draft["controller"] = {"wheeled": "wheels", "legged": "gait"}.get(value, "needs_policy")
        state["questions"] = [q for q in state["questions"] if q["field"] != field]
    if (draft["controller"] == "wheels" and not draft["wheels"]) or (draft["controller"] == "gait" and not draft["legs"]):
        model = _compile((Path(record.folder) / "model.xml").read_text(encoding="utf-8"), Path(record.folder))
        rules = rules_draft(model, facts(model))
        draft.update(wheels=rules.wheels, legs=[leg.model_dump() for leg in rules.legs] or draft["legs"], feet=rules.feet or draft["feet"])
    state["draft"] = draft
    return _finish(record, state)


def _finish(record: UploadRecord, state: dict) -> dict:
    """Normalize with the current forward axis, measure, build the cards, run the checks, list if all is well."""
    folder = Path(record.folder)
    draft = RobotDraft.model_validate({**state["draft"], "questions": []})
    saved = (folder / "converted.xml").read_text(encoding="utf-8")
    references = json.loads((folder / "references.json").read_text(encoding="utf-8"))
    source_model, _, _ = load_robot(folder, record.files)
    text, _ = normalize(saved, source_model, references, draft.forward_axis, folder)
    (folder / "model.xml").write_text(text, encoding="utf-8")
    model = _compile(text, folder)
    found = _upload_facts(facts(model), source_model, references)
    mjcf_path = (folder / "model.xml").resolve().relative_to(PROJECT_ROOT).as_posix()
    robot, policy = _cards(record, model, found, draft, mjcf_path)
    checks = run_checks(text, folder, robot, policy, found)
    passed = all(check["passed"] or check.get("advisory") for check in checks)
    status = "needs_answers" if state["questions"] else "listed" if passed else "checks_failed"
    for name in ("robot_card.json", "policy_card.json"):
        (folder / name).unlink(missing_ok=True)
    if status == "listed":
        (folder / "robot_card.json").write_text(robot.model_dump_json(indent=2), encoding="utf-8")
        (folder / "policy_card.json").write_text(policy.model_dump_json(indent=2), encoding="utf-8")
    state.update(status=status, facts=found, checks=checks, robot_card=robot.model_dump(), policy_card=policy.model_dump(),
                 baseline_gait=robot.controller == "gait")
    _state_path(record).write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state
