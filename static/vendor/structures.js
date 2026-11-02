import * as THREE from "three";
import { GLTFLoader } from "./loaders/GLTFLoader.js";

const NASA_INSIGHT = "/assets/nasa/insight_lander.glb";
let nasaEnvironment = null;

function reflectionEnvironment() {
  if (nasaEnvironment) return nasaEnvironment;
  const faces = ["#c7cdd0", "#bcc5ca", "#eef0ed", "#7d756b", "#d8d6cc", "#c5ced1"].map((color) => {
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 16;
    const context = canvas.getContext("2d"); context.fillStyle = color; context.fillRect(0, 0, 16, 16);
    return canvas;
  });
  nasaEnvironment = new THREE.CubeTexture(faces); nasaEnvironment.colorSpace = THREE.SRGBColorSpace;
  nasaEnvironment.needsUpdate = true;
  return nasaEnvironment;
}

function metal(color, roughness = 0.45, metalness = 0.5) {
  return new THREE.MeshStandardMaterial({ color, roughness, metalness });
}

function addMesh(group, geometry, material, position = [0, 0, 0], name = "") {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position); mesh.name = name;
  mesh.castShadow = true; mesh.receiveShadow = true; group.add(mesh);
  return mesh;
}

function box(group, size, material, position, name) {
  return addMesh(group, new THREE.BoxGeometry(...size), material, position, name);
}

function tube(group, start, end, radius, material, name) {
  const from = new THREE.Vector3(...start), to = new THREE.Vector3(...end);
  const mesh = addMesh(group, new THREE.CylinderGeometry(radius, radius, from.distanceTo(to), 12),
    material, from.clone().add(to).multiplyScalar(0.5).toArray(), name);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), to.sub(from).normalize());
  return mesh;
}

function foilMaterial() {
  const canvas = document.createElement("canvas"); canvas.width = canvas.height = 128;
  const context = canvas.getContext("2d"); context.fillStyle = "#d8aa43"; context.fillRect(0, 0, 128, 128);
  for (let index = 0; index < 70; index++) {
    const x = (index * 47) % 128, y = (index * 71) % 128;
    context.strokeStyle = index % 2 ? "#eed58a" : "#997020"; context.lineWidth = 0.8;
    context.beginPath(); context.moveTo(x, y); context.lineTo((x + 31) % 128, (y + 53) % 128); context.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.MeshStandardMaterial({ map: texture, bumpMap: texture, bumpScale: 0.008,
    color: 0xffffff, metalness: 0.75, roughness: 0.4 });
}

function habitatSkin(record, group) {
  const dimensions = record.dimensions, length = dimensions.module_length_m;
  const radius = dimensions.module_diameter_m / 2, rise = dimensions.airlock_rise_m;
  const shell = metal(0xd8dedb, 0.65, 0.18), dark = metal(0x33474a), alloy = metal(0xa6b8bc);
  const window = metal(0x234f60, 0.12, 0.65), segments = Math.max(3, Math.round(length / 1.2));
  for (let index = 0; index < segments; index++) {
    const segment = addMesh(group, new THREE.CylinderGeometry(radius, radius, length / segments - 0.035, 40),
      shell, [-length / 2 + (index + 0.5) * length / segments, 0, radius], "habitat_segment");
    segment.rotation.z = Math.PI / 2;
  }
  for (let index = 0; index <= segments; index++) {
    const seam = addMesh(group, new THREE.TorusGeometry(radius + 0.018, 0.035, 8, 40), alloy,
      [-length / 2 + index * length / segments, 0, radius], "habitat_rib_band");
    seam.rotation.y = Math.PI / 2;
  }
  const doorWidth = Math.min(1.0, dimensions.ramp.width_m * 0.75), doorHeight = radius * 1.08;
  box(group, [0.18, doorWidth + 0.3, doorHeight + 0.25], dark,
    [length / 2 + 0.07, 0, rise + doorHeight / 2], "airlock_frame");
  box(group, [0.2, doorWidth, doorHeight], alloy,
    [length / 2 + 0.12, 0, rise + doorHeight / 2], "airlock_door");
  tube(group, [length / 2 + 0.24, doorWidth * 0.28, rise + doorHeight * 0.4],
    [length / 2 + 0.24, doorWidth * 0.28, rise + doorHeight * 0.62], 0.025, dark, "airlock_handle");
  for (const sign of [-1, 1]) for (let index = 0; index < 3; index++) {
    const x = -length * 0.3 + index * length * 0.26;
    box(group, [length * 0.12, 0.1, radius * 0.38], dark, [x, sign * radius * 0.98, radius * 1.12], "window_frame");
    box(group, [length * 0.1, 0.11, radius * 0.29], window, [x, sign * radius, radius * 1.12], "habitat_window");
  }
  const run = dimensions.ramp.run_m, width = dimensions.ramp.width_m;
  const ramp = box(group, [Math.hypot(run, rise), width, 0.09], metal(0x869b9a),
    [length / 2 + run / 2, 0, rise / 2], "airlock_ramp");
  ramp.rotation.y = Math.atan2(rise, run);
  for (let index = 1; index < 15; index++) {
    const fraction = index / 15;
    tube(group, [length / 2 + run * fraction, -width / 2, rise * (1 - fraction) + 0.06],
      [length / 2 + run * fraction, width / 2, rise * (1 - fraction) + 0.06], 0.012, dark, "ramp_tread");
  }
  const shielding = record.structural_check?.shielding_thickness_m || 0;
  if (shielding > 0) {
    const geometry = new THREE.SphereGeometry(1, 40, 20, 0, Math.PI * 2, 0, Math.PI / 2);
    geometry.rotateX(Math.PI / 2);
    const mound = addMesh(group, geometry, metal(0xa58c75, 1, 0), [-length * 0.12, 0, radius * 0.45], "regolith_shielding");
    mound.scale.set(length * 0.48, radius + shielding, radius * 1.65 + shielding);
    const positions = geometry.attributes.position;
    for (let index = 0; index < positions.count; index++) {
      const factor = 1 + 0.025 * Math.sin(index * 17.13);
      positions.setXYZ(index, positions.getX(index) * factor, positions.getY(index) * factor, positions.getZ(index) * factor);
    }
    geometry.computeVertexNormals();
  }
}

function landerSkin(record, group) {
  const { footprint_m: width, height_m: height } = record.dimensions;
  const bottom = height - 1.5, alloy = metal(0xb9c4c6), dark = metal(0x303b40), gold = foilMaterial();
  box(group, [width * 0.48, width * 0.48, 1.5], dark, [0, 0, bottom + 0.75], "lander_body");
  for (const side of [-1, 1]) {
    box(group, [width * 0.44, 0.035, 1.25], gold, [0, side * width * 0.243, bottom + 0.75], "gold_foil_panel");
    box(group, [0.035, width * 0.44, 1.25], gold, [side * width * 0.243, 0, bottom + 0.75], "gold_foil_panel");
  }
  for (let index = 0; index < record.dimensions.legs; index++) {
    const angle = Math.PI / 4 + index * Math.PI / 2, inner = width * 0.28, outer = width / 2 - 0.2;
    const foot = [outer * Math.cos(angle), outer * Math.sin(angle), 0.12];
    const top = [inner * Math.cos(angle), inner * Math.sin(angle), bottom];
    tube(group, top, foot, 0.09, alloy, "lander_leg");
    tube(group, [top[0], top[1], bottom + 0.5], [foot[0], foot[1], 0.5], 0.035, dark, "leg_brace");
    const pad = addMesh(group, new THREE.CylinderGeometry(0.28, 0.31, 0.12, 24), alloy,
      [foot[0], foot[1], 0.06], "lander_footpad"); pad.rotation.x = Math.PI / 2;
  }
  const bell = addMesh(group, new THREE.CylinderGeometry(width * 0.065, width * 0.15, height * 0.25, 32, 1, true),
    new THREE.MeshStandardMaterial({ color: 0x4c5558, metalness: 0.7, roughness: 0.58, side: THREE.DoubleSide }),
    [0, 0, bottom - height * 0.12], "engine_bell"); bell.rotation.x = Math.PI / 2;
  tube(group, [0, 0, height], [0, 0, height + 0.3], 0.025, alloy, "antenna_mast");
}

function bridgeNodes(panels, span, depth) {
  const nodes = {};
  for (let index = 0; index <= panels; index++) nodes[index + 1] = [index * span / panels, 0];
  for (let index = 0; index < panels; index++) nodes[panels + 2 + index] = [(index + 0.5) * span / panels, depth];
  const members = [];
  for (let index = 0; index < panels; index++) members.push([index + 1, index + 2]);
  for (let index = 0; index < panels - 1; index++) members.push([panels + 2 + index, panels + 3 + index]);
  for (let index = 0; index < panels; index++) members.push([index + 1, panels + 2 + index], [panels + 2 + index, index + 2]);
  return { nodes, members };
}

function bridgeSkin(record, group) {
  const { span_m: span, deck_width_m: width } = record.dimensions;
  const report = record.structural_check, panels = report?.panels || Math.max(2, Math.ceil(span / 1.5));
  const depth = report?.depth_m || span / 8, alloy = metal(0x9caeae), deck = metal(0x627775, 0.8);
  const geometry = bridgeNodes(panels, span, depth);
  const plates = Math.max(4, Math.ceil(span / 0.5));
  for (let index = 0; index < plates; index++) {
    box(group, [span / plates - 0.018, width, 0.19], deck,
      [-span / 2 + (index + 0.5) * span / plates, 0, 0.1], "bridge_deck_plate");
  }
  for (const side of [-1, 1]) {
    const across = side * (width / 2 - 0.06);
    for (const [start, end] of geometry.members) {
      const from = geometry.nodes[start], to = geometry.nodes[end];
      tube(group, [from[0] - span / 2, across, from[1] + 0.2],
        [to[0] - span / 2, across, to[1] + 0.2], 0.035, alloy, "bridge_truss_member");
    }
    tube(group, [-span / 2, across, 0.47], [span / 2, across, 0.47], 0.045, alloy, "bridge_rail");
    for (let index = 0; index <= panels; index++) {
      const x = index * span / panels - span / 2;
      tube(group, [x, across, 0.2], [x, across, 0.47], 0.025, alloy, "bridge_rail_post");
    }
  }
}

export function utilisationColor(percent) {
  if (!Number.isFinite(percent)) return new THREE.Color(0x8c999e);
  const low = new THREE.Color(0x269866), amber = new THREE.Color(0xe4aa37), red = new THREE.Color(0xd44439);
  return percent <= 75 ? low.lerp(amber, Math.max(0, percent) / 75)
    : amber.lerp(red, Math.min(1, (percent - 75) / 25));
}

function analysisOverlay(record, root) {
  const overlay = new THREE.Group(); overlay.name = "analysed_members"; overlay.visible = false; root.add(overlay);
  const report = record.structural_check;
  if (!report) return overlay;
  let members, nodes, mapPoint;
  const values = new Map();
  if (record.kind === "bridge") {
    const { nodes: points, members: connections } = bridgeNodes(report.panels, report.span_m, report.depth_m);
    nodes = points; members = connections.map((pair, index) => ({ tag: index + 1, nodes: pair }));
    mapPoint = ([x, height]) => [x - report.span_m / 2, -record.dimensions.deck_width_m / 2, height + 0.23];
    for (const mode of ["yield", "buckling"]) {
      const check = report.checks?.[mode];
      if (check?.member) values.set(check.member, Math.max(values.get(check.member) || 0, check.utilisation_pct));
    }
  } else {
    const frameReport = record.kind === "lander" ? report.cases.static : report;
    const frame = frameReport.frame; nodes = frame.nodes; members = frame.members;
    mapPoint = record.kind === "habitat" ? ([across, height]) => [0, across - record.dimensions.module_diameter_m / 2, height]
      : ([across, height]) => [0, across, height];
    for (const member of members) {
      if (record.kind === "lander" && member.group !== "leg") continue;
      let utilisation = 100 * member.stress_pa / frameReport.checks.stress.allowable_pa;
      const buckling = frameReport.checks.buckling;
      if (buckling) {
        const capacity = buckling.capacity_n * (buckling.effective_length_m / member.length_m) ** 2;
        utilisation = Math.max(utilisation, 100 * Math.max(0, -member.axial_force_n) / capacity);
      }
      values.set(member.tag, utilisation);
    }
  }
  for (const member of members) {
    const percent = values.get(member.tag);
    const mesh = tube(overlay, mapPoint(nodes[member.nodes[0]]), mapPoint(nodes[member.nodes[1]]), 0.065,
      metal(utilisationColor(percent), 0.5, 0.15), `analysed_member_${member.tag}`);
    mesh.userData.utilisationPct = Number.isFinite(percent) ? percent : null;
    mesh.userData.memberTag = member.tag; mesh.material.depthTest = false; mesh.renderOrder = 8;
  }
  return overlay;
}

export function setStructureDetails(root, enabled) {
  for (const group of root.children) {
    if (!group.userData.structure) continue;
    const { skin, overlay } = group.userData;
    overlay.visible = enabled;
    skin.traverse((object) => {
      const materials = object.material ? (Array.isArray(object.material) ? object.material : [object.material]) : [];
      for (const material of materials) {
        if (material.userData.skinOpacity === undefined) {
          material.userData.skinOpacity = material.opacity; material.userData.skinTransparent = material.transparent;
        }
        material.opacity = enabled ? material.userData.skinOpacity * 0.3 : material.userData.skinOpacity;
        material.transparent = enabled || material.userData.skinTransparent;
        material.depthWrite = !enabled; material.needsUpdate = true;
      }
    });
  }
}

async function insightSkin(record, sceneData, skin, root, terrainHeight, isCurrent, onCredit) {
  try {
    const gltf = await new GLTFLoader().loadAsync(NASA_INSIGHT);
    if (!isCurrent()) {
      gltf.scene.traverse((object) => { object.geometry?.dispose(); }); return false;
    }
    const model = gltf.scene;
    model.rotation.x = Math.PI / 2;
    model.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(model), size = bounds.getSize(new THREE.Vector3());
    const span = Math.max(size.x, size.y);
    if (!(span > 0)) throw new Error("InSight model has no measurable deployed span.");
    const scale = record.dimensions.footprint_m / span;
    model.scale.setScalar(scale); model.updateMatrixWorld(true);
    const lowVertices = [];
    model.traverse((object) => {
      if (!object.isMesh) return;
      object.castShadow = true; object.receiveShadow = true;
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      const cloned = materials.map((material) => {
        const copy = material.clone(); copy.envMap = reflectionEnvironment(); copy.envMapIntensity = 0.85;
        return copy;
      });
      object.material = Array.isArray(object.material) ? cloned : cloned[0];
      const positions = object.geometry.attributes.position, point = new THREE.Vector3();
      for (let index = 0; index < positions.count; index++) {
        point.fromBufferAttribute(positions, index).applyMatrix4(object.matrixWorld);
        const radius = Math.hypot(point.x, point.y);
        if (radius < record.dimensions.footprint_m * 0.28 && radius > record.dimensions.footprint_m * 0.1) {
          lowVertices.push(point.clone());
        }
      }
    });
    if (!lowVertices.length) throw new Error("InSight footpad geometry is unavailable.");
    const floor = lowVertices.reduce((lowest, point) => Math.min(lowest, point.z), Infinity);
    const clusters = [];
    for (const point of lowVertices) {
      if (point.z > floor + record.dimensions.footprint_m * 0.003) continue;
      let cluster = clusters.find((item) => Math.hypot(item.x / item.count - point.x,
        item.y / item.count - point.y) < record.dimensions.footprint_m * 0.06);
      if (!cluster) { cluster = { x: 0, y: 0, z: Infinity, count: 0 }; clusters.push(cluster); }
      cluster.x += point.x; cluster.y += point.y; cluster.z = Math.min(cluster.z, point.z); cluster.count++;
    }
    const supports = clusters.sort((first, second) => second.count - first.count).slice(0, 3)
      .map((item) => new THREE.Vector3(item.x / item.count, item.y / item.count, item.z - floor));
    if (supports.length !== 3) throw new Error("InSight requires three identifiable footpads.");
    model.position.z = -floor;
    const yaw = record.orientation.yaw_deg * Math.PI / 180;
    const at = record.position;
    const groundPoint = (point) => {
      const x = at.x + Math.cos(yaw) * point.x - Math.sin(yaw) * point.y;
      const y = at.y + Math.sin(yaw) * point.x + Math.cos(yaw) * point.y;
      return new THREE.Vector3(x, y, terrainHeight(sceneData, x, y));
    };
    const normal = new THREE.Plane().setFromCoplanarPoints(...supports.map(groundPoint)).normal;
    if (normal.z < 0) normal.negate();
    root.quaternion.premultiply(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), normal));
    const aligned = supports.map((point) => point.clone().applyQuaternion(root.quaternion));
    const elevation = Math.max(...aligned.map((point) => {
      return terrainHeight(sceneData, at.x + point.x, at.y + point.y) - point.z;
    }));
    root.position.z = elevation;
    skin.traverse((object) => {
      object.geometry?.dispose();
      const materials = object.material ? (Array.isArray(object.material) ? object.material : [object.material]) : [];
      for (const material of materials) { material.map?.dispose(); material.dispose(); }
    });
    skin.clear(); skin.add(model); skin.name = "nasa_insight_skin";
    skin.userData.deployedSpanM = record.dimensions.footprint_m;
    skin.userData.supportPoints = aligned.map((point) => [at.x + point.x, at.y + point.y, point.z + elevation]);
    root.userData.credit = "NASA"; onCredit("NASA");
    return true;
  } catch (error) {
    if (isCurrent()) root.userData.assetFallback = true;
    return false;
  }
}

export function buildStructureVisuals(sceneData, { terrainHeight, isCurrent = () => true, onCredit = () => {} } = {}) {
  const root = new THREE.Group(); root.name = "structure_visuals";
  const ready = [];
  (sceneData.structures || []).forEach((record, index) => {
    const group = new THREE.Group(); group.name = `structure_visual_${index}_${record.kind}`;
    group.position.set(record.position.x, record.position.y, record.position.z);
    group.rotation.z = (record.orientation.yaw_deg || 0) * Math.PI / 180;
    const skin = new THREE.Group(); skin.name = "procedural_skin"; group.add(skin);
    if (record.kind === "habitat") habitatSkin(record, skin);
    else if (record.kind === "lander") landerSkin(record, skin);
    else if (record.kind === "bridge") {
      bridgeSkin(record, skin); group.rotation.y = (record.orientation.pitch_deg || 0) * Math.PI / 180;
    } else return;
    const overlay = analysisOverlay(record, group);
    group.userData = { structure: record.kind, skin, overlay }; root.add(group);
    if (record.kind === "lander" && sceneData.world?.body === "mars") {
      ready.push(insightSkin(record, sceneData, skin, group, terrainHeight, isCurrent, (credit) => onCredit(index, credit)));
    }
  });
  return { root, ready: Promise.all(ready) };
}