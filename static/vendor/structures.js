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

function habitatShellMaterial() {
  const canvas = document.createElement("canvas"); canvas.width = canvas.height = 256;
  const context = canvas.getContext("2d"); context.fillStyle = "#f1f3ed"; context.fillRect(0, 0, 256, 256);
  for (let x = 0; x < 256; x++) {
    const lowerHalf = Math.max(0, -Math.cos(x / 256 * Math.PI * 2));
    context.globalAlpha = lowerHalf * (0.28 + 0.07 * Math.sin(x * 0.37));
    context.fillStyle = "#aa957b"; context.fillRect(x, 0, 1, 256);
  }
  for (let index = 0; index < 120; index++) {
    const x = (index * 73) % 256, y = (index * 131) % 256;
    context.globalAlpha = 0.07; context.strokeStyle = "#64716c"; context.lineWidth = 0.5;
    context.beginPath(); context.moveTo(x, y); context.lineTo(x + 2 + index % 7, y + index % 3); context.stroke();
  }
  context.globalAlpha = 1;
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  const material = metal(0xd8dedb, 0.72, 0.15); material.map = texture;
  material.bumpMap = texture; material.bumpScale = 0.003;
  return material;
}

function habitatSkin(record, group) {
  const dimensions = record.dimensions, length = dimensions.module_length_m;
  const radius = dimensions.module_diameter_m / 2, rise = dimensions.airlock_rise_m;
  const shell = habitatShellMaterial(), dark = metal(0x33474a), alloy = metal(0xa6b8bc);
  const window = metal(0x234f60, 0.12, 0.65), segments = Math.max(3, Math.round(length / 1.2));
  for (let index = 0; index < segments; index++) {
    const segment = addMesh(group, new THREE.CylinderGeometry(radius, radius, length / segments, 40),
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
  const top = dimensions.ramp.top_z !== undefined ? dimensions.ramp.top_z - record.position.z : rise;
  const toe = dimensions.ramp.toe_z !== undefined ? dimensions.ramp.toe_z - record.position.z : 0;
  const rampPitch = Math.atan2(top - toe, run);
  const ramp = box(group, [Math.hypot(run, top - toe), width, 0.09], metal(0x869b9a),
    [length / 2 + run / 2, 0, (top + toe) / 2 - 0.045 * Math.cos(rampPitch)], "airlock_ramp");
  ramp.rotation.y = rampPitch;
  for (let index = 1; index < 15; index++) {
    const fraction = index / 15;
    const height = top * (1 - fraction) + toe * fraction;
    tube(group, [length / 2 + run * fraction, -width / 2, height + 0.06],
      [length / 2 + run * fraction, width / 2, height + 0.06], 0.012, dark, "ramp_tread");
  }
  for (const sign of [-1, 1]) {
    const across = sign * (width / 2 - 0.06);
    tube(group, [length / 2, across, top + 0.8], [length / 2 + run, across, toe + 0.8],
      0.035, alloy, "ramp_side_rail");
    for (let index = 0; index <= 4; index++) {
      const fraction = index / 4, x = length / 2 + run * fraction, z = top * (1 - fraction) + toe * fraction;
      tube(group, [x, across, z + 0.04], [x, across, z + 0.8], 0.025, alloy, "ramp_rail_post");
    }
  }
  const shielding = record.structural_check?.shielding_thickness_m || 0;
  if (shielding > 0) {
    const positions = [], indices = [], colours = [], alongSteps = 24, acrossSteps = 24;
    for (let along = 0; along <= alongSteps; along++) for (let across = 0; across <= acrossSteps; across++) {
      const fraction = along / alongSteps, angle = Math.PI * (0.18 + 0.64 * across / acrossSteps);
      const taper = Math.sin(Math.PI * fraction) ** 0.35;
      const grain = 0.025 * Math.sin(along * 17.13 + across * 5.23);
      const outerRadius = radius + shielding * taper + grain;
      positions.push(-length / 2 + 0.2 + fraction * (length - 1), outerRadius * Math.cos(angle),
        radius + outerRadius * Math.sin(angle));
      const tint = 0.82 + 0.12 * Math.sin(along * 3.2 + across * 9.1);
      colours.push(tint, tint, tint);
      if (along < alongSteps && across < acrossSteps) {
        const corner = along * (acrossSteps + 1) + across;
        indices.push(corner, corner + 1, corner + acrossSteps + 1,
          corner + 1, corner + acrossSteps + 2, corner + acrossSteps + 1);
      }
    }
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3)); geometry.setIndex(indices); geometry.computeVertexNormals();
    const material = metal(0xa58c75, 1, 0); material.vertexColors = true; material.side = THREE.DoubleSide;
    addMesh(group, geometry, material, [0, 0, 0], "regolith_shielding");
  }
}

function landerSkin(record, group) {
  const { footprint_m: width, height_m: height } = record.dimensions;
  const deck = record.dimensions.deck_span_m || width * 0.48;
  const bodyHeight = record.dimensions.body_height_m || 1.5;
  const bottom = record.dimensions.deck_height_m ? record.dimensions.deck_height_m - bodyHeight / 2 : height - bodyHeight;
  const alloy = metal(0xb9c4c6), dark = metal(0x303b40), gold = foilMaterial();
  box(group, [deck, deck, bodyHeight], dark, [0, 0, bottom + bodyHeight / 2], "lander_body");
  for (const side of [-1, 1]) {
    box(group, [deck * 0.94, 0.035, bodyHeight * 0.83], gold, [0, side * deck * 0.505, bottom + bodyHeight / 2], "gold_foil_panel");
    box(group, [0.035, deck * 0.94, bodyHeight * 0.83], gold, [side * deck * 0.505, 0, bottom + bodyHeight / 2], "gold_foil_panel");
  }
  for (let index = 0; index < record.dimensions.legs; index++) {
    const angle = Math.PI / 4 + index * Math.PI * 2 / record.dimensions.legs;
    const inner = deck * 0.45, outer = (record.dimensions.leg_footprint_m || width) / 2 - 0.2;
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
  if (record.dimensions.visual_model === "nasa_insight") for (const sign of [-1, 1]) {
    const radius = (width - deck) / 4;
    const panel = addMesh(group, new THREE.CylinderGeometry(radius, radius, 0.05, 48), metal(0x294351, 0.6, 0.3),
      [sign * (deck / 2 + radius), 0, record.dimensions.deck_height_m], "lander_solar_panel");
    panel.rotation.x = Math.PI / 2;
  }
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
  if (record.mid_support) {
    box(group, [0.3, 0.3, 0.44], metal(0x89928c, 0.85), [0, 0, -0.38], "bridge_mid_support");
  }
  const plates = Math.max(4, Math.ceil(span / 0.5));
  for (let index = 0; index < plates; index++) {
    const start = -span / 2 + index * span / plates + (index ? 0.009 : 0);
    const end = -span / 2 + (index + 1) * span / plates - (index < plates - 1 ? 0.009 : 0);
    box(group, [end - start, width, 0.19], deck, [(start + end) / 2, 0, -0.07], "bridge_deck_plate");
  }
  for (const side of [-1, 1]) {
    const across = side * (width / 2 - 0.06);
    for (const [start, end] of geometry.members) {
      const from = geometry.nodes[start], to = geometry.nodes[end];
      tube(group, [from[0] - span / 2, across, from[1] + 0.03],
        [to[0] - span / 2, across, to[1] + 0.03], 0.035, alloy, "bridge_truss_member");
    }
    tube(group, [-span / 2, across, 0.47], [span / 2, across, 0.47], 0.045, alloy, "bridge_rail");
    for (let index = 0; index <= panels; index++) {
      const x = index * span / panels - span / 2;
      tube(group, [x, across, 0.03], [x, across, 0.47], 0.025, alloy, "bridge_rail_post");
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
    mapPoint = ([x, height]) => [x - report.span_m / 2, -record.dimensions.deck_width_m / 2, height + 0.06];
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
    if (record.kind === "lander" && sceneData.world?.body === "mars" && record.dimensions.visual_model === "nasa_insight") {
      ready.push(insightSkin(record, sceneData, skin, group, terrainHeight, isCurrent, (credit) => onCredit(index, credit)));
    }
  });
  return { root, ready: Promise.all(ready) };
}

export function measureStructureVisuals(root) {
  root.updateMatrixWorld(true);
  return root.children.map((group) => {
    const skin = group.userData.skin;
    const bounds = new THREE.Box3().setFromObject(skin);
    const moduleBounds = new THREE.Box3();
    skin.traverse((object) => {
      if (object.name === "habitat_segment" || object.name === "lander_body" || object.name === "bridge_deck_plate") {
        moduleBounds.expandByObject(object);
      }
    });
    const size = bounds.getSize(new THREE.Vector3()).toArray();
    return { kind: group.userData.structure, skin: skin.name, world_size_m: size,
      body_size_m: moduleBounds.isEmpty() ? null : moduleBounds.getSize(new THREE.Vector3()).toArray(),
      deployed_span_m: skin.userData.deployedSpanM || null };
  });
}