import * as THREE from "three";

export function createMissionFlag(kind, x, y, z) {
  const group = new THREE.Group(); group.name = `mission_marker_${kind}`;
  group.position.set(x, y, z); group.userData.kind = kind;
  const color = kind === "start" ? 0x279669 : 0xe4a92d;
  const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1.2, 10),
    new THREE.MeshStandardMaterial({ color: 0xd5dcd8, metalness: 0.35, roughness: 0.5 }));
  pole.rotation.x = Math.PI / 2; pole.position.z = 0.6; group.add(pole);
  const ring = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.21, 24),
    new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
  ring.position.z = 0.018; group.add(ring);
  const canvas = document.createElement("canvas"); canvas.width = 128; canvas.height = 80;
  const context = canvas.getContext("2d"); context.fillStyle = `#${new THREE.Color(color).getHexString()}`;
  context.fillRect(0, 0, 128, 80); context.fillStyle = "#ffffff";
  context.font = "bold 48px sans-serif"; context.textAlign = "center"; context.textBaseline = "middle";
  context.fillText(kind === "start" ? "S" : "G", 64, 40);
  const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace;
  const cloth = new THREE.Mesh(new THREE.PlaneGeometry(0.48, 0.3),
    new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }));
  cloth.rotation.x = Math.PI / 2; cloth.position.set(0.24, 0, 1.02); group.add(cloth);
  return group;
}

export function disposeMissionMarkers(group) {
  group.traverse(object => {
    object.geometry?.dispose();
    if (object.material) { object.material.map?.dispose(); object.material.dispose(); }
  });
  group.removeFromParent();
}