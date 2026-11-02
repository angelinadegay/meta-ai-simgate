import * as THREE from "three";

export function frameBase(camera, controls, root, sceneData, terrainHeight, index = null) {
  root.updateWorldMatrix(true, true);
  const bounds = new THREE.Box3();
  const groups = index === null ? root.children : [root.children[index]];
  for (const group of groups) if (group) bounds.expandByObject(group.userData.skin);
  if (index === null) {
    const missions = Object.values(sceneData.robots || {}).map(robot => robot.mission);
    if (!missions.length) missions.push(sceneData.mission);
    for (const mission of missions) {
      if (!mission) continue;
      for (const kind of ["start", "goal"]) {
        const point = mission[kind]; if (!point) continue;
        const height = kind === "goal" && Number.isFinite(mission.goal_z) ? mission.goal_z : terrainHeight(sceneData, ...point);
        const bottom = new THREE.Vector3(point[0], point[1], height - 0.3);
        const top = new THREE.Vector3(point[0], point[1], height + 1.8);
        if (root.parent) { root.parent.localToWorld(bottom); root.parent.localToWorld(top); }
        bounds.expandByPoint(bottom); bounds.expandByPoint(top);
      }
    }
  }
  if (bounds.isEmpty()) return null;
  const centre = bounds.getCenter(new THREE.Vector3()), size = bounds.getSize(new THREE.Vector3());
  const vertical = camera.fov * Math.PI / 180, horizontal = 2 * Math.atan(Math.tan(vertical / 2) * camera.aspect);
  const distance = Math.max(2, size.length() / 2 / Math.sin(Math.min(vertical, horizontal) / 2) * 1.16);
  controls.target.copy(centre);
  camera.position.copy(centre).add(new THREE.Vector3(0.8, 0.65, 1).normalize().multiplyScalar(distance));
  camera.lookAt(centre); camera.updateMatrixWorld(true);
  return bounds;
}

export function easeIntoFollow(camera, controls, robotPosition, dt) {
  const blend = 1 - Math.exp(-Math.max(0, dt) / 0.55);
  controls.target.lerp(robotPosition, blend);
  const desired = robotPosition.clone().add(new THREE.Vector3(-3.2, 2.2, 2.6));
  camera.position.lerp(desired, blend);
}

export function advanceFollowTransition(camera, controls, robotPosition, dt, remaining) {
  const step = Math.min(Math.max(0, dt), 0.1, remaining);
  easeIntoFollow(camera, controls, robotPosition, step);
  return Math.max(0, remaining - step);
}