import * as THREE from "three";

export function terrainSample(heightfield, x, y) {
  const { nrow, ncol, data, size } = heightfield;
  const col = THREE.MathUtils.clamp((x / size[0] + 1) * (ncol - 1) / 2, 0, ncol - 1);
  const row = THREE.MathUtils.clamp((y / size[1] + 1) * (nrow - 1) / 2, 0, nrow - 1);
  const left = Math.min(Math.floor(col), ncol - 2), bottom = Math.min(Math.floor(row), nrow - 2);
  const along = col - left, across = row - bottom;
  const lower = THREE.MathUtils.lerp(data[bottom * ncol + left], data[bottom * ncol + left + 1], along);
  const upper = THREE.MathUtils.lerp(data[(bottom + 1) * ncol + left], data[(bottom + 1) * ncol + left + 1], along);
  return THREE.MathUtils.lerp(lower, upper, across) * size[2];
}

export function continuedHeight(heightfield, x, y) {
  const [halfX, halfY] = heightfield.size;
  const edgeX = THREE.MathUtils.clamp(x, -halfX, halfX), edgeY = THREE.MathUtils.clamp(y, -halfY, halfY);
  const edge = terrainSample(heightfield, edgeX, edgeY);
  const cellX = 2 * halfX / (heightfield.ncol - 1), cellY = 2 * halfY / (heightfield.nrow - 1);
  const inwardX = edgeX - Math.sign(edgeX) * cellX, inwardY = edgeY - Math.sign(edgeY) * cellY;
  const slopeX = Math.abs(x) > halfX ? (edge - terrainSample(heightfield, inwardX, edgeY)) / (edgeX - inwardX) : 0;
  const slopeY = Math.abs(y) > halfY ? (edge - terrainSample(heightfield, edgeX, inwardY)) / (edgeY - inwardY) : 0;
  const distance = Math.hypot(x - edgeX, y - edgeY);
  const decay = Math.exp(-distance / Math.max(12, Math.max(halfX, halfY)));
  return edge + (slopeX * (x - edgeX) + slopeY * (y - edgeY)) * decay;
}

export function buildTerrainHorizon(heightfield, sourceMesh) {
  const [halfX, halfY] = heightfield.size;
  const steps = Math.max(heightfield.nrow, heightfield.ncol) - 1;
  const offsets = [0, 0.1, 0.5, 2, 6, 16, 40, 100, 300, 900, 2400];
  const positions = [], colors = [], uvs = [], indices = [];
  const sourceColors = sourceMesh.geometry.attributes.color;
  const outer = Math.max(halfX, halfY) + offsets.at(-1);
  for (const offset of offsets) {
    for (let side = 0; side < 4; side++) for (let index = 0; index < steps; index++) {
      const fraction = index / steps;
      const perimeter = [[-1 + 2 * fraction, -1], [1, -1 + 2 * fraction],
        [1 - 2 * fraction, 1], [-1, 1 - 2 * fraction]][side];
      const x = perimeter[0] * (halfX + offset), y = perimeter[1] * (halfY + offset);
      positions.push(x, y, continuedHeight(heightfield, x, y));
      const col = Math.round(THREE.MathUtils.clamp((x / halfX + 1) / 2, 0, 1) * (heightfield.ncol - 1));
      const row = Math.round(THREE.MathUtils.clamp((y / halfY + 1) / 2, 0, 1) * (heightfield.nrow - 1));
      const pixel = row * heightfield.ncol + col;
      const fade = 1 - THREE.MathUtils.smoothstep(Math.max(Math.abs(x), Math.abs(y)), outer * 0.65, outer);
      colors.push(sourceColors?.getX(pixel) ?? 1, sourceColors?.getY(pixel) ?? 1, sourceColors?.getZ(pixel) ?? 1, fade);
      uvs.push((x / halfX + 1) / 2, (y / halfY + 1) / 2);
    }
  }
  const count = steps * 4;
  for (let ring = 0; ring < offsets.length - 1; ring++) for (let index = 0; index < count; index++) {
    const next = (index + 1) % count;
    const inner = ring * count, outerRing = (ring + 1) * count;
    indices.push(inner + index, outerRing + index, inner + next,
      inner + next, outerRing + index, outerRing + next);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 4));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices); geometry.computeVertexNormals();
  const material = sourceMesh.material.clone(); material.vertexColors = true; material.transparent = true;
  material.depthWrite = false; material.side = THREE.DoubleSide;
  const horizon = new THREE.Mesh(geometry, material); horizon.name = "visual_terrain_horizon";
  horizon.receiveShadow = true; horizon.userData.visualOnly = true;
  return horizon;
}