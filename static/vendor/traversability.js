import * as THREE from "three";

const COLORS = { green: 0x40d8a0, amber: 0xf2bc55, red: 0xf4777d, grey: 0x8e98ab };

export function mapOverlay(report, groundHeight, replay) {
  const root = new THREE.Group(), markers = [];
  root.name = "traversability-overlay";
  const cell = report.cell_size_m, half = cell / 2;
  const span = Math.max(report.x_m.length, report.y_m.length) * cell;
  const ribbonRadius = Math.max(0.055, span * 0.0015), markerSize = Math.max(0.65, span * 0.014);
  const material = new THREE.MeshBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.42, side: THREE.DoubleSide, fog: false,
    depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2,
  });
  const positions = [], colors = [], outlines = [];
  const vertex = (x, y, color) => {
    positions.push(x, y, groundHeight(x, y) + 0.035);
    colors.push(color.r, color.g, color.b);
  };
  report.y_m.forEach((y, row) => report.x_m.forEach((x, col) => {
    const color = new THREE.Color(COLORS[report.classes[row][col]]);
    const corners = [[x - half, y - half], [x + half, y - half],
      [x + half, y + half], [x - half, y + half]];
    for (const index of [0, 1, 2, 0, 2, 3]) vertex(...corners[index], color);
    if (report.measured?.[row]?.[col]) {
      for (let i = 0; i < 4; i++) {
        for (const p of [corners[i], corners[(i + 1) % 4]]) outlines.push(...p, groundHeight(...p) + 0.055);
      }
    }
  }));
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  root.add(new THREE.Mesh(geometry, material));
  if (outlines.length) {
    const lines = new THREE.BufferGeometry();
    lines.setAttribute("position", new THREE.Float32BufferAttribute(outlines, 3));
    root.add(new THREE.LineSegments(lines, new THREE.LineBasicMaterial({ color: 0xffffff, fog: false })));
  }
  function line(points, dashed = false) {
    if (points.length < 2) return;
    const sampled = [];
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i], distance = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const count = Math.max(1, Math.ceil(distance / 0.15));
      for (let j = 0; j <= count; j++) {
        const x = a[0] + (b[0] - a[0]) * j / count, y = a[1] + (b[1] - a[1]) * j / count;
        sampled.push(new THREE.Vector3(x, y, groundHeight(x, y) + 0.12));
      }
    }
    if (dashed) {
      const dashedLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(sampled),
        new THREE.LineDashedMaterial({ color: 0xe9e5ff, dashSize: 0.24, gapSize: 0.16, depthTest: false, fog: false }));
      dashedLine.computeLineDistances(); root.add(dashedLine);
    } else {
      const path = new THREE.CurvePath();
      for (let i = 1; i < sampled.length; i++) path.add(new THREE.LineCurve3(sampled[i - 1], sampled[i]));
      const tube = new THREE.TubeGeometry(path, sampled.length * 2, ribbonRadius, 5, false);
      root.add(new THREE.Mesh(tube, new THREE.MeshBasicMaterial({ color: 0x8effd7, depthTest: false, fog: false })));
      root.add(new THREE.Mesh(new THREE.TubeGeometry(path, sampled.length * 2, ribbonRadius * 2.4, 5, false),
        new THREE.MeshBasicMaterial({ color: 0x5fffc9, transparent: true, opacity: 0.18, depthWrite: false, fog: false })));
    }
  }
  for (const leg of report.comparison.legs || [report.comparison]) {
    const route = leg.route;
    if (route) {
      line([route.start_xy, route.goal_xy], true);
      line(route.path_xy || route.partial_path_xy || []);
    }
  }
  for (const spot of report.spot_checks) {
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 96;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = spot.passed === null ? "#8e98ab" : spot.passed ? "#127955" : "#b63f53";
    ctx.beginPath(); ctx.arc(48, 48, 40, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "white"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.font = "bold 54px sans-serif";
    ctx.fillText(spot.passed === null ? "?" : spot.passed ? "✓" : "✕", 48, 50);
    const texture = new THREE.CanvasTexture(canvas);
    const marker = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, fog: false }));
    const x = report.x_m[spot.col], y = report.y_m[spot.row];
    marker.position.set(x, y, groundHeight(x, y) + 0.55); marker.scale.set(markerSize, markerSize, markerSize);
    marker.userData.runId = spot.id; root.add(marker); markers.push(marker);
  }
  return { root, markers, replay };
}
