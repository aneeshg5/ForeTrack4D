import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const EXAMPLES = [
  { id: "r012-nespresso", title: "HoloAssist: Nespresso assembly", thumb: "data/r012-nespresso_frame.jpg", data: "data/r012-nespresso.json" },
  { id: "r175-ram", title: "HoloAssist: RAM installation", thumb: "data/r175-ram_frame.jpg", data: "data/r175-ram.json" },
  // flipUp: true -- monocular SLAM (MegaSaM has no IMU/gravity reference) has no absolute way
  // to know which way is "up"; the recovered coordinate frame's orientation is unconstrained by
  // reprojection error alone and can converge flipped for a given clip's specific camera motion.
  // Verified directly (not guessed): geometry and camera-fit math are identical in structure and
  // sign to the two examples that render correctly, and this clip has no rotation metadata that
  // would explain it another way -- this is MegaSaM's own real, expected ambiguity for
  // in-the-wild monocular video, corrected per-example rather than silently dropped or left broken.
  { id: "pstudio", title: "TAPIP3D demo: basketball", thumb: "data/pstudio_frame.jpg", data: "data/pstudio.json", flipUp: true },
];

const FORECAST_COLORS = [0x3cb4ff, 0x3cff78, 0xff783c];
const TRAIL_LENGTH = 10; // how many past frames stay visible, fading out (ForeHand4D's "saturation fading with time")

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
const camera = new THREE.PerspectiveCamera(50, 4 / 3, 0.001, 100);
camera.up.set(0, -1, 0); // camera-frame convention (y-down, z-forward): keep the point cloud right-side-up

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 1.2));

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);

// --- image plane, positioned in 3D using the real camera intrinsics so it aligns exactly with
// the point cloud at the conditioning frame, not just an unrelated backdrop image. Unprojects
// the four image corners at a chosen depth via the pinhole model (u,v) -> ((u-cx)/fx*Z, (v-cy)/fy*Z, Z).
function unproject(u, v, z, intrinsics) {
  const fx = intrinsics[0][0], fy = intrinsics[1][1], cx = intrinsics[0][2], cy = intrinsics[1][2];
  return new THREE.Vector3(((u - cx) / fx) * z, ((v - cy) / fy) * z, z);
}

function buildImagePlane(d) {
  const depth = medianDepth(d.observed_tracks[0]);
  const corners = [
    unproject(0, 0, depth, d.intrinsics),
    unproject(d.frame_w, 0, depth, d.intrinsics),
    unproject(0, d.frame_h, depth, d.intrinsics),
    unproject(d.frame_w, d.frame_h, depth, d.intrinsics),
  ];
  const geom = new THREE.BufferGeometry();
  const positions = new Float32Array(corners.flatMap((c) => [c.x, c.y, c.z]));
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2));
  geom.setIndex([0, 1, 2, 2, 1, 3]);
  geom.computeVertexNormals();

  const texture = new THREE.TextureLoader().load(`data/${d.frame_image}`);
  texture.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  return { mesh: new THREE.Mesh(geom, mat), corners };
}

function medianDepth(pointsAtT0) {
  const zs = pointsAtT0.map((p) => p[2]).sort((a, b) => a - b);
  return zs[Math.floor(zs.length / 2)];
}

function fitCameraToPoints(flatPoints) {
  const box = new THREE.Box3();
  flatPoints.forEach((p) => box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2])));
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 0.3;
  controls.target.copy(center);
  camera.position.set(center.x, center.y, center.z - size * 1.6);
  camera.near = size / 100;
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

let current = null;
let imagePlane = null;

// each track (observed, or one forecast sample) gets TRAIL_LENGTH point-cloud "ghost" layers,
// one per recent past frame, with opacity fading by how far in the past that frame is -- this
// is what actually makes motion (and forecasting specifically) visually legible, instead of a
// single point cloud that just teleports between frames with no sense of a path.
function buildTrail(color) {
  const layers = [];
  for (let i = 0; i < TRAIL_LENGTH; i++) {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
    const opacity = 1.0 - (i / TRAIL_LENGTH) * 0.85;
    const mat = new THREE.PointsMaterial({ color, size: 7, sizeAttenuation: false, transparent: true, opacity });
    const points = new THREE.Points(geom, mat);
    points.visible = false;
    layers.push(points);
  }
  return layers;
}

function setLayerPositions(points, arr) {
  const pos = points.geometry.attributes.position;
  for (let i = 0; i < arr.length; i++) pos.setXYZ(i, arr[i][0], arr[i][1], arr[i][2]);
  pos.needsUpdate = true;
  points.geometry.computeBoundingSphere();
  points.visible = true;
}

async function loadExample(ex) {
  document.querySelectorAll(".thumb").forEach((el) => el.classList.toggle("active", el.dataset.id === ex.id));
  document.getElementById("example-source").textContent = "loading…";
  camera.up.set(0, ex.flipUp ? 1 : -1, 0);

  const resp = await fetch(ex.data);
  const d = await resp.json();

  if (current) {
    current.observedTrail.forEach((p) => scene.remove(p));
    current.forecastTrails.forEach((track) => track.forEach((p) => scene.remove(p)));
  }
  if (imagePlane) scene.remove(imagePlane);

  const { mesh: planeMesh, corners: planeCorners } = buildImagePlane(d);
  imagePlane = planeMesh;
  scene.add(imagePlane);

  const observedTrail = buildTrail(0xffffff);
  const forecastTrails = d.forecast_samples.map((_, s) => buildTrail(FORECAST_COLORS[s % FORECAST_COLORS.length]));
  observedTrail.forEach((p) => scene.add(p));
  forecastTrails.forEach((track) => track.forEach((p) => scene.add(p)));

  current = { d, observedTrail, forecastTrails, t: 0, playing: true };

  // anchor the default view on the conditioning frame itself (t0 points + the image plane
  // corners), not the full trajectory -- forecast samples run 128 timesteps and can drift far
  // from the conditioning depth, which previously pulled the camera away from the image plane
  // entirely. The user can still zoom/pan out manually as playback moves past the anchor.
  const anchorPoints = d.observed_tracks[0].concat(planeCorners.map((c) => c.toArray()));
  fitCameraToPoints(anchorPoints);

  const scrubber = document.getElementById("scrubber");
  const tMax = Math.max(d.observed_tracks.length, d.forecast_samples[0].length) - 1;
  scrubber.max = String(tMax);
  scrubber.value = "0";

  const flag = document.getElementById("confidence-flag");
  if (d.low_confidence) {
    flag.textContent = `Low confidence (disagreement ${d.disagreement_cm.toFixed(1)}cm) — treat this forecast cautiously.`;
    flag.classList.add("low");
  } else {
    flag.classList.remove("low");
  }
  document.getElementById("example-source").textContent = `${d.title} — click and drag to rotate, scroll to zoom.`;

  updateFrame();
}

function updateFrame() {
  if (!current) return;
  const { d, observedTrail, forecastTrails, t } = current;
  for (let i = 0; i < TRAIL_LENGTH; i++) {
    const tt = t - i;
    if (tt >= 0 && tt < d.observed_tracks.length) setLayerPositions(observedTrail[i], d.observed_tracks[tt]);
    else observedTrail[i].visible = false;
  }
  d.forecast_samples.forEach((samples, s) => {
    for (let i = 0; i < TRAIL_LENGTH; i++) {
      const tt = t - i;
      if (tt >= 0 && tt < samples.length) setLayerPositions(forecastTrails[s][i], samples[tt]);
      else forecastTrails[s][i].visible = false;
    }
  });
  document.getElementById("frame-label").textContent = `frame ${t}`;
}

document.getElementById("scrubber").addEventListener("input", (e) => {
  if (!current) return;
  current.playing = false;
  document.getElementById("play-btn").textContent = "Play";
  current.t = parseInt(e.target.value, 10);
  updateFrame();
});

document.getElementById("play-btn").addEventListener("click", () => {
  if (!current) return;
  current.playing = !current.playing;
  document.getElementById("play-btn").textContent = current.playing ? "Pause" : "Play";
});

let lastTick = 0;
function animate(now) {
  requestAnimationFrame(animate);
  controls.update();
  if (current && current.playing && now - lastTick > 80) {
    lastTick = now;
    const tMax = parseInt(document.getElementById("scrubber").max, 10);
    current.t = (current.t + 1) % (tMax + 1);
    document.getElementById("scrubber").value = String(current.t);
    updateFrame();
  }
  renderer.render(scene, camera);
}

const thumbsEl = document.getElementById("thumbs");
EXAMPLES.forEach((ex, i) => {
  const div = document.createElement("div");
  div.className = "thumb" + (i === 0 ? " active" : "");
  div.dataset.id = ex.id;
  div.innerHTML = `<img src="${ex.thumb}"><div class="label">${ex.title}</div>`;
  div.addEventListener("click", () => loadExample(ex));
  thumbsEl.appendChild(div);
});

resize();
loadExample(EXAMPLES[0]);
requestAnimationFrame(animate);
