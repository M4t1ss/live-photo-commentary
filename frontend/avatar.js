import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const canvas = document.getElementById('avatar-canvas');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 500);

scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const keyLight = new THREE.DirectionalLight(0xfff4e0, 2.5);
keyLight.position.set(1, 2, 3);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xd0e8ff, 0.8);
fillLight.position.set(-2, 0.5, -1);
scene.add(fillLight);

const clock = new THREE.Clock();
let mixer = null;

// Lip sync state — written by setLipSyncData (called from app.js),
// read every frame by applyLipSync.
let lipSyncTimeline = [];
let lipSyncAudio    = null;
let lipSyncDisabled = false;  // set by showPhoneme for debugging

// Built from the loaded model; used by applyLipSync.
const morphMeshes = new Map();  // Map<Mesh, morphTargetDictionary>
let jawBone       = null;
let jawAxis       = 'x';
let jawRestAngle  = 0;      // initial angle of jaw bone in its axis
let minJawAngle   = 0;      // relative angle from rest (should be ≤ 0)
let maxJawAngle   = 0.15;   // relative angle from rest (should be ≥ 0)
let visemeMap     = {};     // phoneme → {morph: value, ...}

// Scroll-wheel zoom — 0 = full body, 1 = head shot.
let zoomT = 0;
let _fullBodyCam = null;  // { camY, camZ, lookY } — set after model loads
let _headShotCam = null;

function _applyZoom() {
  if (!_fullBodyCam) return;
  const s = zoomT * zoomT * (3 - 2 * zoomT);  // smoothstep
  const camY  = _fullBodyCam.camY  + s * (_headShotCam.camY  - _fullBodyCam.camY);
  const camZ  = _fullBodyCam.camZ  + s * (_headShotCam.camZ  - _fullBodyCam.camZ);
  const lookY = _fullBodyCam.lookY + s * (_headShotCam.lookY - _fullBodyCam.lookY);
  camera.position.set(0, camY, camZ);
  camera.lookAt(0, lookY, 0);
}

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoomT = Math.max(0, Math.min(1, zoomT - e.deltaY * 0.001));
  _applyZoom();
}, { passive: false });

// --- Envelope curve ---
// x: normalised position within one half of the bell (0 = edge, 1 = peak).
// Replace body to try a different curve shape (smoothstep, cosine, etc.).
function envelopeCurve(x) {
  return Math.exp(-2 * (x - 1) ** 2);  // Gaussian half
}

function sampleEnvelope({ absStart, absPeak, absEnd, value }, t) {
  if (t < absStart || t > absEnd) return 0;
  const span = t <= absPeak ? (absPeak - absStart) : (absEnd - absPeak);
  if (span < 1e-6) return value;
  const x = 1 - Math.abs(t - absPeak) / span;
  return value * envelopeCurve(x);
}

// Apply the current lip sync timeline at audio time t.
// Always iterates all morph meshes so influences are zeroed when inactive.
function applyLipSync(t) {
  const influences = {};
  let jawValue = 0;

  for (const ev of lipSyncTimeline) {
    const v = sampleEnvelope(ev, t);
    if (v <= 0) continue;
    if (ev.key === 'jaw') jawValue = Math.max(jawValue, v);
    else influences[ev.key] = Math.max(influences[ev.key] ?? 0, v);
  }

  for (const [mesh, dict] of morphMeshes) {
    for (const [name, idx] of Object.entries(dict)) {
      mesh.morphTargetInfluences[idx] = influences[name] ?? 0;
    }
  }

  if (jawBone) {
    const relativeAngle = minJawAngle + jawValue * (maxJawAngle - minJawAngle);
    jawBone.rotation[jawAxis] = jawRestAngle + relativeAngle;
  }
}

// Called by app.js when a new audio chunk starts playing.
window.setLipSyncData = function (timeline, audio) {
  lipSyncTimeline = timeline;
  lipSyncAudio    = audio;
};

// Called by app.js after it resolves the backend port and fetches /model-config.
window.initAvatar = function (config, port) {
  visemeMap = config.visemeMap ?? {};
  jawAxis = config.jawAxis ?? 'x';
  minJawAngle = config.minJawAngle ?? 0;
  maxJawAngle = config.maxJawAngle ?? 0.15;

  new GLTFLoader().load(`http://127.0.0.1:${port}/model`, (gltf) => {
    const model = gltf.scene;

    // Center horizontally, feet at y=0.
    const box    = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    model.position.set(-center.x, -box.min.y, -center.z);
    scene.add(model);

    // Frame camera to fit full character height.
    const halfFov = (camera.fov / 2) * (Math.PI / 180);
    const dist    = (size.y / 2) / Math.tan(halfFov) * 1.15;
    camera.position.set(0, size.y * 0.5, dist);
    camera.lookAt(0, size.y * 0.5, 0);

    _fullBodyCam = { camY: size.y * 0.50, camZ: dist,        lookY: size.y * 0.50 };
    _headShotCam = { camY: size.y * 0.90, camZ: dist * 0.18, lookY: size.y * 0.90 };

    if (gltf.animations.length > 0) {
      mixer = new THREE.AnimationMixer(model);
      // mixer.clipAction(gltf.animations[0]).play();
    }

    const jawBoneName = config.jawBone ?? 'CC_Base_JawRoot';
    model.traverse(node => {
      if (node.isMesh && node.morphTargetDictionary && node.morphTargetInfluences) {
        morphMeshes.set(node, node.morphTargetDictionary);
      }
      if (node.name === jawBoneName) {
        jawBone = node;
        // Store the initial rotation angle in the jaw axis.
        jawRestAngle = node.rotation[jawAxis] ?? 0;
      }
    });
  }, undefined, (err) => console.error('[avatar] model load failed', err));
};

// --- Resize handling ---
function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(canvas);
resize();

// --- Render loop ---
function _tick() {
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);
  if (!lipSyncDisabled && (morphMeshes.size > 0 || jawBone)) {
    const t = lipSyncAudio ? lipSyncAudio.currentTime : Infinity;
    applyLipSync(t);
  }
  renderer.render(scene, camera);
}

(function frame() {
  requestAnimationFrame(frame);
  _tick();
})();

// requestAnimationFrame is paused when the page is hidden (user switches away).
// A self-scheduling setTimeout keeps lip sync running for screen-capture tools
// that grab this window while it is in the background.  setTimeout rather than
// setInterval so callbacks never pile up if a frame runs long.
let _hiddenLoopActive = false;
function _hiddenTick() {
  if (!_hiddenLoopActive) return;
  _tick();
  setTimeout(_hiddenTick, 16);
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    _hiddenLoopActive = true;
    _hiddenTick();
  } else {
    _hiddenLoopActive = false;
    clock.getDelta(); // reset accumulated delta so the first rAF frame is normal
  }
});

// Stub — will drive emotion animations in a later stage.
window.setEmotion = (_emotion) => {};

// Debug helper: show a phoneme shape, optionally blending to another.
// ratio: 1.0 = fully ph1, 0.0 = fully ph2, 0.5 = blend (using max)
// Usage: showPhoneme('ɑ')  or  showPhoneme('p', 'ə', 0.5)
window.showPhoneme = function (ph1, ph2 = null, ratio = 1.0) {
  lipSyncDisabled = true;   // Prevent render loop from clearing morphs
  lipSyncAudio = null;      // Cleanup just in case
  lipSyncTimeline = [];     // Clear timeline

  const mapping1 = visemeMap[ph1] || {};
  const mapping2 = ph2 ? (visemeMap[ph2] || {}) : {};

  const influences = {};
  let jawValue = 0;

  const allKeys = new Set([...Object.keys(mapping1), ...Object.keys(mapping2)]);
  for (const key of allKeys) {
    const v1 = mapping1[key] ?? 0;
    const v2 = mapping2[key] ?? 0;
    if (key === 'jaw') {
      jawValue = Math.max(v1 * ratio, v2 * (1 - ratio));
    } else {
      influences[key] = Math.max(v1 * ratio, v2 * (1 - ratio));
    }
  }

  console.log(`[showPhoneme] computed influences:`, influences, `jawValue=${jawValue}`);

  for (const [mesh, dict] of morphMeshes) {
    for (const [name, idx] of Object.entries(dict)) {
      mesh.morphTargetInfluences[idx] = influences[name] ?? 0;
    }
  }

  if (jawBone) {
    const relativeAngle = minJawAngle + jawValue * (maxJawAngle - minJawAngle);
    jawBone.rotation[jawAxis] = jawRestAngle + relativeAngle;
    console.log(`[showPhoneme] jaw: axis=${jawAxis}, restAngle=${jawRestAngle}, relative=${minJawAngle + jawValue * (maxJawAngle - minJawAngle)}, final=${jawRestAngle + relativeAngle}`);
  }
};
