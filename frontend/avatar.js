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

// Built from the loaded model; used by applyLipSync.
const morphMeshes = new Map();  // Map<Mesh, morphTargetDictionary>
let jawBone    = null;
let minJawAngle = 0;
let maxJawAngle = 0.15;

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
    jawBone.rotation.x = minJawAngle + jawValue * (maxJawAngle - minJawAngle);
  }
}

// Called by app.js when a new audio chunk starts playing.
window.setLipSyncData = function (timeline, audio) {
  lipSyncTimeline = timeline;
  lipSyncAudio    = audio;
};

// Called by app.js after it resolves the backend port and fetches /model-config.
window.initAvatar = function (config, port) {
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

    if (gltf.animations.length > 0) {
      mixer = new THREE.AnimationMixer(model);
      mixer.clipAction(gltf.animations[0]).play();
    }

    const jawBoneName = config.jawBone ?? 'CC_Base_JawRoot';
    model.traverse(node => {
      if (node.isMesh && node.morphTargetDictionary && node.morphTargetInfluences) {
        morphMeshes.set(node, node.morphTargetDictionary);
      }
      if (node.name === jawBoneName) jawBone = node;
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
(function frame() {
  requestAnimationFrame(frame);
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);

  // Always run applyLipSync so morphs are zeroed when there is no active audio.
  if (morphMeshes.size > 0 || jawBone) {
    const t = lipSyncAudio ? lipSyncAudio.currentTime : Infinity;
    applyLipSync(t);
  }

  renderer.render(scene, camera);
})();

// Stub — will drive emotion animations in a later stage.
window.setEmotion = (_emotion) => {};
