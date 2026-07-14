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

// Model state — populated by initAvatar after the GLB loads.
const morphMeshes = new Map();  // Map<Mesh, morphTargetDictionary>
let jawBone      = null;
let jawAxis      = 'x';
let jawRestAngle = 0;
let minJawAngle  = 0;
let maxJawAngle  = 0.15;
let _visemeMap   = {};   // phoneme → {morph: value, ...}; used by showPhoneme
let _tagsConfig  = {};   // tag name → {morph: value, ...}; used by setEmotion

// Scroll-wheel zoom — 0 = full body, 1 = head shot.
let zoomT = 0;
let _fullBodyCam = null;
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

// ── Utilities ─────────────────────────────────────────────────────────────────

// Box-Muller — standard normal sample.
function _randn() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function _smoothstep(t) { return t * t * (3 - 2 * t); }

// Gaussian half-bell used for lip sync coarticulation envelopes.
function _envelopeCurve(x) { return Math.exp(-2 * (x - 1) ** 2); }

function _sampleEnvelope({ absStart, absPeak, absEnd, value }, t) {
  if (t < absStart || t > absEnd) return 0;
  const span = t <= absPeak ? (absPeak - absStart) : (absEnd - absPeak);
  if (span < 1e-6) return value;
  const x = 1 - Math.abs(t - absPeak) / span;
  return value * _envelopeCurve(x);
}

// ── Animation drivers ─────────────────────────────────────────────────────────
// Each driver implements tick(delta) → { morphs: {name: 0..1, ...}, jaw?: 0..1 }

class LipSyncDriver {
  constructor() {
    this._timeline = [];
    this._audio    = null;
  }

  setData(timeline, audio) {
    this._timeline = timeline;
    this._audio    = audio;
  }

  tick(_delta) {
    const t = this._audio ? this._audio.currentTime : Infinity;
    const morphs = {};
    let jaw = 0;

    for (const ev of this._timeline) {
      const v = _sampleEnvelope(ev, t);
      if (v <= 0) continue;
      if (ev.key === 'jaw') jaw = Math.max(jaw, v);
      else morphs[ev.key] = Math.max(morphs[ev.key] ?? 0, v);
    }

    return { morphs, jaw };
  }
}

// Schedules blinks using an Ornstein-Uhlenbeck (AR(1)) process for inter-blink
// intervals: IBI_{n+1} = μ + ρ·(IBI_n − μ) + σ_eff·N(0,1), ρ = exp(−θ).
// This reproduces the observed autocorrelation in human blink sequences — short
// gaps tend to cluster, then revert toward the mean.
class BlinkDriver {
  constructor(cfg = {}) {
    this._morphNames = cfg.morphs         ?? ['Eyes_Blink'];
    this._closeT     = cfg.close_duration ?? 0.12;
    this._openT      = cfg.open_duration  ?? 0.20;
    this._mu         = cfg.ou_mean        ?? 4.0;
    this._rho        = Math.exp(-(cfg.ou_theta ?? 0.7));
    this._sigEff     = (cfg.ou_sigma ?? 1.5) * Math.sqrt(1 - this._rho ** 2);
    this._min        = cfg.ou_min ?? 0.4;
    this._max        = cfg.ou_max ?? 12.0;
    this._lastIBI    = this._mu;
    this._countdown  = this._sampleIBI();
    this._phase      = 'open';   // 'open' | 'closing' | 'opening'
    this._phaseT     = 0;
  }

  _sampleIBI() {
    const raw = this._mu + this._rho * (this._lastIBI - this._mu)
              + this._sigEff * _randn();
    this._lastIBI = Math.max(this._min, Math.min(this._max, raw));
    return this._lastIBI;
  }

  tick(delta) {
    let value = 0;

    if (this._phase === 'open') {
      this._countdown -= delta;
      if (this._countdown <= 0) {
        this._phase  = 'closing';
        this._phaseT = 0;
      }
    } else if (this._phase === 'closing') {
      this._phaseT += delta;
      const t = this._phaseT / this._closeT;
      if (t >= 1) { this._phase = 'opening'; this._phaseT = 0; value = 1; }
      else          value = _smoothstep(t);
    } else {   // 'opening'
      this._phaseT += delta;
      const t = this._phaseT / this._openT;
      if (t >= 1) { this._phase = 'open'; this._countdown = this._sampleIBI(); }
      else          value = _smoothstep(1 - t);
    }

    if (value <= 0) return { morphs: {} };
    const morphs = {};
    for (const name of this._morphNames) morphs[name] = value;
    return { morphs };
  }
}

// Drives emotion tag morph shapes. Registered with 'add' blend so emotion
// contributions stack on top of lip sync without zeroing it. The active tag's
// morphs hold steady (no envelope) until replaced by the next tag or cleared.
class EmotionDriver {
  constructor() {
    this._morphs = {};
  }
  tick(_delta) { return { morphs: this._morphs }; }
  setMorphs(morphs) { this._morphs = morphs; }
}

// ── Animation compositor ──────────────────────────────────────────────────────
// Collects contributions from all registered drivers each frame, merges them,
// and performs a single write to mesh.morphTargetInfluences + jaw bone.
//
// Blend modes per driver:
//   'max'      – per-morph maximum across all max-mode drivers (default)
//   'add'      – accumulated on top of the max result, clamped to 1
//   'override' – applied unconditionally after all other modes

class AnimationCompositor {
  constructor() {
    this._drivers = [];    // [{name, driver, blend}]
    this.bypass   = false; // when true, tick() is a no-op (e.g. during showPhoneme)
  }

  // Returns the driver instance for convenience.
  register(name, driver, blend = 'max') {
    this._drivers.push({ name, driver, blend });
    return driver;
  }

  tick(delta) {
    if (this.bypass) return;

    const maxM = {}, addM = {}, ovM = {};
    let maxJaw = 0, addJaw = 0, ovJaw = null;

    for (const { driver, blend } of this._drivers) {
      const { morphs = {}, jaw = null } = driver.tick(delta);
      if (blend === 'max') {
        for (const [k, v] of Object.entries(morphs)) maxM[k] = Math.max(maxM[k] ?? 0, v);
        if (jaw !== null) maxJaw = Math.max(maxJaw, jaw);
      } else if (blend === 'add') {
        for (const [k, v] of Object.entries(morphs)) addM[k] = (addM[k] ?? 0) + v;
        if (jaw !== null) addJaw += jaw;
      } else {   // 'override'
        Object.assign(ovM, morphs);
        if (jaw !== null) ovJaw = jaw;
      }
    }

    // Merge phases: max → add on top → override wins last.
    const final = { ...maxM };
    for (const [k, v] of Object.entries(addM)) final[k] = Math.min(1, (final[k] ?? 0) + v);
    Object.assign(final, ovM);
    const finalJaw = ovJaw ?? Math.min(1, maxJaw + addJaw);

    for (const [mesh, dict] of morphMeshes) {
      for (const [name, idx] of Object.entries(dict))
        mesh.morphTargetInfluences[idx] = final[name] ?? 0;
    }

    if (jawBone) {
      const rel = minJawAngle + finalJaw * (maxJawAngle - minJawAngle);
      jawBone.rotation[jawAxis] = jawRestAngle + rel;
    }
  }
}

// ── Singleton instances ───────────────────────────────────────────────────────

const compositor    = new AnimationCompositor();
const lipSyncDriver = compositor.register('lipSync', new LipSyncDriver());
const emotionDriver = compositor.register('emotion', new EmotionDriver(), 'add');
// BlinkDriver is registered inside initAvatar once blink config is available.

// ── Public API ────────────────────────────────────────────────────────────────

// Called by app.js when a new audio chunk starts playing.
window.setLipSyncData = function (timeline, audio) {
  lipSyncDriver.setData(timeline, audio);
};

// Called by app.js when a {tag} mark's time is reached, or with null/undefined
// to clear back to no emotion override (e.g. when speech ends).
window.setEmotion = function (tagName) {
  emotionDriver.setMorphs(tagName ? (_tagsConfig[tagName] ?? {}) : {});
};

// Called by app.js after it resolves the backend port and fetches /model-config.
window.initAvatar = function (config, port) {
  _visemeMap  = config.visemeMap   ?? {};
  _tagsConfig = config.tags        ?? {};
  jawAxis     = config.jawAxis     ?? 'x';
  minJawAngle = config.minJawAngle ?? 0;
  maxJawAngle = config.maxJawAngle ?? 0.15;

  if (config.blink) {
    compositor.register('blink', new BlinkDriver(config.blink), 'override');
  }

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
        jawRestAngle = node.rotation[jawAxis] ?? 0;
      }
    });
  }, undefined, (err) => console.error('[avatar] model load failed', err));
};

// ── Resize handling ───────────────────────────────────────────────────────────

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

// ── Render loop ───────────────────────────────────────────────────────────────

function _tick() {
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);
  compositor.tick(delta);
  renderer.render(scene, camera);
}

(function frame() {
  requestAnimationFrame(frame);
  _tick();
})();

// requestAnimationFrame is paused when the page is hidden (user switches away).
// A self-scheduling setTimeout keeps animations running for screen-capture tools
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

// ── Debug helper ──────────────────────────────────────────────────────────────

// Show a phoneme shape, optionally blending to another.
// ratio: 1.0 = fully ph1, 0.0 = fully ph2, 0.5 = blend (using max)
// Usage: showPhoneme('ɑ')  or  showPhoneme('p', 'ə', 0.5)
window.showPhoneme = function (ph1, ph2 = null, ratio = 1.0) {
  compositor.bypass = true;
  lipSyncDriver.setData([], null);

  const mapping1 = _visemeMap[ph1] || {};
  const mapping2 = ph2 ? (_visemeMap[ph2] || {}) : {};

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
    for (const [name, idx] of Object.entries(dict))
      mesh.morphTargetInfluences[idx] = influences[name] ?? 0;
  }

  if (jawBone) {
    const relativeAngle = minJawAngle + jawValue * (maxJawAngle - minJawAngle);
    jawBone.rotation[jawAxis] = jawRestAngle + relativeAngle;
    console.log(`[showPhoneme] jaw: axis=${jawAxis}, restAngle=${jawRestAngle}, relative=${relativeAngle}, final=${jawRestAngle + relativeAngle}`);
  }
};
