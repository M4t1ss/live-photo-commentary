import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
import { VRMAnimationLoaderPlugin, createVRMAnimationClip } from '@pixiv/three-vrm-animation';

const canvas = document.getElementById('avatar-canvas');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 500);


const clock = new THREE.Clock();
let mixer         = null;
let channelMixer  = null;
let currentVrm    = null;
let _idleClips    = [];
let _talkingClips = [];
let _walkInClips  = [];
let _walkOutClips = [];
let _helloClips    = [];
let _goodbyeClips  = [];
let _dancingClips  = [];
let _lonelyClips   = [];
let _tagAnimClips  = {};   // tag name → [clip, ...] for one-shot emotion animations
let sceneModel    = null;
// hidden → walk_in → greeting → running → farewell → walking_out → hidden
let _lifecycleState = 'hidden';
// Sub-state within 'running': 'idle' | 'lonely' | 'dancing'
let _runSubstate  = 'idle';
let _idleToDanceSecs = 180;
const AVATAR_FADE_SECS = 2;   // canvas opacity fade on spawn / despawn
let _idleTimer       = 0;
let _isTalking       = false;   // pipeline TTS overlay
let _isSystemTalking = false;   // system message TTS overlay
let _systemTtsActive = false;   // synthesis request in-flight
let _greetingNeedsTransition = false;
let _farewellNeedsTransition = false;
let _isEmotionAnimPlaying    = false;
let _requestSynthesis = null;   // fn(text) — set by app.js

// Hardcoded messages — placeholders until configurable per-model.
const WELCOME_MESSAGE = "Hello! I'm your interactive live commentary assistant. I'll be watching what you do and sharing my thoughts!";
const FAREWELL_MESSAGE = "Goodbye! It was lovely watching over you today. See you next time!";
const LONELY_MESSAGE   = "Hey... are you still there? It's getting a bit quiet. I'm starting to feel lonely.";

// Model state — populated by initAvatar after the GLB loads.
const morphMeshes = new Map();  // Map<Mesh, morphTargetDictionary>
let jawBone      = null;
let jawAxis      = 'x';
let jawRestAngle = 0;
let minJawAngle  = 0;
let maxJawAngle  = 0.15;

// Gaze-tracking (look-at-camera) state — see _computeSwingClampedLookAt /
// _updateGazeTracking. Bones are re-aimed at the camera in this order, neck
// first, so each bone's parent-space math sees its parent's already-applied
// correction — eyes only pick up the remaining angle neck+head couldn't
// reach within their own swing limits.
const GAZE_BONE_KEYS = ['neck', 'head', 'leftEye', 'rightEye'];
const GAZE_DEFAULT_MAX_ANGLE = { neck: 0.3, head: 0.5, leftEye: 0.15, rightEye: 0.15 };
// Each entry: { bone, restLocalQuat, bindWorldQuat, bindForwardWorld, maxAngle,
//              elastic, currentQuat }
// elastic=true for neck/head (spring-smoothed); eyes snap directly to target.
let _gazeChain = [];
let _gazeSpeed = 4.0;          // exponential-smoothing speed for neck/head (rad⁻¹)
let _gazeSuspendDriver = null; // GazeSuspendDriver instance, or null if not configured

let _visemeMap   = {};   // phoneme → {morph: value, ...}; used by showPhoneme
let _tagMorphs   = {};   // tag name → {morph: value, ...}; used by setEmotion
let _emotionRampDuration = 0.3;  // seconds for a full 0→1 emotion sweep; reuses maxEnvelopeDuration

// Scroll-wheel zoom — 0 = full body, 1 = head shot.
let zoomT = 0;
let _fullBodyCam = null;
let _headShotCam = null;

// Right-click + drag orbit — yaw/pitch offset applied on top of the
// zoom-driven camera position. No panning: the focus is always the avatar.
let orbitYaw   = 0;
let orbitPitch = 0;
let _orbiting  = false; // true from right-mousedown until the resulting contextmenu event is consumed
const ORBIT_SPEED = 0.005; // radians per pixel dragged
const ORBIT_PITCH_LIMIT = Math.PI / 2 - 0.05; // clamp just short of straight up/down

function _applyZoom() {
  if (!_fullBodyCam) return;
  const s = zoomT * zoomT * (3 - 2 * zoomT);  // smoothstep
  const camY  = _fullBodyCam.camY  + s * (_headShotCam.camY  - _fullBodyCam.camY);
  const camZ  = _fullBodyCam.camZ  + s * (_headShotCam.camZ  - _fullBodyCam.camZ);
  const lookY = _fullBodyCam.lookY + s * (_headShotCam.lookY - _fullBodyCam.lookY);

  // Re-express the zoom-driven position in spherical terms around the look
  // target, then add the orbit yaw/pitch offset on top.
  const r     = Math.hypot(camZ, camY - lookY);
  const pitch = Math.atan2(camY - lookY, camZ) + orbitPitch;
  camera.position.set(
    r * Math.cos(pitch) * Math.sin(orbitYaw),
    lookY + r * Math.sin(pitch),
    r * Math.cos(pitch) * Math.cos(orbitYaw)
  );
  camera.lookAt(0, lookY, 0);
}

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  zoomT = Math.max(0, Math.min(1, zoomT - e.deltaY * 0.001));
  _applyZoom();
}, { passive: false });

// Suppressed at the document level (not just on canvas) so releasing the
// right button outside the avatar pane — another panel, or outside the
// window — doesn't pop the native context menu there instead.
document.addEventListener('contextmenu', (e) => {
  if (!_orbiting) return;
  e.preventDefault();
  _orbiting = false;
});

canvas.addEventListener('mousedown', (e) => {
  if (e.button !== 2) return;
  e.preventDefault();
  _orbiting = true;
  let lastX = e.clientX;
  let lastY = e.clientY;

  function onMove(e) {
    orbitYaw   -= (e.clientX - lastX) * ORBIT_SPEED;
    orbitPitch  = Math.max(-ORBIT_PITCH_LIMIT, Math.min(ORBIT_PITCH_LIMIT, orbitPitch + (e.clientY - lastY) * ORBIT_SPEED));
    lastX = e.clientX;
    lastY = e.clientY;
    _applyZoom();
  }

  function onUp() {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
  }

  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
});

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
// contributions stack on top of lip sync without zeroing it. Values ramp
// linearly toward the active tag's target instead of snapping, at the same
// rate as the lip sync envelope (a full 0→1 sweep takes _emotionRampDuration).
class EmotionDriver {
  constructor() {
    this._current = {};   // morph name → current (animated) value
    this._target  = {};   // morph name → target value from the active tag
  }

  setMorphs(morphs) {
    this._target = morphs ?? {};
  }

  tick(delta) {
    const step = _emotionRampDuration > 0 ? delta / _emotionRampDuration : Infinity;
    const keys = new Set([...Object.keys(this._current), ...Object.keys(this._target)]);
    for (const key of keys) {
      const from = this._current[key] ?? 0;
      const to   = this._target[key] ?? 0;
      if (from === to) continue;
      const diff = to - from;
      const next = Math.abs(diff) <= step ? to : from + Math.sign(diff) * step;
      if (next === 0) delete this._current[key];
      else this._current[key] = next;
    }
    return { morphs: this._current };
  }
}

// Alternates between "tracking" (gaze aimed at camera) and "suspended" (let
// animation drive the bones). Durations use the same OU process as BlinkDriver.
// Two mode configs (idle / talking) are selected via setTalking(); the active
// config is applied on the next duration sample — no mid-phase reset.
//
// No explicit blend states: the elastic spring in _updateGazeTracking provides
// smooth transitions automatically — the target just changes, the spring does
// the rest.
class GazeSuspendDriver {
  constructor(cfg = {}) {
    this._state      = 'tracking';
    this._isTalking  = false;
    this._idleCfg    = cfg.idle    ?? {};
    this._talkingCfg = cfg.talking ?? {};
    this._lastContactIBI = null;
    this._lastAwayIBI    = null;
    this._countdown  = this._sampleDuration(this._idleCfg, true);
  }

  setTalking(isTalking) { this._isTalking = isTalking; }

  // Returns 'tracking' or 'suspended'.
  tick(delta) {
    this._countdown -= delta;
    if (this._countdown <= 0) {
      const modeCfg = this._isTalking ? this._talkingCfg : this._idleCfg;
      if (this._state === 'tracking') {
        this._state     = 'suspended';
        this._countdown = this._sampleDuration(modeCfg, false);
      } else {
        this._state     = 'tracking';
        this._countdown = this._sampleDuration(modeCfg, true);
      }
    }
    return this._state;
  }

  _sampleDuration(cfg, isContact) {
    const mu   = isContact ? (cfg.contact_mean ?? 3.0) : (cfg.away_mean ?? 5.0);
    const minV = isContact ? (cfg.contact_min  ?? 1.0) : (cfg.away_min  ?? 1.0);
    const maxV = isContact ? (cfg.contact_max  ?? 8.0) : (cfg.away_max  ?? 12.0);
    const rho  = Math.exp(-(cfg.ou_theta ?? 0.5));
    const sig  = (cfg.ou_sigma ?? 1.0) * Math.sqrt(1 - rho ** 2);
    const last = isContact ? (this._lastContactIBI ?? mu) : (this._lastAwayIBI ?? mu);
    const raw  = mu + rho * (last - mu) + sig * _randn();
    const v    = Math.max(minV, Math.min(maxV, raw));
    if (isContact) this._lastContactIBI = v; else this._lastAwayIBI = v;
    return v;
  }
}

// Generic multi-channel animation mixer. Each named channel holds a clip pool
// and a single active Three.js action. Channels fade in/out independently;
// channels that reach weight 0 are removed on the next cleanup() call.
//
// Three.js weight note: effectiveWeight = action.weight * interpolant.
// fadeIn/fadeOut only move the interpolant (0→1 or 1→0). action.weight must
// be 1 for the fade to actually reach full strength. setEffectiveWeight(0)
// would zero the product permanently — never use it before fadeIn.
class ChannelMixer {
  constructor(mixer) {
    this._mixer    = mixer;
    this._channels = new Map(); // name → { clips, action, targetWeight, blendDuration, once, onFinish }
    mixer.addEventListener('loop',     e => this._onLoop(e));
    mixer.addEventListener('finished', e => this._onFinished(e));
  }

  // Start or fade in a named channel. If already active, only the clip pool
  // is updated. If it was stopping, the fade is reversed.
  play(name, clips, { blendDuration = 0.4 } = {}) {
    if (!clips.length) return;
    const existing = this._channels.get(name);

    if (existing) {
      existing.clips = clips;
      if (existing.targetWeight > 0) return;
      // Was stopping — reverse the fade
      existing.targetWeight = 1;
      existing.action.stopFading();
      existing.action.enabled = true;
      existing.action.setEffectiveWeight(1).play();
      console.log(`[anim] resume ${name}: "${existing.action.getClip().name}"`);
      return;
    }

    const clip = this._pickRandom(clips);
    const action = this._mixer.clipAction(clip);
    action.enabled = true;
    action.setLoop(THREE.LoopRepeat, Infinity);

    // If other channels are active, fade in so we don't pop. Otherwise start
    // immediately — avoids a T-pose flash while the fade-in ramps up.
    const hasOtherActive = [...this._channels.values()].some(ch => ch.targetWeight > 0);
    if (hasOtherActive) {
      action.setEffectiveWeight(1).fadeIn(blendDuration).play();
    } else {
      action.setEffectiveWeight(1).play();
    }
    console.log(`[anim] play ${name} (${clips.length} clips): "${clip.name}"`);
    this._channels.set(name, { clips, action, targetWeight: 1, blendDuration });
  }

  // Fade out a named channel. It is removed from the state map once Three.js
  // disables the action (see cleanup()).
  stop(name, { blendDuration } = {}) {
    const ch = this._channels.get(name);
    if (!ch || ch.targetWeight === 0) return;
    ch.action.fadeOut(blendDuration ?? ch.blendDuration);
    ch.targetWeight = 0;
  }

  // Play a clip once (LoopOnce). When the animation finishes naturally it fades
  // out and calls onFinish. If clips is empty, onFinish is called immediately.
  playOnce(name, clips, { blendDuration = 0.4, onFinish = null } = {}) {
    if (!clips.length) { console.log(`[anim] playOnce ${name}: no clips, skipping`); onFinish?.(); return; }
    const existing = this._channels.get(name);
    if (existing) { existing.action.stop(); this._channels.delete(name); }
    const clip   = this._pickRandom(clips);
    const action = this._mixer.clipAction(clip);
    action.reset();
    action.enabled = true;
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    const hasOtherActive = [...this._channels.values()].some(ch => ch.targetWeight > 0);
    if (hasOtherActive) {
      action.setEffectiveWeight(1).fadeIn(blendDuration).play();
    } else {
      action.setEffectiveWeight(1).play();
    }
    console.log(`[anim] playOnce ${name} (${clips.length} clips): "${clip.name}" duration=${clip.duration.toFixed(2)}s`);
    this._channels.set(name, { clips, action, targetWeight: 1, blendDuration, once: true, onFinish });
  }

  // Must be called every frame after mixer.update(). Removes channels whose
  // fadeOut has completed (Three.js sets action.enabled = false when weight
  // reaches 0 — more reliable than polling getEffectiveWeight()).
  cleanup() {
    for (const [name, ch] of this._channels) {
      if (ch.targetWeight === 0 && !ch.action.enabled) {
        ch.action.stop();
        this._channels.delete(name);
      }
    }
  }

  // Duration (s) of the clip currently active on a channel, or 0 if none.
  channelDuration(name) {
    return this._channels.get(name)?.action.getClip().duration ?? 0;
  }

  // Returns true/false if any active clip has a non-null eyeContact value, else null.
  // Scans channels in insertion order; first non-null wins.
  getEyeContactOverride() {
    for (const [, ch] of this._channels) {
      if (ch.targetWeight > 0) {
        const ec = ch.action.getClip().userData?.eyeContact;
        if (ec !== null && ec !== undefined) return ec;
      }
    }
    return null;
  }

  _pickRandom(clips) {
    return clips[Math.floor(Math.random() * clips.length)];
  }

  _onLoop(e) {
    for (const [, ch] of this._channels) {
      if (e.action === ch.action && ch.targetWeight > 0) {
        this._swapClip(ch);
        return;
      }
    }
  }

  _onFinished(e) {
    for (const [, ch] of this._channels) {
      if (e.action === ch.action && ch.once) {
        ch.action.fadeOut(ch.blendDuration);
        ch.targetWeight = 0;
        ch.onFinish?.();
        return;
      }
    }
  }

  // Crossfade to a randomly-picked clip from the channel's pool.
  // setEffectiveWeight(1) sets weight=1 before fadeIn so the interpolant
  // (0→1) actually reaches full strength: 1 * (0→1) = 0→1.
  _swapClip(ch) {
    const newClip = this._pickRandom(ch.clips);
    if (!newClip || newClip === ch.action.getClip()) return;
    const newAct = this._mixer.clipAction(newClip);
    newAct.enabled = true;
    newAct.setLoop(THREE.LoopRepeat, Infinity).setEffectiveWeight(1).fadeIn(ch.blendDuration).play();
    ch.action.fadeOut(ch.blendDuration);
    console.log(`[anim] swap → "${newClip.name}"`);
    ch.action = newAct;
  }
}

// ── Avatar lifecycle ──────────────────────────────────────────────────────────
// hidden → (startAvatar) → walk_in → greeting → running → (stopAvatar) → farewell → walking_out → hidden
//
// startAvatar / stopAvatar are called from app.js on Play / Stop. If walk_in /
// walk_out / hello clip lists are empty (model.yaml omits them), the
// corresponding phase is skipped (playOnce calls onFinish immediately).

let _pendingStart = false;

// Starts a system TTS synthesis; returns true if the request was dispatched,
// false if no synthesis callback is configured (transition should be skipped).
function _startSystemTts(text) {
  if (!_requestSynthesis) return false;
  _systemTtsActive = true;
  _requestSynthesis(text);
  return true;
}

function _onWalkInDone() {
  if (_lifecycleState !== 'walk_in') return;
  _lifecycleState = 'greeting';
  _greetingNeedsTransition = false;
  const hasTts = _startSystemTts(WELCOME_MESSAGE);
  channelMixer.playOnce('greeting', _helloClips, { onFinish: hasTts ? _onHelloDone : _onGreetingDone });
}

function _onHelloDone() {
  if (_lifecycleState !== 'greeting') return;
  if (!_systemTtsActive) {
    _onGreetingDone();
  } else {
    // Hello finished but message still in-flight — idle fills the wait.
    _greetingNeedsTransition = true;
    if (_idleClips.length) channelMixer.play('idle', _idleClips);
  }
}

function _onGreetingDone() {
  if (_lifecycleState !== 'greeting') return;
  _lifecycleState = 'running';
  _runSubstate = 'idle';
  _idleTimer = 0;
  _isTalking = false;
  channelMixer.stop('greeting');
  if (_idleClips.length) channelMixer.play('idle', _idleClips);
}

function _onGoodbyeDone() {
  if (_lifecycleState !== 'farewell') return;
  if (!_systemTtsActive) {
    _onFarewellDone();
  } else {
    // Goodbye finished but message still in-flight — idle fills the wait.
    _farewellNeedsTransition = true;
    if (_idleClips.length) channelMixer.play('idle', _idleClips);
  }
}

function _onFarewellDone() {
  if (_lifecycleState !== 'farewell') return;
  _lifecycleState = 'walking_out';
  channelMixer.stop('idle');
  channelMixer.playOnce('walk_out', _walkOutClips, { onFinish: _onWalkOutDone });

  // Fade the canvas out so it hits 0 exactly as she's hidden — in case the
  // camera is aimed at where she vanishes. No-op if walk_out was skipped.
  const walkOutSecs = channelMixer.channelDuration('walk_out');
  if (walkOutSecs > 0) {
    setTimeout(() => {
      canvas.style.transition = `opacity ${AVATAR_FADE_SECS}s linear`;
      canvas.style.opacity = '0';
    }, Math.max(0, walkOutSecs - AVATAR_FADE_SECS) * 1000);
  }
}

function _onLonelyMessageDone() {
  if (_lifecycleState !== 'running' || _runSubstate !== 'lonely') return;
  if (_dancingClips.length) {
    _runSubstate = 'dancing';
    channelMixer.play('dancing', _dancingClips);
    channelMixer.stop('lonely');
  } else {
    _runSubstate = 'idle';
    channelMixer.play('idle', _idleClips);
    channelMixer.stop('lonely');
  }
}

function _onWalkOutDone() {
  _lifecycleState = 'hidden';
  if (sceneModel) sceneModel.visible = false;
  canvas.style.transition = 'none';
  canvas.style.opacity = '1';
}

window.startAvatar = function () {
  if (_lifecycleState !== 'hidden') return;
  if (!sceneModel) { _pendingStart = true; return; }
  _lifecycleState = 'walk_in';
  sceneModel.visible = true;

  // Fade the canvas in over the walk-in — in case the camera is aimed at where
  // she appears. Snap to 0 with no transition, flush that so the browser commits
  // it, then transition up to 1.
  canvas.style.transition = 'none';
  canvas.style.opacity = '0';
  void canvas.offsetHeight;
  canvas.style.transition = `opacity ${AVATAR_FADE_SECS}s linear`;
  canvas.style.opacity = '1';

  channelMixer.playOnce('walk_in', _walkInClips, { onFinish: _onWalkInDone });
};

window.stopAvatar = function () {
  if (_lifecycleState === 'hidden' || _lifecycleState === 'farewell' || _lifecycleState === 'walking_out') return;
  if (!channelMixer) { _lifecycleState = 'hidden'; if (sceneModel) sceneModel.visible = false; return; }
  _lifecycleState = 'farewell';
  _runSubstate = 'idle';
  _isTalking = false;
  _idleTimer = 0;
  _farewellNeedsTransition = false;
  channelMixer.stop('idle');
  channelMixer.stop('talking');
  channelMixer.stop('dancing');
  channelMixer.stop('lonely');
  channelMixer.stop('walk_in');
  channelMixer.stop('greeting');
  channelMixer.stop('emotion_anim');
  const hasTts = _startSystemTts(FAREWELL_MESSAGE);
  channelMixer.playOnce('farewell', _goodbyeClips, { onFinish: hasTts ? _onGoodbyeDone : _onFarewellDone });
};

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

// ── Gaze tracking (look-at-camera) ──────────────────────────────────────────────
// Computes a local quaternion that re-aims a bone's known-good bind-pose
// forward direction at a world-space target, clamped to a maximum swing angle
// away from the bone's neutral orientation relative to its current parent.
// Bone-agnostic — drives neck, head, and each eye via the same math.
function _computeSwingClampedLookAt(bone, restLocalQuat, bindForwardWorld, bindWorldQuat, targetWorldPos, maxAngle) {
  const boneWorldPos = new THREE.Vector3();
  bone.getWorldPosition(boneWorldPos);
  const desiredForwardWorld = targetWorldPos.clone().sub(boneWorldPos).normalize();

  const deltaQuat = new THREE.Quaternion().setFromUnitVectors(bindForwardWorld, desiredForwardWorld);
  const desiredWorldQuat = deltaQuat.multiply(bindWorldQuat);

  const parentWorldQuat = new THREE.Quaternion();
  bone.parent.getWorldQuaternion(parentWorldQuat);
  const desiredLocalQuat = parentWorldQuat.invert().multiply(desiredWorldQuat);

  const swingQuat = restLocalQuat.clone().invert().multiply(desiredLocalQuat);
  const angle = 2 * Math.acos(Math.min(1, Math.abs(swingQuat.w)));
  const clampedSwing = angle > maxAngle
    ? new THREE.Quaternion().identity().slerp(swingQuat, maxAngle / angle)
    : swingQuat;

  return restLocalQuat.clone().multiply(clampedSwing);
}

// Re-aims each configured gaze bone in chain order (neck → head → eyes).
//
// The key insight: whether tracking the camera or returning to animation, it's
// always the same spring — currentQuat.slerp(target, alpha). Only the target
// changes:
//   tracking  → target is the clamped camera direction
//   suspended → target is the live animation pose (mixer.update() already wrote
//               it to bone.quaternion before this function runs each frame)
//
// This gives elastic easing in both directions with no separate blend logic.
// Neck/head use _gazeSpeed; eyes snap to the tracking target (alpha=1) but
// spring back toward animation when suspended, so suspension is still smooth.
function _updateGazeTracking(delta) {
  const suspendState  = _gazeSuspendDriver ? _gazeSuspendDriver.tick(delta) : 'tracking';
  const ecOverride    = channelMixer?.getEyeContactOverride() ?? null;
  const isTracking    = ecOverride !== null ? ecOverride : (suspendState === 'tracking');
  const alpha = 1 - Math.exp(-_gazeSpeed * delta);

  for (const g of _gazeChain) {
    if (isTracking) {
      const target = _computeSwingClampedLookAt(
        g.bone, g.restLocalQuat, g.bindForwardWorld, g.bindWorldQuat, camera.position, g.maxAngle
      );
      // Neck/head: elastic approach. Eyes: snap (alpha=1) per original design.
      g.currentQuat.slerp(target, g.elastic ? alpha : 1);
    } else {
      // Suspended: spring toward the animation pose the mixer wrote this frame.
      g.currentQuat.slerp(g.bone.quaternion, alpha);
    }
    g.bone.quaternion.copy(g.currentQuat);
    g.bone.updateMatrix();
    g.bone.updateWorldMatrix(false, false);
  }
}

// ── Singleton instances ───────────────────────────────────────────────────────

const compositor    = new AnimationCompositor();
const lipSyncDriver = compositor.register('lipSync', new LipSyncDriver());
const emotionDriver = compositor.register('emotion', new EmotionDriver(), 'add');
// BlinkDriver is registered inside initAvatar once blink config is available.

// ── Public API ────────────────────────────────────────────────────────────────

// Called by app.js when a pipeline TTS chunk starts or ends (audio=null).
window.setLipSyncData = function (timeline, audio) {
  lipSyncDriver.setData(timeline, audio);
  _gazeSuspendDriver?.setTalking(!!audio);
  if (!channelMixer || _lifecycleState !== 'running') return;
  if (audio && _talkingClips.length) {
    if (_runSubstate === 'lonely') {
      // Pipeline TTS interrupts the lonely sequence.
      _systemTtsActive = false;
      window.cancelSystemTts?.();
      channelMixer.stop('lonely');
      _runSubstate = 'idle';
    } else if (_runSubstate === 'dancing') {
      channelMixer.stop('dancing');
      _runSubstate = 'idle';
    }
    _idleTimer = 0;
    _isTalking = true;
    channelMixer.play('talking', _talkingClips);
    channelMixer.stop('idle');
  } else {
    if (_isTalking) _idleTimer = 0;
    _isTalking = false;
    _isEmotionAnimPlaying = false;
    channelMixer.stop('emotion_anim');
    if (_runSubstate === 'idle') {
      channelMixer.play('idle', _idleClips);
      channelMixer.stop('talking');
    } else if (_runSubstate === 'lonely') {
      channelMixer.stop('talking');
      if (!_systemTtsActive) _onLonelyMessageDone();
    } else if (_runSubstate === 'dancing') {
      channelMixer.stop('talking');
    }
  }
};

// Called by app.js when a {tag} mark's time is reached, or with null/undefined
// to clear back to no emotion override (e.g. when speech ends).
// A plain object is also accepted (e.g. from the console while tuning
// model.yaml) and used as the morph definition directly, bypassing tags config.
window.setEmotion = function (tagOrMorphs) {
  const morphs = typeof tagOrMorphs === 'string' ? (_tagMorphs[tagOrMorphs] ?? {}) : (tagOrMorphs ?? {});
  emotionDriver.setMorphs(morphs);

  if (typeof tagOrMorphs === 'string' && tagOrMorphs && tagOrMorphs !== 'neutral') {
    const clips = _tagAnimClips[tagOrMorphs];
    if (clips && clips.length && (_isTalking || _isSystemTalking) && channelMixer) {
      _isEmotionAnimPlaying = true;
      channelMixer.stop('talking');
      channelMixer.playOnce('emotion_anim', clips, {
        onFinish: () => {
          _isEmotionAnimPlaying = false;
          if ((_isTalking || _isSystemTalking) && _talkingClips.length && channelMixer) {
            channelMixer.play('talking', _talkingClips);
          }
        }
      });
    }
  }
};

// Resolves a trim bound to seconds: a number passes through unchanged; a string
// ending in '%' is read as a fraction of the clip's duration ("50%" of a 4 s
// clip → 2). null/undefined → null.
function _resolveTrimBound(bound, duration) {
  if (bound == null) return null;
  if (typeof bound === 'string' && bound.trim().endsWith('%')) return duration * parseFloat(bound) / 100;
  return bound;
}

// Applies start/end trimming to a clip by sub-clipping it to [start, end].
// Bounds are seconds (number) or a percentage of the clip's duration (string
// ending in '%'). Uses subclip with fps=1000 to treat milliseconds as integer
// "frames".
function _trimClip(clip, start, end) {
  const s = _resolveTrimBound(start, clip.duration) ?? 0;
  const e = _resolveTrimBound(end,   clip.duration) ?? clip.duration;
  if (s === 0 && e === clip.duration) return clip;
  const trimmed = THREE.AnimationUtils.subclip(clip, clip.name, Math.round(s * 1000), Math.round(e * 1000), 1000);
  trimmed.userData = clip.userData;
  return trimmed;
}

// Resolves an animation clip from an entry (string filename, or object with
// {filename, eyeContact, start, end}). eyeContact (true/false/null) and any
// trim bounds are stored/applied so callers don't need to know the difference.
// Loads .vrma files from the backend; looks up other names in gltf.animations.
function _resolveAnimClip(entry, port, gltf, vrm) {
  const nameOrPath = typeof entry === 'string' ? entry : entry?.filename;
  const eyeContact = typeof entry === 'string' ? null  : (entry?.eyeContact ?? null);
  const start      = typeof entry === 'string' ? null  : (entry?.start      ?? null);
  const end        = typeof entry === 'string' ? null  : (entry?.end        ?? null);
  if (!nameOrPath) return Promise.resolve(null);
  if (nameOrPath.endsWith('.vrma')) {
    if (!vrm) {
      console.warn('[avatar] VRMA animations require a VRM model; skipping', nameOrPath);
      return Promise.resolve(null);
    }
    return new Promise((resolve) => {
      const vrmaLoader = new GLTFLoader();
      vrmaLoader.register((parser) => new VRMAnimationLoaderPlugin(parser));
      vrmaLoader.load(
        `http://127.0.0.1:${port}/animation/${encodeURIComponent(nameOrPath)}`,
        (vrmaGltf) => {
          try {
            const vrmAnimation = vrmaGltf.userData.vrmAnimations?.[0];
            if (!vrmAnimation) {
              console.warn(`[avatar] no VRM animation found in ${nameOrPath}`);
              resolve(null);
              return;
            }
            let clip = createVRMAnimationClip(vrmAnimation, vrm);
            clip.name = nameOrPath;
            clip.userData = { eyeContact };
            clip = _trimClip(clip, start, end);
            resolve(clip);
          } catch (err) {
            console.warn(`[avatar] failed to build clip from ${nameOrPath}:`, err);
            resolve(null);
          }
        },
        undefined,
        (err) => { console.warn(`[avatar] failed to load ${nameOrPath}:`, err); resolve(null); }
      );
    });
  }
  let clip = THREE.AnimationClip.findByName(gltf.animations, nameOrPath);
  if (!clip) { console.warn(`[avatar] animation "${nameOrPath}" not found in model`); return Promise.resolve(null); }
  clip = clip.clone();
  clip.userData = { eyeContact };
  clip = _trimClip(clip, start, end);
  return Promise.resolve(clip);
}

// Normalises a single entry or list of entries, loads all clips, returns the array.
async function _resolveAnimClips(namesOrPaths, port, gltf, vrm) {
  const names = Array.isArray(namesOrPaths) ? namesOrPaths : (namesOrPaths ? [namesOrPaths] : []);
  const clips = await Promise.all(names.map(n => _resolveAnimClip(n, port, gltf, vrm)));
  const resolved = clips.filter(Boolean);
  const label = names.map(n => (typeof n === 'string' ? n : n?.filename ?? '?')).join(', ');
  console.log(`[anim] resolved [${label}] → ${resolved.length}/${names.length} clips`);
  return resolved;
}

// Parse tags config — supports both old flat format ({ tag: {morph: v} }) and new
// format ({ tag: { morphs: {...}, animation: [...] } }). Returns morph-only dict.
function _parseTagMorphs(rawTags) {
  const result = {};
  for (const [tag, def] of Object.entries(rawTags)) {
    if (!def || typeof def !== 'object') { result[tag] = {}; continue; }
    result[tag] = ('morphs' in def || 'animation' in def) ? (def.morphs ?? {}) : def;
  }
  return result;
}

// Load one-shot animation clips for any tag that has an 'animation' array.
async function _loadTagAnimClips(rawTags, port, gltf, vrm) {
  const result = {};
  for (const [tag, def] of Object.entries(rawTags)) {
    if (def && typeof def === 'object' && Array.isArray(def.animation) && def.animation.length) {
      const clips = await _resolveAnimClips(def.animation, port, gltf, vrm);
      if (clips.length) result[tag] = clips;
    }
  }
  return result;
}

// Called by app.js after it resolves the backend port and fetches /model-config.
window.initAvatar = function (config, port) {
  _visemeMap  = config.visemeMap   ?? {};
  const rawTags = config.tags ?? {};
  _tagMorphs  = _parseTagMorphs(rawTags);
  _gazeSpeed  = config.gazeSpeed   ?? 4.0;
  if (config.gazeSuspension) _gazeSuspendDriver = new GazeSuspendDriver(config.gazeSuspension);
  jawAxis     = config.jawAxis     ?? 'x';
  minJawAngle = config.minJawAngle ?? 0;
  maxJawAngle = config.maxJawAngle ?? 0.15;
  _emotionRampDuration = config.maxEnvelopeDuration ?? 0.3;

  const lightDefs = config.lighting ?? [
    { type: 'ambient',     color: '#ffffff', intensity: 1.0 },
    { type: 'directional', color: '#fff4e0', intensity: 2.5, position: [1,  2,  3] },
    { type: 'directional', color: '#d0e8ff', intensity: 0.8, position: [-2, 0.5, -1] },
  ];
  for (const def of lightDefs) {
    let light;
    if (def.type === 'ambient') {
      light = new THREE.AmbientLight(def.color ?? '#ffffff', def.intensity ?? 1.0);
    } else if (def.type === 'directional') {
      light = new THREE.DirectionalLight(def.color ?? '#ffffff', def.intensity ?? 1.0);
      if (def.position) light.position.set(...def.position);
    } else {
      console.warn(`[avatar] unknown light type: ${def.type}`);
      continue;
    }
    scene.add(light);
  }

  if (config.blink) {
    compositor.register('blink', new BlinkDriver(config.blink), 'override');
  }
  
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));
  loader.register((parser) => new VRMAnimationLoaderPlugin(parser));
  loader.load(`http://127.0.0.1:${port}/model?t=${Date.now()}`, async (gltf) => {
    // const model = gltf.scene;
    const vrm = gltf.userData.vrm;
    currentVrm = vrm;
    window.gltf = gltf;
    const model = vrm ? vrm.scene : gltf.scene;

    // Center horizontally, feet at y=0.
    const box    = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    model.position.set(-center.x, -box.min.y, -center.z);
    model.visible = false; // hidden until startAvatar() is called
    scene.add(model);

    // Frame camera to fit full character height.
    const halfFov = (camera.fov / 2) * (Math.PI / 180);
    const dist    = (size.y / 2) / Math.tan(halfFov) * 1.15;
    camera.position.set(0, size.y * 0.9, dist);
    camera.lookAt(0, size.y * 0.5, 0);

    _fullBodyCam = { camY: size.y * 0.90, camZ: dist,        lookY: size.y * 0.50 };
    _headShotCam = { camY: size.y * 0.90, camZ: dist * 0.18, lookY: size.y * 0.90 };

    mixer = new THREE.AnimationMixer(model);
    [_idleClips, _talkingClips, _walkInClips, _walkOutClips, _helloClips, _goodbyeClips, _dancingClips, _lonelyClips] = await Promise.all([
      _resolveAnimClips(config.idleAnimation,    port, gltf, vrm),
      _resolveAnimClips(config.talkingAnimation, port, gltf, vrm),
      _resolveAnimClips(config.walkInAnimation,  port, gltf, vrm),
      _resolveAnimClips(config.walkOutAnimation, port, gltf, vrm),
      _resolveAnimClips(config.helloAnimation,   port, gltf, vrm),
      _resolveAnimClips(config.goodbyeAnimation, port, gltf, vrm),
      _resolveAnimClips(config.dancingAnimation, port, gltf, vrm),
      _resolveAnimClips(config.lonelyAnimation,  port, gltf, vrm),
    ]);
    _tagAnimClips = await _loadTagAnimClips(rawTags, port, gltf, vrm);
    channelMixer = new ChannelMixer(mixer);
    sceneModel = model; // set only after channelMixer is ready; startAvatar() uses !sceneModel as its "fully initialized" guard
    if (_pendingStart) { _pendingStart = false; window.startAvatar(); }
    // Idle starts after the walk_in → greeting sequence via startAvatar().

    const jawBoneName = config.jawBone ?? 'CC_Base_JawRoot';

    // config.{key}Bone / config.max{Key}Angle for each gaze-tracking bone —
    // e.g. neckBone/maxNeckAngle, leftEyeBone (shared maxEyeAngle with rightEye).
    const gazeBoneNames = {
      neck:     config.neckBone     ?? null,
      head:     config.headBone     ?? null,
      leftEye:  config.leftEyeBone  ?? null,
      rightEye: config.rightEyeBone ?? null,
    };
    const gazeMaxAngles = {
      neck:     config.maxNeckAngle ?? GAZE_DEFAULT_MAX_ANGLE.neck,
      head:     config.maxHeadAngle ?? GAZE_DEFAULT_MAX_ANGLE.head,
      leftEye:  config.maxEyeAngle  ?? GAZE_DEFAULT_MAX_ANGLE.leftEye,
      rightEye: config.maxEyeAngle  ?? GAZE_DEFAULT_MAX_ANGLE.rightEye,
    };
    const gazeBonesByKey = {};

    model.traverse(node => {
      if (node.isMesh) {
        // Skinned meshes cache their frustum-cull sphere from the first pose
        // they're tested in and never refresh it, so a clip that displaces the
        // avatar on spawn (a trimmed walk_in) leaves the sphere stranded and the
        // mesh culled forever. The avatar is always framed — just never cull it.
        node.frustumCulled = false;
      }
      if (node.isMesh && node.morphTargetDictionary && node.morphTargetInfluences) {
        morphMeshes.set(node, node.morphTargetDictionary);
      }
      if (node.name === jawBoneName) {
        jawBone = node;
        jawRestAngle = node.rotation[jawAxis] ?? 0;
      }
      for (const key of GAZE_BONE_KEYS) {
        if (gazeBoneNames[key] && node.name === gazeBoneNames[key]) {
          const bindWorldQuat = new THREE.Quaternion();
          node.getWorldQuaternion(bindWorldQuat);
          const worldPos = new THREE.Vector3();
          node.getWorldPosition(worldPos);
          const isElastic = key === 'neck' || key === 'head';
          gazeBonesByKey[key] = {
            bone: node,
            restLocalQuat: node.quaternion.clone(),
            bindWorldQuat,
            bindForwardWorld: camera.position.clone().sub(worldPos).normalize(),
            maxAngle: gazeMaxAngles[key],
            elastic:     isElastic,
            currentQuat: node.quaternion.clone(), // spring state; all bones need it
          };
        }
      }
    });
    _gazeChain = GAZE_BONE_KEYS.map(key => gazeBonesByKey[key]).filter(Boolean);
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

function _tickIdleTimer(delta) {
  if (_lifecycleState !== 'running' || _runSubstate !== 'idle' || _isTalking) return;
  _idleTimer += delta;
  if (_idleTimer < _idleToDanceSecs) return;
  _idleTimer = 0;
  if (_lonelyClips.length) {
    _runSubstate = 'lonely';
    channelMixer.play('lonely', _lonelyClips);
    channelMixer.stop('idle');
    if (!_startSystemTts(LONELY_MESSAGE)) _onLonelyMessageDone();
  } else if (_dancingClips.length) {
    _runSubstate = 'dancing';
    channelMixer.play('dancing', _dancingClips);
    channelMixer.stop('idle');
  }
}

function _tick() {
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);
  channelMixer?.cleanup();
  if (currentVrm) currentVrm.update(delta);
  _updateGazeTracking(delta);
  _tickIdleTimer(delta);
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
window.setIdleToDanceSecs = function (secs) {
  _idleToDanceSecs = (Number.isFinite(secs) && secs > 0) ? secs : 180;
};

window.setSynthesisCallback = function (fn) {
  _requestSynthesis = fn;
};

// Called by app.js for each system (non-pipeline) audio chunk.
window.setSystemLipSyncData = function (timeline, audio) {
  lipSyncDriver.setData(timeline, audio);
  _gazeSuspendDriver?.setTalking(!!audio);
  _isSystemTalking = !!audio;
  if (channelMixer) {
    if (audio && _talkingClips.length) {
      channelMixer.play('talking', _talkingClips);
    } else if (!audio) {
      _isEmotionAnimPlaying = false;
      channelMixer.stop('emotion_anim');
      channelMixer.stop('talking');
    }
  }
};

// Called by app.js when the system audio queue is fully drained.
window.systemAudioEnded = function () {
  _isSystemTalking = false;
  _systemTtsActive = false;
  lipSyncDriver.setData([], null);
  _gazeSuspendDriver?.setTalking(false);
  if (channelMixer) {
    _isEmotionAnimPlaying = false;
    channelMixer.stop('emotion_anim');
    channelMixer.stop('talking');
  }
  if (_greetingNeedsTransition) {
    _greetingNeedsTransition = false;
    _onGreetingDone();
  } else if (_farewellNeedsTransition) {
    _farewellNeedsTransition = false;
    _onFarewellDone();
  } else if (_lifecycleState === 'running' && _runSubstate === 'lonely') {
    _onLonelyMessageDone();
  }
};

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
