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

let mixer = null;
const clock = new THREE.Clock();

new GLTFLoader().load('./model.glb', (gltf) => {
  const model = gltf.scene;

  // Center horizontally, feet at y=0
  const box = new THREE.Box3().setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  model.position.set(-center.x, -box.min.y, -center.z);
  scene.add(model);

  // Frame camera to fit full character height
  const halfFov = (camera.fov / 2) * (Math.PI / 180);
  const dist = (size.y / 2) / Math.tan(halfFov) * 1.15;
  camera.position.set(0, size.y * 0.5, dist);
  camera.lookAt(0, size.y * 0.5, 0);

  if (gltf.animations.length > 0) {
    mixer = new THREE.AnimationMixer(model);
    mixer.clipAction(gltf.animations[0]).play();
  }
}, undefined, (err) => console.error('[avatar] model load failed', err));

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

(function frame() {
  requestAnimationFrame(frame);
  const delta = clock.getDelta();
  if (mixer) mixer.update(delta);
  renderer.render(scene, camera);
})();

// Stub — will drive emotion animations in a later stage
window.setEmotion = (_emotion) => {};
