const { invoke } = window.__TAURI__?.core ?? { invoke: () => Promise.resolve(0) };
const { listen }  = window.__TAURI__?.event ?? { listen: () => Promise.resolve(() => {}) };

// --- DOM refs ---
const startStopBtn       = document.getElementById("start-stop");
const settingsBtn        = document.getElementById("settings-btn");
const currentFrameEl     = document.getElementById("current-frame");
const prevFrameEl        = document.getElementById("prev-frame");
const subtitleEl         = document.getElementById("subtitle");
const toggleSubtitleBtn  = document.getElementById("toggle-subtitle");
const statusPhaseEl      = document.getElementById("status-phase");
const statusTimerEl      = document.getElementById("status-timer");
const modalOverlay       = document.getElementById("modal-overlay");
const modalCancel        = document.getElementById("modal-cancel");
const modalOk            = document.getElementById("modal-ok");
const vlmSelect          = document.getElementById("cfg-vlm");
const ttsVoiceInput      = document.getElementById("cfg-tts-voice");
const ttsVoiceDatalist   = document.getElementById("tts-voice-list");
const preScreenshotInput    = document.getElementById("cfg-pre-screenshot-delay");
const diffThreshInput    = document.getElementById("cfg-diff-threshold");
const diffMeasureSelect  = document.getElementById("cfg-diff-measure");
const maxHistoryInput    = document.getElementById("cfg-max-history");
const geminiKeyInput     = document.getElementById("cfg-gemini-key");
const openaiKeyInput     = document.getElementById("cfg-openai-key");
const elevenlabsKeyInput = document.getElementById("cfg-elevenlabs-key");
const volumeInput        = document.getElementById("cfg-volume");
const volumePctEl        = document.getElementById("cfg-volume-pct");
const barVolumeInput     = document.getElementById("bar-volume");
const barMuteBtn         = document.getElementById("bar-mute");
const cudaBanner         = document.getElementById("cuda-banner");
const cudaBannerMsg      = document.getElementById("cuda-banner-msg");
const cudaInstallBtn     = document.getElementById("cuda-install-btn");
const cudaDismissBtn     = document.getElementById("cuda-dismiss-btn");
const splashEl           = document.getElementById("splash");
const splashStatusEl     = document.getElementById("splash-status");

// --- App state ---
let volume = parseFloat(localStorage.getItem("lpc_volume") ?? "1");
let preMuteVolume = null;
barVolumeInput.value = volume;

let port = null;
let ws = null;
let running = false;
let subtitlesVisible = true;
let currentConfig = {};
let vlmCatalogue = [];
let ttsVoices = [];

let currentFrameUrl = null;

// All three must be true before the UI is usable.
let configReceived = false;
let modelsReceived = false;
let backendReady    = false;

function checkReady() {
  const ready = configReceived && modelsReceived && backendReady;
  startStopBtn.disabled = !ready;
  settingsBtn.disabled  = !ready;
  if (ready && !running) { setPhase("Idle", "none"); dismissSplash(); }
}

function setRunning(value) {
  running = value;
  if (running) {
    startStopBtn.innerHTML = "&#9208;";
    startStopBtn.title = "Stop";
    startStopBtn.classList.add("running");
  } else {
    startStopBtn.innerHTML = "&#9205;";
    startStopBtn.title = "Start";
    startStopBtn.classList.remove("running");
  }
}

// Audio queue state
let audioQueue = [];
let ttsAllReceived = false;
let isPlaying = false;
let currentAudio = null;
// True from first chunk of a batch until sendSpeechEnded, to gate "Synthesizing" phase
let firstChunkReceived = false;

let _startCycleTimer = null;

// Subtitle cycling state
let _subtitles = null;
let _subtitleIdx = 0;
let _subtitleTimer = null;

function _clearSubtitleTimer() {
  if (_subtitleTimer !== null) { clearInterval(_subtitleTimer); _subtitleTimer = null; }
}

function _startSubtitleTimer() {
  _clearSubtitleTimer();
  if (!_subtitles || _subtitles.length <= 1) return;
  _subtitleTimer = setInterval(() => {
    if (!currentAudio || !_subtitles) return;
    const t = currentAudio.currentTime;
    let idx = 0;
    for (let i = _subtitles.length - 1; i > 0; i--) {
      if (_subtitles[i].time <= t) { idx = i; break; }
    }
    if (idx !== _subtitleIdx) {
      _subtitleIdx = idx;
      if (subtitlesVisible) subtitleEl.textContent = _subtitles[idx].text;
    }
  }, 50);
}

// --- Status / timer ---
let timerInterval = null;
let timerStart = null;
let timerEnd = null; // null = countup, timestamp = countdown target

let splashDismissed = false;
function dismissSplash() {
  if (splashDismissed) return;
  splashDismissed = true;
  splashEl.classList.add("splash-gone");
}

function setPhase(label, mode = "up", durationMs = 0) {
  statusPhaseEl.textContent = label;
  splashStatusEl.textContent = label;
  clearInterval(timerInterval);

  if (mode === "none") {
    statusTimerEl.textContent = "";
    return;
  }

  timerStart = Date.now();
  timerEnd = mode === "down" ? timerStart + durationMs : null;

  timerInterval = setInterval(() => {
    const now = Date.now();
    if (timerEnd !== null) {
      const remaining = Math.max(0, timerEnd - now) / 1000;
      statusTimerEl.textContent = remaining.toFixed(1) + "s";
      if (remaining <= 0) {
        clearInterval(timerInterval);
        statusTimerEl.textContent = "";
      }
    } else {
      statusTimerEl.textContent = ((now - timerStart) / 1000).toFixed(1) + "s";
    }
  }, 100);
}

// --- WebSocket ---
let reconnectDelay = 500;
const RECONNECT_MAX = 30_000;

function send(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.debug("[ws ->]", data);
    ws.send(JSON.stringify(data));
  }
}

function connectWebSocket() {
  ws = new WebSocket(`ws://127.0.0.1:${port}/ws`);

  ws.addEventListener("open", () => {
    reconnectDelay = 500;
    configReceived = false;
    modelsReceived = false;
    backendReady   = false;
    setRunning(false);
    checkReady();
    setPhase("Initialising…", "up");
  });

  ws.addEventListener("message", (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }
    console.debug("[ws <-]", data);
    handleMessage(data);
  });

  ws.addEventListener("close", () => {
    ws = null;
    setTimeout(connectWebSocket, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX);
  });

  // Suppress browser console error; reconnect is driven by "close".
  ws.addEventListener("error", () => {});
}

// --- Message handler ---
function handleMessage(data) {
  switch (data.type) {
    case "frame": {
      const newUrl = `http://127.0.0.1:${port}${data.url}?t=${Date.now()}`;
      if (currentFrameUrl) {
        prevFrameEl.src = currentFrameUrl;
        prevFrameEl.classList.remove("hidden");
      }
      currentFrameUrl = newUrl;
      currentFrameEl.src = newUrl;
      firstChunkReceived = false;
      resetAudio();
      setPhase("Describing…", "up");
      break;
    }
    case "chunk":
      if (running) {
        if (!firstChunkReceived) {
          firstChunkReceived = true;
          setPhase("Synthesizing…", "up");
        }
        enqueueChunk(data);
      }
      break;

    case "tts_done":
      if (running) {
        ttsAllReceived = true;
        if (!isPlaying) sendSpeechEnded();
      }
      break;

    case "skipped":
      setPhase(`Skipped  Δ=${data.diff}`, "none");
      break;

    case "busy":
      setPhase("Busy", "none");
      break;

    case "error":
      setPhase(`Error: ${data.message}`, "none");
      console.error("[backend]", data.message);
      break;

    case "ready_state":
      backendReady = data.ready;
      checkReady();
      break;

    case "model_loading":
      if (!running) setPhase("Initialising…", "up");
      break;

    case "load_progress":
      clearInterval(timerInterval);
      timerInterval = null;
      statusPhaseEl.textContent = data.message;
      statusTimerEl.textContent = "";
      break;

    case "download_progress": {
      // Stop the countup timer so it doesn't overwrite these values.
      clearInterval(timerInterval);
      timerInterval = null;
      const pct = Math.round(data.downloaded / data.total * 100);
      statusPhaseEl.textContent = `Downloading ${data.file}`;
      if (data.total >= 1_000_000) {
        const mb = (data.downloaded / 1e6).toFixed(0);
        const totalMb = (data.total / 1e6).toFixed(0);
        statusTimerEl.textContent = `${pct}%  ${mb}/${totalMb} MB`;
      } else {
        // Small total = file count bar, not bytes.
        statusTimerEl.textContent = `${pct}%`;
      }
      break;
    }

    case "model_ready":
      break;

    case "config":
      currentConfig = data.data ?? {};
      configReceived = true;
      checkReady();
      break;

    case "models":
      vlmCatalogue = data.vlm ?? [];
      ttsVoices = (data.tts ?? []).map((t) => t.voice);
      modelsReceived = true;
      checkReady();
      break;
  }
}

// --- Audio queue ---
function resetAudio() {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = "";
    currentAudio = null;
  }
  audioQueue = [];
  ttsAllReceived = false;
  isPlaying = false;
  _clearSubtitleTimer();
  _subtitles = null;
  _subtitleIdx = 0;
  window.setLipSyncData?.([], null);
}

function enqueueChunk({ audio_url, text, phonemes, subtitles }) {
  audioQueue.push({
    audio_url,
    text,
    timeline: buildTimeline(phonemes ?? []),
    subtitles: subtitles ?? [{ text, time: 0 }],
  });
  if (!isPlaying) playNext();
}

function playNext() {
  _clearSubtitleTimer();
  if (audioQueue.length === 0) {
    isPlaying = false;
    currentAudio = null;
    _subtitles = null;
    _subtitleIdx = 0;
    window.setLipSyncData?.([], null);
    if (ttsAllReceived) sendSpeechEnded();
    return;
  }
  isPlaying = true;
  const { audio_url, text, timeline, subtitles } = audioQueue.shift();
  _subtitles = subtitles;
  _subtitleIdx = 0;
  if (subtitlesVisible) subtitleEl.textContent = subtitles[0].text;
  const audio = new Audio(`http://127.0.0.1:${port}${audio_url}`);
  audio.volume = volume;
  currentAudio = audio;
  window.setLipSyncData?.(timeline, audio);
  _startSubtitleTimer();
  audio.addEventListener("ended", playNext);
  audio.addEventListener("error", playNext);
  audio.play().catch(playNext);
}

function sendSpeechEnded() {
  ttsAllReceived = false;
  subtitleEl.textContent = "";
  const delayMs = (currentConfig.pre_screenshot_delay ?? 2.0) * 1000;
  setPhase("Screenshot in", "down", delayMs);
  setTimeout(() => {
    if (running) send({ type: "take_screenshot" });
  }, delayMs);
}

// --- Controls ---
startStopBtn.addEventListener("click", () => {
  if (running) {
    const pendingStart = _startCycleTimer !== null;
    if (pendingStart) {
      clearTimeout(_startCycleTimer);
      _startCycleTimer = null;
    }
    setRunning(false);
    resetAudio();
    subtitleEl.textContent = "";
    setPhase("Idle", "none");
    if (!pendingStart) send({ type: "stop_cycle" });
  } else {
    setRunning(true);
    const delayMs = (currentConfig.pre_screenshot_delay ?? 2.0) * 1000;
    setPhase("Screenshot in", "down", delayMs);
    _startCycleTimer = setTimeout(() => {
      _startCycleTimer = null;
      if (running) send({ type: "start_cycle" });
    }, delayMs);
  }
});

toggleSubtitleBtn.addEventListener("click", () => {
  subtitlesVisible = !subtitlesVisible;
  toggleSubtitleBtn.classList.toggle("active", subtitlesVisible);
  if (!subtitlesVisible) subtitleEl.textContent = "";
  else if (_subtitles) subtitleEl.textContent = _subtitles[_subtitleIdx].text;
});

// --- Settings modal ---
settingsBtn.addEventListener("click", openModal);
modalCancel.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

function openModal() {
  populateModal();
  modalOverlay.classList.remove("hidden");
}

function closeModal() {
  modalOverlay.classList.add("hidden");
}

function populateModal() {
  vlmSelect.innerHTML = "";
  let vlmMatched = false;
  for (const entry of vlmCatalogue) {
    const opt = document.createElement("option");
    opt.value = `${entry.provider}|${entry.model_id}`;
    opt.textContent = `${capitalize(entry.provider)}: ${entry.model_id}`;
    if (!vlmMatched) {
      const providerMatch = currentConfig.vlm_provider === entry.provider;
      const modelMatch = !currentConfig.vlm_model || currentConfig.vlm_model === entry.model_id;
      if (providerMatch && modelMatch) { opt.selected = true; vlmMatched = true; }
    }
    vlmSelect.appendChild(opt);
  }
  if (!vlmMatched && vlmSelect.options.length > 0) vlmSelect.options[0].selected = true;

  ttsVoiceDatalist.innerHTML = "";
  for (const voice of ttsVoices) {
    const opt = document.createElement("option");
    opt.value = voice;
    ttsVoiceDatalist.appendChild(opt);
  }
  ttsVoiceInput.value = currentConfig.tts_voice ?? "af_heart";

  volumeInput.value = volume;
  volumePctEl.textContent = Math.round(volume * 100) + "%";

  preScreenshotInput.value   = currentConfig.pre_screenshot_delay ?? 2.0;
  diffThreshInput.value   = currentConfig.difference_threshold ?? 0.0;
  diffMeasureSelect.value = currentConfig.difference_measure ?? "mse";
  maxHistoryInput.value   = currentConfig.max_history_size ?? 0;
  geminiKeyInput.value    = "";
  openaiKeyInput.value    = "";
  elevenlabsKeyInput.value = "";
}

function setVolume(v, save = false) {
  volume = v;
  barVolumeInput.value = v;
  volumeInput.value = v;
  volumePctEl.textContent = Math.round(v * 100) + "%";
  barMuteBtn.textContent = v <= 0 ? "🔇" : "🔊";
  if (currentAudio) currentAudio.volume = v;
  if (save) localStorage.setItem("lpc_volume", v);
}

volumeInput.addEventListener("input", () => {
  preMuteVolume = null;
  setVolume(parseFloat(volumeInput.value));
});

barVolumeInput.addEventListener("input", () => {
  preMuteVolume = null;
  setVolume(parseFloat(barVolumeInput.value));
});

barMuteBtn.addEventListener("click", () => {
  if (preMuteVolume !== null) {
    const v = preMuteVolume;
    preMuteVolume = null;
    setVolume(v);
  } else if (volume > 0) {
    preMuteVolume = volume;
    setVolume(0);
  }
});

modalOk.addEventListener("click", () => {
  const [provider, model_id] = (vlmSelect.value || "").split("|");
  const updates = {
    vlm_provider: provider || "gemini",
    vlm_model: model_id || null,
    tts_voice: ttsVoiceInput.value.trim() || "af_heart",
    pre_screenshot_delay: parseFloat(preScreenshotInput.value) || 2.0,
    difference_threshold: parseFloat(diffThreshInput.value) || 0.0,
    difference_measure: diffMeasureSelect.value || "mse",
    max_history_size: parseInt(maxHistoryInput.value, 10) || 0,
  };
  if (geminiKeyInput.value)     updates.gemini_api_key     = geminiKeyInput.value;
  if (openaiKeyInput.value)     updates.openai_api_key     = openaiKeyInput.value;
  if (elevenlabsKeyInput.value) updates.elevenlabs_api_key = elevenlabsKeyInput.value;

  setVolume(parseFloat(volumeInput.value), true);

  send({ type: "set_config", data: updates, persist: true });
  closeModal();
});

function capitalize(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

// --- CUDA upgrade banner ---
let pendingCuIndex = null;

cudaInstallBtn.addEventListener("click", () => {
  if (!pendingCuIndex) return;
  cudaInstallBtn.disabled = true;
  cudaDismissBtn.disabled = true;
  invoke("install_cuda_torch", { cuIndex: pendingCuIndex });
});

cudaDismissBtn.addEventListener("click", () => {
  cudaBanner.classList.add("hidden");
});

// --- Init ---
async function main() {
  setPhase("Starting up…", "up");

  // Register early so splash status updates during uv sync / backend setup,
  // and so backend_crashed / cuda_upgrade_available are never missed while
  // invoke is still pending.
  await listen("setup_progress", ({ payload }) => setPhase(payload, "up"));
  await listen("backend_crashed", () => {
    dismissSplash();
    setRunning(false);
    configReceived = false;
    modelsReceived = false;
    backendReady   = false;
    checkReady();
    setPhase("Backend crashed", "none");
  });

  // cuda_upgrade_available fires right after uv sync (before uvicorn even
  // starts), so it must be registered here alongside the other early listeners.
  await listen("cuda_upgrade_available", ({ payload }) => {
    pendingCuIndex = payload;
    cudaBannerMsg.textContent =
      `NVIDIA GPU detected (${payload}). Install CUDA-optimised PyTorch for faster inference?`;
    cudaBanner.classList.remove("hidden");
  });
  await listen("cuda_install_progress", ({ payload }) => {
    cudaBannerMsg.textContent = payload;
  });
  await listen("cuda_install_done", () => {
    cudaBannerMsg.textContent = "CUDA PyTorch installed. Restart the app to use GPU acceleration.";
    cudaInstallBtn.textContent = "Restart";
    cudaInstallBtn.disabled = false;
    cudaInstallBtn.onclick = () => invoke("restart_app");
    cudaDismissBtn.classList.add("hidden");
  });
  await listen("cuda_install_failed", ({ payload }) => {
    cudaBannerMsg.textContent = `Installation failed: ${payload}`;
    cudaInstallBtn.disabled = false;
    cudaDismissBtn.disabled = false;
  });

  // Fetch model config and init avatar only once the backend is confirmed
  // ready. Registering before invoke("get_backend_port") ensures the event
  // is never missed even when the backend starts very quickly.
  const unlistenReady = await listen("backend_ready", async () => {
    unlistenReady();
    const modelCfg = await fetch(`http://127.0.0.1:${port}/model-config`).then(r => r.json()).catch(() => ({}));
    window.initLipSync?.(modelCfg);
    window.initAvatar?.(modelCfg, port);
  });

  port = await invoke("get_backend_port");
  connectWebSocket();
}

main();
