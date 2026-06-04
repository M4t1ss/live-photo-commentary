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
const postSpeechInput    = document.getElementById("cfg-post-speech-delay");
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
  if (ready && !running) setPhase("Idle", "none");
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

// --- Status / timer ---
let timerInterval = null;
let timerStart = null;
let timerEnd = null; // null = countup, timestamp = countdown target

function setPhase(label, mode = "up", durationMs = 0) {
  statusPhaseEl.textContent = label;
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
}

function enqueueChunk({ audio_url, text }) {
  audioQueue.push({ audio_url, text });
  if (!isPlaying) playNext();
}

function playNext() {
  if (audioQueue.length === 0) {
    isPlaying = false;
    currentAudio = null;
    if (ttsAllReceived) sendSpeechEnded();
    return;
  }
  isPlaying = true;
  const { audio_url, text } = audioQueue.shift();
  if (subtitlesVisible) subtitleEl.textContent = text;
  const audio = new Audio(`http://127.0.0.1:${port}${audio_url}`);
  audio.volume = volume;
  currentAudio = audio;
  audio.addEventListener("ended", playNext);
  audio.addEventListener("error", playNext);
  audio.play().catch(playNext);
}

function sendSpeechEnded() {
  ttsAllReceived = false;
  subtitleEl.textContent = "";
  const delayMs = (currentConfig.post_speech_delay ?? 2.0) * 1000;
  setPhase("Next in", "down", delayMs);
  send({ type: "speech_ended" });
}

// --- Controls ---
startStopBtn.addEventListener("click", () => {
  if (running) {
    setRunning(false);
    resetAudio();
    subtitleEl.textContent = "";
    setPhase("Idle", "none");
    send({ type: "stop_cycle" });
  } else {
    setRunning(true);
    send({ type: "start_cycle" });
  }
});

toggleSubtitleBtn.addEventListener("click", () => {
  subtitlesVisible = !subtitlesVisible;
  toggleSubtitleBtn.classList.toggle("active", subtitlesVisible);
  if (!subtitlesVisible) subtitleEl.textContent = "";
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

  postSpeechInput.value   = currentConfig.post_speech_delay ?? 2.0;
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
    post_speech_delay: parseFloat(postSpeechInput.value) || 2.0,
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

// --- Init ---
async function main() {
  setPhase("Idle", "none");
  port = await invoke("get_backend_port");
  connectWebSocket();

  await listen("backend_crashed", () => {
    setRunning(false);
    configReceived = false;
    modelsReceived = false;
    backendReady   = false;
    checkReady();
    setPhase("Backend crashed", "none");
  });
}

main();
