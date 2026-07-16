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
const splashRestartBtn   = document.getElementById("splash-restart-btn");
const leftPanelEl        = document.getElementById("left-panel");
const subtitlePanelEl    = document.getElementById("subtitle-panel");
const subtitleRowDivider = document.getElementById("subtitle-row-divider");
const cfgSubtitleMode    = document.getElementById("cfg-subtitle-mode");
const cfgBgColor         = document.getElementById("cfg-bg-color");
const cfgBgPreview       = document.getElementById("cfg-bg-preview");
const cfgFgColor         = document.getElementById("cfg-fg-color");
const cfgFgPreview       = document.getElementById("cfg-fg-preview");
const cfgPromptset       = document.getElementById("cfg-promptset");
const promptsetComboBtn  = document.getElementById("promptset-combo-btn");
const promptsetDropdown  = document.getElementById("promptset-dropdown");
const promptsetLoad      = document.getElementById("promptset-load");
const promptsetSave      = document.getElementById("promptset-save");
const promptsetDelete    = document.getElementById("promptset-delete");
const cfgSystemPrompt    = document.getElementById("cfg-system-prompt");
const cfgPromptText      = document.getElementById("cfg-prompt");
const cfgFirstPrompt     = document.getElementById("cfg-first-prompt");
const cfgHistoryPrompt   = document.getElementById("cfg-history-prompt");
const cfgCompactPrompt   = document.getElementById("cfg-compact-prompt");

// --- App state ---
let volume = parseFloat(localStorage.getItem("lpc_volume") ?? "1");
let preMuteVolume = null;
barVolumeInput.value = volume;

let port = null;
let ws = null;
let running = false;
let subtitlesVisible = true;
let subtitleMode = localStorage.getItem("lpc_subtitle_mode") ?? "overlay";
let bgColor = localStorage.getItem("lpc_bg_color") ?? "#000000";
let fgColor = localStorage.getItem("lpc_fg_color") ?? "#ffffff";
let currentConfig = {};
let vlmCatalogue = [];
let ttsVoices = [];
let promptsetNames = [];
let currentPromptsetName = "default";

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

// Emotion tag scheduling state (tags persist across chunks of one speech act)
let _tags = [];
let _tagIdx = 0;

function setSubtitleText(text) {
  subtitleEl.textContent = text;
  subtitlePanelEl.textContent = text;
}

function _clearSubtitleTimer() {
  if (_subtitleTimer !== null) { clearInterval(_subtitleTimer); _subtitleTimer = null; }
}

function _startSubtitleTimer() {
  _clearSubtitleTimer();
  if ((!_subtitles || _subtitles.length <= 1) && _tags.length === 0) return;
  _subtitleTimer = setInterval(() => {
    if (!currentAudio) return;
    const t = currentAudio.currentTime;
    if (_subtitles) {
      let idx = 0;
      for (let i = _subtitles.length - 1; i > 0; i--) {
        if (_subtitles[i].time <= t) { idx = i; break; }
      }
      if (idx !== _subtitleIdx) {
        _subtitleIdx = idx;
        if (subtitlesVisible) setSubtitleText(_subtitles[idx].text);
      }
    }
    // Emotion tags hold until replaced, so just apply each in order as its time passes.
    while (_tagIdx < _tags.length && _tags[_tagIdx].time <= t) {
      window.setEmotion?.(_tags[_tagIdx].name);
      _tagIdx++;
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
function showSplash() {
  splashDismissed = false;
  splashEl.classList.remove("splash-gone");
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
    setPhase("Initializing…", "up");
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

    case "take_screenshot":
      if (running) takeScreenshot();
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
      if (!running) setPhase("Initializing…", "up");
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

    case "promptset_loaded":
      currentPromptsetName = data.name;
      if (data.names) { promptsetNames = data.names; }
      cfgPromptset.value     = data.name;
      cfgSystemPrompt.value  = data.fields?.system_prompt ?? "";
      cfgPromptText.value    = data.fields?.prompt ?? "";
      cfgFirstPrompt.value   = data.fields?.first_prompt ?? "";
      cfgHistoryPrompt.value = data.fields?.history_prompt ?? "";
      cfgCompactPrompt.value = data.fields?.compact_prompt ?? "";
      _updatePromptsetBtns();
      break;

    case "promptsets":
      promptsetNames = data.names ?? [];
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
  _tags = [];
  _tagIdx = 0;
  window.setLipSyncData?.([], null);
  window.setEmotion?.(null);
}

function enqueueChunk({ audio_url, text, phonemes, subtitles, tags }) {
  audioQueue.push({
    audio_url,
    text,
    timeline: buildTimeline(phonemes ?? []),
    subtitles: subtitles ?? [{ text, time: 0 }],
    tags: tags ?? [],
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
    _tags = [];
    _tagIdx = 0;
    window.setLipSyncData?.([], null);
    window.setEmotion?.(null);
    if (ttsAllReceived) sendSpeechEnded();
    return;
  }
  isPlaying = true;
  const { audio_url, text, timeline, subtitles, tags } = audioQueue.shift();
  _subtitles = subtitles;
  _subtitleIdx = 0;
  _tags = tags;
  _tagIdx = 0;
  if (subtitlesVisible) setSubtitleText(subtitles[0].text);
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
  setSubtitleText("");
  const delayMs = (currentConfig.pre_screenshot_delay ?? 2.0) * 1000;
  setPhase("Screenshot in", "down", delayMs);
  setTimeout(() => {
    if (running) takeScreenshot();
  }, delayMs);
}

async function takeScreenshot() {
  try {
    const path = await invoke("take_screenshot");
    send({ type: "frame_ready", path });
  } catch (e) {
    console.error("[screenshot] take_screenshot failed:", e);
    setPhase(`Error: screenshot failed: ${e}`, "none");
  }
}

// --- Controls ---
function stopCycle() {
  if (!running) return;
  const pendingStart = _startCycleTimer !== null;
  if (pendingStart) {
    clearTimeout(_startCycleTimer);
    _startCycleTimer = null;
  }
  setRunning(false);
  resetAudio();
  setSubtitleText("");
  setPhase("Idle", "none");
  if (!pendingStart) send({ type: "stop_cycle" });
}

startStopBtn.addEventListener("click", () => {
  if (running) {
    stopCycle();
  } else {
    setRunning(true);
    const delayMs = (currentConfig.pre_screenshot_delay ?? 2.0) * 1000;
    setPhase("Screenshot in", "down", delayMs);
    _startCycleTimer = setTimeout(() => {
      _startCycleTimer = null;
      if (!running) return;
      send({ type: "start_cycle" });
      takeScreenshot();
    }, delayMs);
  }
});

toggleSubtitleBtn.addEventListener("click", () => {
  subtitlesVisible = !subtitlesVisible;
  toggleSubtitleBtn.classList.toggle("active", subtitlesVisible);
  if (!subtitlesVisible) setSubtitleText("");
  else if (_subtitles) setSubtitleText(_subtitles[_subtitleIdx].text);
});

// --- Settings modal ---
settingsBtn.addEventListener("click", openModal);
modalCancel.addEventListener("click", closeModal);
modalOverlay.addEventListener("click", (e) => { if (e.target === modalOverlay) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
    document.getElementById("tab-general").classList.toggle("hidden", btn.dataset.tab !== "general");
    document.getElementById("tab-prompts").classList.toggle("hidden", btn.dataset.tab !== "prompts");
  });
});

function openModal() {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === "general"));
  document.getElementById("tab-general").classList.remove("hidden");
  document.getElementById("tab-prompts").classList.add("hidden");
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
  cfgSubtitleMode.checked = subtitleMode === "separated";
  cfgPromptset.value = currentPromptsetName;
  _updatePromptsetBtns();
  cfgBgColor.value = bgColor;
  cfgBgPreview.style.background = bgColor;
  cfgFgColor.value = fgColor;
  cfgFgPreview.style.background = fgColor;
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

  // Changing the VLM or TTS model tears down and rebuilds it on the backend,
  // so a running cycle would hit a "no model configured" error mid-flight.
  // Stop first (full UI reset included) so the change lands on an idle app.
  const modelChanged =
    updates.vlm_provider !== currentConfig.vlm_provider ||
    updates.vlm_model !== currentConfig.vlm_model ||
    updates.tts_voice !== currentConfig.tts_voice;
  if (modelChanged) stopCycle();

  setVolume(parseFloat(volumeInput.value), true);

  const newMode = cfgSubtitleMode.checked ? "separated" : "overlay";
  if (newMode !== subtitleMode) {
    applySubtitleMode(newMode);
    localStorage.setItem("lpc_subtitle_mode", newMode);
  }

  const newBg = _normalizeHex(cfgBgColor.value);
  const newFg = _normalizeHex(cfgFgColor.value);
  if (isValidHex(newBg) && isValidHex(newFg)) {
    applyColors(newBg, newFg);
    localStorage.setItem("lpc_bg_color", newBg);
    localStorage.setItem("lpc_fg_color", newFg);
  }

  send({ type: "set_config", data: updates, persist: true });
  closeModal();
});

function capitalize(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

// --- Promptset combobox ---
function _openPromptsetDropdown() {
  const current = cfgPromptset.value.trim();
  promptsetDropdown.innerHTML = "";
  for (const name of promptsetNames) {
    const li = document.createElement("li");
    li.textContent = name;
    if (name === current) li.classList.add("combo-selected");
    li.addEventListener("mousedown", (e) => {
      e.preventDefault(); // keep focus on input
      cfgPromptset.value = name;
      _closePromptsetDropdown();
      _updatePromptsetBtns();
    });
    promptsetDropdown.appendChild(li);
  }
  promptsetDropdown.classList.remove("hidden");
}

function _closePromptsetDropdown() {
  promptsetDropdown.classList.add("hidden");
}

promptsetComboBtn.addEventListener("click", () => {
  if (promptsetDropdown.classList.contains("hidden")) {
    _openPromptsetDropdown();
    cfgPromptset.focus();
  } else {
    _closePromptsetDropdown();
  }
});

document.addEventListener("mousedown", (e) => {
  if (!e.target.closest("#promptset-combo")) _closePromptsetDropdown();
});

function _updatePromptsetBtns() {
  const isDefault = cfgPromptset.value.trim() === "default";
  promptsetSave.disabled   = isDefault;
  promptsetDelete.disabled = isDefault;
}

cfgPromptset.addEventListener("input", _updatePromptsetBtns);

promptsetLoad.addEventListener("click", () => {
  const name = cfgPromptset.value.trim() || "default";
  send({ type: "load_promptset", name });
});

promptsetSave.addEventListener("click", () => {
  const name = cfgPromptset.value.trim();
  if (!name || name === "default") return;
  send({
    type: "save_promptset",
    name,
    fields: {
      system_prompt:  cfgSystemPrompt.value,
      prompt:         cfgPromptText.value,
      first_prompt:   cfgFirstPrompt.value,
      history_prompt: cfgHistoryPrompt.value,
      compact_prompt: cfgCompactPrompt.value,
    },
  });
});

promptsetDelete.addEventListener("click", () => {
  const name = cfgPromptset.value.trim();
  if (!name || name === "default") return;
  send({ type: "delete_promptset", name });
});

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

splashRestartBtn.addEventListener("click", () => {
  splashRestartBtn.classList.add("hidden");
  setPhase("Restarting…", "up");
  invoke("restart_backend");
});

// --- Init ---
async function main() {
  setPhase("Starting up…", "up");

  // Register early so splash status updates during uv sync / backend setup,
  // and so backend_crashed / cuda_upgrade_available are never missed while
  // invoke is still pending.
  await listen("setup_progress", ({ payload }) => setPhase(payload, "up"));
  await listen("backend_crashed", ({ payload }) => {
    setRunning(false);
    configReceived = false;
    modelsReceived = false;
    backendReady   = false;
    checkReady();
    showSplash();
    setPhase(payload || "Backend crashed", "none");
    splashRestartBtn.classList.remove("hidden");
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
    cudaBannerMsg.textContent = "CUDA PyTorch installed. Restart the backend to use GPU acceleration.";
    cudaInstallBtn.textContent = "Restart";
    cudaInstallBtn.disabled = false;
    cudaInstallBtn.onclick = () => { stopCycle(); cudaBanner.classList.add("hidden"); showSplash(); invoke("restart_backend"); };
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

// --- Appearance colors ---
function isValidHex(s) { return /^#[0-9a-fA-F]{6}$/.test(s); }

function applyColors(bg, fg) {
  bgColor = bg;
  fgColor = fg;
  document.documentElement.style.setProperty("--lpc-bg", bg);
  document.documentElement.style.setProperty("--lpc-fg", fg);
}

applyColors(bgColor, fgColor);

function _normalizeHex(s) {
  s = s.trim();
  if (/^[0-9a-fA-F]{6}$/.test(s)) s = "#" + s;
  return s;
}

cfgBgColor.addEventListener("input", () => {
  const v = _normalizeHex(cfgBgColor.value);
  if (isValidHex(v)) cfgBgPreview.style.background = v;
});
cfgFgColor.addEventListener("input", () => {
  const v = _normalizeHex(cfgFgColor.value);
  if (isValidHex(v)) cfgFgPreview.style.background = v;
});

// --- Subtitle row divider ---
const SUBTITLE_HEIGHT_KEY = "lpc_subtitle_height";
const DEFAULT_SUBTITLE_HEIGHT = 80;
const MIN_SUBTITLE_HEIGHT = 40;

let subtitlePanelHeight = parseInt(localStorage.getItem(SUBTITLE_HEIGHT_KEY) ?? String(DEFAULT_SUBTITLE_HEIGHT), 10);

function applySubtitleHeight(h) {
  subtitlePanelEl.style.height = h + "px";
}

// --- Subtitle mode ---
function applySubtitleMode(mode) {
  subtitleMode = mode;
  if (mode === "separated") {
    leftPanelEl.classList.add("subtitle-separated");
    applySubtitleHeight(subtitlePanelHeight);
  } else {
    leftPanelEl.classList.remove("subtitle-separated");
  }
}

applySubtitleMode(subtitleMode);

subtitleRowDivider.addEventListener("mousedown", (e) => {
  e.preventDefault();
  const startY = e.clientY;
  const startH = subtitlePanelEl.offsetHeight;

  subtitleRowDivider.classList.add("dragging");

  function onMove(e) {
    const maxH = leftPanelEl.offsetHeight * 0.5;
    const newH = Math.max(MIN_SUBTITLE_HEIGHT, Math.min(maxH, startH - (e.clientY - startY)));
    subtitlePanelHeight = newH;
    applySubtitleHeight(newH);
  }

  function onUp() {
    subtitleRowDivider.classList.remove("dragging");
    localStorage.setItem(SUBTITLE_HEIGHT_KEY, String(subtitlePanelHeight));
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  }

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
});

// --- Split pane ---
const SPLIT_KEY = 'lpc_split_ratio';
const DEFAULT_RATIO = 0.75;
const MIN_RATIO = 0.2;
const MAX_RATIO = 0.85;

const mainEl     = document.getElementById('main');
const leftPanel  = document.getElementById('left-panel');
const rightPanel = document.getElementById('right-panel');
const dividerEl  = document.getElementById('divider');

function applyPanelSplit(ratio) {
  const available = mainEl.offsetWidth - dividerEl.offsetWidth;
  leftPanel.style.flex  = `0 0 ${ratio * available}px`;
  rightPanel.style.flex = `0 0 ${(1 - ratio) * available}px`;
}

let splitRatio = parseFloat(localStorage.getItem(SPLIT_KEY) ?? String(DEFAULT_RATIO));
applyPanelSplit(splitRatio);
window.addEventListener('resize', () => applyPanelSplit(splitRatio));

dividerEl.addEventListener('mousedown', (e) => {
  e.preventDefault();
  const startX    = e.clientX;
  const startLeft = leftPanel.offsetWidth;
  const available = mainEl.offsetWidth - dividerEl.offsetWidth;

  dividerEl.classList.add('dragging');

  function onMove(e) {
    const newLeft = startLeft + (e.clientX - startX);
    splitRatio = Math.max(MIN_RATIO, Math.min(MAX_RATIO, newLeft / available));
    applyPanelSplit(splitRatio);
  }

  function onUp() {
    dividerEl.classList.remove('dragging');
    localStorage.setItem(SPLIT_KEY, String(splitRatio));
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
  }

  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
});
