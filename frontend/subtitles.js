// Subtitle rendering: per-word spans, current-word highlight, and client-side
// line fitting. The backend sends each chunk's clean text plus a `words` array
// of { s, cs, ce, ts, te } (surface, chunk-local char offsets, start/end time in
// seconds from the chunk's audio start). This module owns the two subtitle
// sinks (#subtitle overlay + #subtitle-panel) and is driven by app.js:
//   setSubtitleChunk(words, text)  when a chunk starts playing
//   tickSubtitle(currentTime)      every audio poll (~50 ms)
//   clearSubtitle()                on stop / speech end
//   setSubtitlesVisible(bool)      CC toggle
//   setSubtitleMode("overlay"|"separated")
//   applySubtitleAppearance()      after a settings change / on load

(() => {
  const overlayEl = document.getElementById("subtitle");
  const panelEl = document.getElementById("subtitle-panel");
  const SENT_END = /[.?!…。！？]$/;
  const CLAUSE_END = /[,;:、；：]$/;

  // Inset (px) kept at both ends of the vertical-position slider: vpos 0 sits
  // this far below the top edge, vpos 100 this far above the bottom edge, vpos
  // 50 is always dead centre. Set to 0 for the slider extremes to reach the
  // edges flush.
  const SUBTITLE_VPOS_MARGIN_PX = 8;

  let mode = "overlay";
  let visible = true;
  let chunk = null;          // { text, words } or null
  let segments = [];         // [{ lo, hi }] word-index ranges (K>0), else []
  let curWord = -1;
  let settings = readSettings();

  function readSettings() {
    const g = (k, d) => localStorage.getItem(k) ?? d;
    let lines = parseInt(g("lpc_subtitle_lines", "2"), 10);
    if (!Number.isFinite(lines) || lines < 0) lines = 2;
    let fontPx = parseInt(g("lpc_subtitle_font_px", "17"), 10);
    if (!Number.isFinite(fontPx) || fontPx < 8) fontPx = 17;
    let vpos = parseInt(g("lpc_subtitle_vpos", "100"), 10);
    if (!Number.isFinite(vpos)) vpos = 100;
    vpos = Math.min(100, Math.max(0, vpos));
    return {
      vpos,
      fontPx,
      lines,
      highlightEnabled: g("lpc_highlight_enabled", "1") === "1",
      highlightColor: g("lpc_highlight_color", "#4a9eff"),
    };
  }

  // vpos: 0 = top, 50 = centred, 100 = bottom (inset by SUBTITLE_VPOS_MARGIN_PX
  // at both extremes). Anchors the overlay by interpolating its own height, so
  // it never spills past the avatar area.
  function positionOverlay(vpos) {
    const m = SUBTITLE_VPOS_MARGIN_PX;
    overlayEl.style.bottom = "auto";
    overlayEl.style.top = `calc(${m}px + (100% - ${2 * m}px) * ${vpos / 100})`;
    overlayEl.style.transform = `translateY(-${vpos}%)`;
  }

  function activeEl() {
    return mode === "separated" ? panelEl : overlayEl;
  }

  function sepBetween(words, i) {
    if (!chunk || i <= 0) return "";
    return chunk.text.slice(words[i - 1].ce, words[i].cs);
  }

  function renderSpansInto(container) {
    container.textContent = "";
    if (!chunk) return;
    const { words, text } = chunk;
    if (!words || !words.length) {
      container.textContent = text;
      return;
    }
    const frag = document.createDocumentFragment();
    for (let i = 0; i < words.length; i++) {
      const sep = sepBetween(words, i);
      if (sep) frag.appendChild(document.createTextNode(sep));
      const span = document.createElement("span");
      span.className = "lpc-word";
      span.dataset.wi = String(i);
      span.textContent = words[i].s;
      frag.appendChild(span);
    }
    container.appendChild(frag);
  }

  function render() {
    renderSpansInto(overlayEl);
    renderSpansInto(panelEl);
  }

  function median(nums) {
    if (!nums.length) return 0;
    const s = [...nums].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  // Lay the remaining words out in an offscreen clone of the visible container,
  // find where line K+1 begins, then back up to the nearest linguistic break.
  function computeSegments() {
    const words = chunk?.words;
    if (!words || !words.length) return [];
    const K = settings.lines;
    if (K <= 0) return [];

    const host = activeEl();
    const cs = getComputedStyle(host);
    const probe = document.createElement("div");
    for (const p of ["fontFamily", "fontSize", "fontWeight", "fontStyle",
      "letterSpacing", "lineHeight", "wordSpacing", "whiteSpace",
      "textAlign", "paddingLeft", "paddingRight", "textIndent"]) {
      probe.style[p] = cs[p];
    }
    probe.style.position = "absolute";
    probe.style.visibility = "hidden";
    probe.style.left = "-99999px";
    probe.style.top = "0";
    probe.style.boxSizing = "border-box";
    probe.style.width = (host.clientWidth || 400) + "px";
    document.body.appendChild(probe);

    const lineTol = (parseFloat(cs.lineHeight) || settings.fontPx * 1.4) * 0.5;
    const segs = [];
    try {
      let start = 0;
      let guard = 0;
      while (start < words.length && guard++ < 1000) {
        probe.textContent = "";
        const spans = [];
        for (let i = start; i < words.length; i++) {
          const sep = sepBetween(words, i);
          if (i > start && sep) probe.appendChild(document.createTextNode(sep));
          const sp = document.createElement("span");
          sp.textContent = words[i].s;
          probe.appendChild(sp);
          spans.push(sp);
        }
        const lineStarts = [];
        let lastTop = -1e9;
        for (let i = 0; i < spans.length; i++) {
          const top = spans[i].offsetTop;
          if (top - lastTop > lineTol) { lineStarts.push(i); lastTop = top; }
        }
        if (lineStarts.length <= K) {
          segs.push({ lo: start, hi: words.length });
          break;
        }
        const overflow = lineStarts[K];                 // first span on line K+1
        const lineKStart = lineStarts[K - 1];
        const floor = Math.floor((lineKStart + overflow) / 2);

        const gaps = [];
        for (let rel = 1; rel < overflow; rel++) {
          gaps.push(words[start + rel].ts - words[start + rel - 1].te);
        }
        const gapThresh = Math.max(0.15, 2.5 * (median(gaps.filter((g) => g > 0)) || 0.06));

        let cut = -1;
        let bestScore = 0;
        for (let rel = overflow - 1; rel >= floor && rel > 0; rel--) {
          const prev = words[start + rel - 1].s;
          let score = 0;
          if (SENT_END.test(prev)) score = 4;
          else if (CLAUSE_END.test(prev)) score = 3;
          else if (words[start + rel].ts - words[start + rel - 1].te > gapThresh) score = 2;
          if (score > bestScore) { bestScore = score; cut = rel; }
          if (bestScore === 4) break;
        }
        if (cut < 1) cut = Math.max(1, overflow);
        segs.push({ lo: start, hi: start + cut });
        start += cut;
      }
    } catch (e) {
      probe.remove();
      return [{ lo: 0, hi: words.length }];
    }
    probe.remove();
    return segs.length ? segs : [{ lo: 0, hi: words.length }];
  }

  function segIndexForWord(wi) {
    for (let i = 0; i < segments.length; i++) {
      if (wi >= segments[i].lo && wi < segments[i].hi) return i;
    }
    return segments.length ? segments.length - 1 : -1;
  }

  function applyVisibility(activeWord) {
    const words = chunk?.words;
    if (!words || !words.length) return;

    let lo = 0;
    let hi = words.length;
    if (settings.lines <= 0) {
      lo = activeWord < 0 ? 0 : activeWord;
      hi = lo + 1;
    } else if (segments.length) {
      const si = segIndexForWord(activeWord < 0 ? 0 : activeWord);
      if (si >= 0) { lo = segments[si].lo; hi = segments[si].hi; }
    }

    for (const container of [overlayEl, panelEl]) {
      const spans = container.querySelectorAll(".lpc-word");
      for (const span of spans) {
        const wi = Number(span.dataset.wi);
        span.classList.toggle("lpc-word-hidden", !(wi >= lo && wi < hi));
        span.classList.toggle(
          "lpc-word-active",
          settings.highlightEnabled && wi === activeWord,
        );
      }
    }
  }

  function wordAt(t) {
    const words = chunk?.words;
    if (!words || !words.length) return -1;
    let idx = 0;
    for (let i = 0; i < words.length; i++) {
      if (words[i].ts <= t) idx = i;
      else break;
    }
    return idx;
  }

  // ── public API ─────────────────────────────────────────────────────────────

  window.setSubtitleChunk = function (words, text) {
    chunk = { words: words || null, text: text || "" };
    curWord = -1;
    render();
    segments = (words && words.length && settings.lines > 0) ? computeSegments() : [];
    applyVisibility(-1);
  };

  window.tickSubtitle = function (t) {
    if (!chunk || !visible) return;
    const wi = wordAt(t);
    if (wi === curWord) return;
    curWord = wi;
    applyVisibility(wi);
  };

  window.clearSubtitle = function () {
    chunk = null;
    segments = [];
    curWord = -1;
    overlayEl.textContent = "";
    panelEl.textContent = "";
  };

  window.setSubtitlesVisible = function (v) {
    visible = !!v;
    overlayEl.classList.toggle("subtitles-off", !visible);
    panelEl.classList.toggle("subtitles-off", !visible);
  };

  window.setSubtitleMode = function (m) {
    mode = m === "separated" ? "separated" : "overlay";
    if (chunk) {
      segments = (chunk.words && chunk.words.length && settings.lines > 0)
        ? computeSegments() : [];
      applyVisibility(curWord);
    }
  };

  window.setSubtitleVpos = function (vpos) {
    positionOverlay(Math.min(100, Math.max(0, Number(vpos) || 0)));
  };

  window.applySubtitleAppearance = function () {
    settings = readSettings();
    const root = document.documentElement.style;
    root.setProperty("--lpc-subtitle-font", settings.fontPx + "px");
    root.setProperty("--lpc-subtitle-lines", String(settings.lines || 1));
    root.setProperty("--lpc-highlight", settings.highlightColor);
    positionOverlay(settings.vpos);
    if (chunk) {
      segments = (chunk.words && chunk.words.length && settings.lines > 0)
        ? computeSegments() : [];
      applyVisibility(curWord);
    }
  };

  window.applySubtitleAppearance();
})();
