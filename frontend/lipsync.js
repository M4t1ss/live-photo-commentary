// lipsync.js — loaded as a plain <script> before app.js and avatar.js.
// Exposes:
//   initLipSync(config)    — called by app.js after fetching /model-config
//   buildTimeline(phonemes) — called by app.js when a TTS chunk arrives

let _visemeMap = {};

window.initLipSync = function (config) {
  _visemeMap = config.visemeMap ?? {};
};

// buildTimeline converts Kokoro phoneme timing into a flat list of morph events.
//
// phonemes: [[ph, cumulative_end_secs], ...]  (as received in the chunk message)
//
// Timing model (anticipatory coarticulation):
//   absPeak  = start of this phoneme  (mouth shape peaks at utterance onset)
//   absStart = start of previous phoneme  (mouth begins moving toward this shape)
//   absEnd   = start of next phoneme  (mouth finishes transitioning away)
// First/last phonemes get a half-duration margin at their open end.
window.buildTimeline = function (phonemes) {
  if (!phonemes || !phonemes.length) return [];

  // starts[i] = absolute start time (seconds) of phoneme[i]
  const starts = [0, ...phonemes.map(([, t]) => t)];

  // A phoneme is "visual" if its mapping has at least one morph/jaw entry.
  // Stress marks, intonation, length marks etc. have {} and are non-visual.
  const isVisual = phonemes.map(([ph]) => {
    const m = _visemeMap[ph];
    return m != null && Object.keys(m).length > 0;
  });

  const events = [];
  for (let i = 0; i < phonemes.length; i++) {
    const mapping = _visemeMap[phonemes[i][0]];
    if (!mapping || !isVisual[i]) continue;

    const dur     = starts[i + 1] - starts[i];
    const absPeak = starts[i];

    // absStart: start of the nearest previous visual phoneme.
    // Non-visual phonemes between this one and its visual predecessor are skipped.
    let prevVisual = i - 1;
    while (prevVisual >= 0 && !isVisual[prevVisual]) prevVisual--;
    const absStart = prevVisual < 0
      ? Math.max(0, absPeak - dur * 0.5)
      : starts[prevVisual];

    // absEnd: start of the nearest next visual phoneme.
    // Non-visual phonemes between this one and its visual successor are skipped.
    let nextVisual = i + 1;
    while (nextVisual < phonemes.length && !isVisual[nextVisual]) nextVisual++;
    const absEnd = nextVisual >= phonemes.length
      ? starts[phonemes.length] + dur * 0.5
      : starts[nextVisual];

    for (const [key, value] of Object.entries(mapping)) {
      events.push({ key, absStart, absPeak, absEnd, value });
    }
  }
  return events;
};
