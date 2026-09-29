/* Voice Emotion Recognition — static web app (runs the CNNs in the browser with ONNX Runtime Web) */
"use strict";

const REPO_URL = "";          // e.g. "https://github.com/your-user/your-repo" → shows a link in the footer
const MAX_REC_S = 10;         // recording stops automatically after this many seconds

// ---------------------------------------------------------------------------
// Texts
// ---------------------------------------------------------------------------
const I18N = {
  en: {
    title: "What emotion is in your voice?",
    subtitle: "Speech emotion recognition with convolutional neural networks",
    recordTitle: "1 · Record",
    recordHint: "Say a 2–5 second sentence with a clear emotion: angry, happy, sad, scared… The words don't matter, the way you say them does.",
    record: "Record", stop: "Stop",
    upload: "…or upload an audio file",
    live: "live spectrogram",
    recording: "Recording… {s} s",
    processing: "Analyzing…",
    micError: "Microphone unavailable. Allow access in your browser, or upload a file instead.",
    decodeError: "This audio file could not be read. Try WAV or MP3.",
    tooShort: "The audio is too short or silent. Record at least 1–2 seconds of speech.",
    manifestError: "Could not load the models (models/manifest.json). Run export_onnx.py and serve the folder over HTTP.",
    signalTitle: "2 · What the network sees",
    signalEmpty: "Your signal and its spectrogram will appear here.",
    waveTitle: "Speech signal (silence trimmed)",
    specTitle: "Log-mel spectrogram · 64 mel bands × 301 frames — the image fed to the CNN",
    windowLabel: "analyzed window (3 s)",
    time: "time (s)", mel: "mel band",
    duration: "Speech duration: {d} s", windowNote: " — only the central 3 s are analyzed.",
    modelsTitle: "3 · Predictions",
    modelsHint: "Choose which models to compare. UAR = unweighted average recall on speakers never heard during training.",
    loadingModels: "Loading models {i}/{n}…",
    noModel: "Select at least one model.",
    waitingModel: "Waiting for audio…",
    loadingModel: "Loading model…",
    modelError: "This model failed to run.",
    emotionsWord: "emotions",
    uarHelp: "Unweighted average recall, speaker-independent cross-validation",
    agreement: "{k} of {n} models hear {emo}.",
    disagree: "The models disagree: emotion in the voice is often ambiguous.",
    aboutTitle: "How it works",
    aboutPipeline: "Audio is resampled to 16 kHz mono, leading and trailing silence is trimmed, and the central 3 seconds become a <b>64-band log-mel spectrogram</b>. Each CNN reads that image and outputs a probability for every emotion. The same preprocessing was used during training.",
    aboutCaveat: "Each model was trained on a single <b>acted</b> corpus ({corpora}) and evaluated on speakers it had never heard. Spontaneous speech, other languages or a different microphone make the task harder, so real-world accuracy is lower than the UAR shown.",
    privacy: "Everything runs in your browser: your voice is never uploaded to a server.",
    footer: "Course project · Voice emotion recognition with CNNs",
  },
  fr: {
    title: "Quelle émotion y a-t-il dans votre voix ?",
    subtitle: "Reconnaissance des émotions dans la parole avec des réseaux de neurones convolutifs",
    recordTitle: "1 · Enregistrer",
    recordHint: "Dites une phrase de 2 à 5 secondes avec une émotion claire : en colère, joyeux, triste, effrayé… Les mots importent peu, c'est la façon de les dire qui compte.",
    record: "Enregistrer", stop: "Arrêter",
    upload: "…ou importer un fichier audio",
    live: "spectrogramme en direct",
    recording: "Enregistrement… {s} s",
    processing: "Analyse…",
    micError: "Micro indisponible. Autorisez l'accès dans votre navigateur ou importez un fichier.",
    decodeError: "Impossible de lire ce fichier audio. Essayez WAV ou MP3.",
    tooShort: "L'audio est trop court ou silencieux. Enregistrez au moins 1 à 2 secondes de parole.",
    manifestError: "Impossible de charger les modèles (models/manifest.json). Lancez export_onnx.py et servez le dossier en HTTP.",
    signalTitle: "2 · Ce que voit le réseau",
    signalEmpty: "Votre signal et son spectrogramme apparaîtront ici.",
    waveTitle: "Signal de parole (silences supprimés)",
    specTitle: "Spectrogramme log-mel · 64 bandes mel × 301 trames — l'image donnée au CNN",
    windowLabel: "fenêtre analysée (3 s)",
    time: "temps (s)", mel: "bande mel",
    duration: "Durée de parole : {d} s", windowNote: " — seules les 3 s centrales sont analysées.",
    modelsTitle: "3 · Prédictions",
    modelsHint: "Choisissez les modèles à comparer. UAR = rappel moyen non pondéré sur des locuteurs jamais entendus à l'entraînement.",
    loadingModels: "Chargement des modèles {i}/{n}…",
    noModel: "Sélectionnez au moins un modèle.",
    waitingModel: "En attente d'un audio…",
    loadingModel: "Chargement du modèle…",
    modelError: "Ce modèle n'a pas pu s'exécuter.",
    emotionsWord: "émotions",
    uarHelp: "Rappel moyen non pondéré, validation croisée indépendante du locuteur",
    agreement: "{k} modèles sur {n} entendent : {emo}.",
    disagree: "Les modèles ne sont pas d'accord : l'émotion dans la voix est souvent ambiguë.",
    aboutTitle: "Comment ça marche",
    aboutPipeline: "L'audio est rééchantillonné à 16 kHz mono, les silences de début et de fin sont supprimés, et les 3 secondes centrales deviennent un <b>spectrogramme log-mel à 64 bandes</b>. Chaque CNN lit cette image et donne une probabilité pour chaque émotion. Le même prétraitement a été utilisé à l'entraînement.",
    aboutCaveat: "Chaque modèle a été entraîné sur un seul corpus <b>joué par des acteurs</b> ({corpora}) et évalué sur des locuteurs qu'il n'avait jamais entendus. La parole spontanée, une autre langue ou un autre micro rendent la tâche plus difficile : la précision réelle est inférieure à l'UAR affiché.",
    privacy: "Tout s'exécute dans votre navigateur : votre voix n'est jamais envoyée à un serveur.",
    footer: "Projet de cours · Reconnaissance des émotions dans la voix avec des CNN",
  },
  es: {
    title: "¿Qué emoción hay en tu voz?",
    subtitle: "Reconocimiento de emociones en el habla con redes neuronales convolucionales",
    recordTitle: "1 · Graba",
    recordHint: "Di una frase de 2 a 5 segundos con una emoción clara: enojado, alegre, triste, asustado… No importan las palabras, sino cómo las dices.",
    record: "Grabar", stop: "Detener",
    upload: "…o sube un archivo de audio",
    live: "espectrograma en vivo",
    recording: "Grabando… {s} s",
    processing: "Analizando…",
    micError: "Micrófono no disponible. Permite el acceso en el navegador o sube un archivo.",
    decodeError: "No se pudo leer este archivo de audio. Prueba con WAV o MP3.",
    tooShort: "El audio es muy corto o está en silencio. Graba al menos 1–2 segundos de voz.",
    manifestError: "No se pudieron cargar los modelos (models/manifest.json). Ejecuta export_onnx.py y sirve la carpeta por HTTP.",
    signalTitle: "2 · Lo que ve la red",
    signalEmpty: "Aquí aparecerán tu señal y su espectrograma.",
    waveTitle: "Señal de voz (sin silencios)",
    specTitle: "Espectrograma log-mel · 64 bandas mel × 301 tramas — la imagen que recibe la CNN",
    windowLabel: "ventana analizada (3 s)",
    time: "tiempo (s)", mel: "banda mel",
    duration: "Duración de voz: {d} s", windowNote: " — solo se analizan los 3 s centrales.",
    modelsTitle: "3 · Predicciones",
    modelsHint: "Elige qué modelos comparar. UAR = recall promedio no ponderado en locutores que el modelo nunca escuchó.",
    loadingModels: "Cargando modelos {i}/{n}…",
    noModel: "Selecciona al menos un modelo.",
    waitingModel: "Esperando audio…",
    loadingModel: "Cargando modelo…",
    modelError: "Este modelo no se pudo ejecutar.",
    emotionsWord: "emociones",
    uarHelp: "Recall promedio no ponderado, validación cruzada independiente del locutor",
    agreement: "{k} de {n} modelos escuchan: {emo}.",
    disagree: "Los modelos no coinciden: la emoción en la voz suele ser ambigua.",
    aboutTitle: "Cómo funciona",
    aboutPipeline: "El audio se remuestrea a 16 kHz mono, se recortan los silencios del inicio y el final, y los 3 segundos centrales se convierten en un <b>espectrograma log-mel de 64 bandas</b>. Cada CNN lee esa imagen y da una probabilidad para cada emoción. Es el mismo preprocesamiento usado en el entrenamiento.",
    aboutCaveat: "Cada modelo se entrenó con un solo corpus <b>actuado</b> ({corpora}) y se evaluó con locutores que nunca escuchó. El habla espontánea, otro idioma u otro micrófono hacen la tarea más difícil: la precisión real es menor que el UAR mostrado.",
    privacy: "Todo se ejecuta en tu navegador: tu voz nunca se envía a un servidor.",
    footer: "Proyecto de curso · Reconocimiento de emociones en la voz con CNN",
  },
};

const EMO = {
  anger:     { e: "😠", en: "Anger",     fr: "Colère",   es: "Enojo" },
  happiness: { e: "😄", en: "Happiness", fr: "Joie",     es: "Alegría" },
  sadness:   { e: "😢", en: "Sadness",   fr: "Tristesse", es: "Tristeza" },
  fear:      { e: "😨", en: "Fear",      fr: "Peur",     es: "Miedo" },
  disgust:   { e: "🤢", en: "Disgust",   fr: "Dégoût",   es: "Asco" },
  boredom:   { e: "🥱", en: "Boredom",   fr: "Ennui",    es: "Aburrimiento" },
  neutral:   { e: "😐", en: "Neutral",   fr: "Neutre",   es: "Neutral" },
  calm:      { e: "😌", en: "Calm",      fr: "Calme",    es: "Calma" },
  surprise:  { e: "😲", en: "Surprise",  fr: "Surprise", es: "Sorpresa" },
};
const EMO_SYNONYMS = { angry: "anger", happy: "happiness", joy: "happiness", sad: "sadness",
  fearful: "fear", surprised: "surprise", disgusted: "disgust", bored: "boredom" };
const ARCH = {
  scratch:  { en: "CNN from scratch", fr: "CNN entraîné de zéro", es: "CNN desde cero" },
  resnet18: { en: "ResNet-18 (transfer learning)", fr: "ResNet-18 (apprentissage par transfert)", es: "ResNet-18 (transfer learning)" },
};
const LANG_NAMES = {
  de: { en: "German", fr: "allemand", es: "alemán" },
  en: { en: "English", fr: "anglais", es: "inglés" },
  es: { en: "Spanish", fr: "espagnol", es: "español" },
  pt: { en: "Portuguese", fr: "portugais", es: "portugués" },
  fr: { en: "French", fr: "français", es: "francés" },
  multi: { en: "multilingual", fr: "multilingue", es: "multilingüe" },
};
const CORPUS_DESC = {
  EmoDB:     { en: "German actors, studio", fr: "acteurs allemands, studio", es: "actores alemanes, estudio" },
  RAVDESS:   { en: "North-American actors", fr: "acteurs nord-américains", es: "actores norteamericanos" },
  "CREMA-D": { en: "American actors", fr: "acteurs américains", es: "actores estadounidenses" },
};
const MAGMA = ("00000402010903031206051a0a08220e0b2b130d34180f3d1d114722115029115a2f116336106b3d0f71440f764a" +
  "107952137c59157e5f187f651a806b1d81721f817822817e24828426818b2981912b81982d809e2f7fa5317eab337cb2357bba" +
  "3878c03a76c73d73cd4071d3436ed9466bdf4a68e44f64e95462ed5a5ff1605df4675cf66e5cf8765cfa7d5efb8560fc8e64fd" +
  "9668fe9d6cfea571feac76feb47bfebb81fec287feca8dfed194fed89afde0a1fde7a9fceeb0fcf6b8fcfdbf")
  .match(/.{6}/g).map((h) => [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)));

// ---------------------------------------------------------------------------
// State & helpers
// ---------------------------------------------------------------------------
const $ = (s) => document.querySelector(s);
const state = { lang: "en", manifest: null, selected: new Set(), sessions: {}, results: {},
                pre: null, recording: null, runId: 0, loading: "" };

function t(key, vars = {}) {
  const s = (I18N[state.lang] && I18N[state.lang][key]) ?? I18N.en[key] ?? key;
  return s.replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "");
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const canon = (c) => EMO_SYNONYMS[c.toLowerCase()] || c.toLowerCase();
const emoName = (c) => (EMO[canon(c)] && EMO[canon(c)][state.lang]) || c;
const emoEmoji = (c) => (EMO[canon(c)] && EMO[canon(c)].e) || "🎭";
const archName = (m) => (ARCH[m.arch] && ARCH[m.arch][state.lang]) || m.arch;
const langName = (code) => (LANG_NAMES[code] && LANG_NAMES[code][state.lang]) || code;
const pct = (v, digits = 0) => new Intl.NumberFormat(state.lang, { style: "percent", maximumFractionDigits: digits, minimumFractionDigits: digits }).format(v);
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

function setStatus(msg = "", kind = "info") {
  const el = $("#status");
  el.textContent = msg;
  el.className = "status " + kind;
  el.hidden = !msg;
}

// ---------------------------------------------------------------------------
// Language
// ---------------------------------------------------------------------------
function applyLang() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.lang === state.lang));
  $("#recLabel").textContent = state.recording ? t("stop") : t("record");
  $("#liveLabel").textContent = t("live");
  $("#aboutPipeline").innerHTML = t("aboutPipeline");
  const corpora = state.manifest
    ? [...new Set(state.manifest.models.map((m) => m.corpus))]
        .map((c) => (CORPUS_DESC[c] ? `${c}: ${CORPUS_DESC[c][state.lang]}` : c)).join("; ")
    : "…";
  $("#aboutCaveat").innerHTML = t("aboutCaveat", { corpora: esc(corpora) });
  renderPicker();
  renderResults();
  drawSignal();
}

function setLang(lang) {
  state.lang = I18N[lang] ? lang : "en";
  try { localStorage.setItem("ser-lang", state.lang); } catch (e) { /* storage unavailable */ }
  applyLang();
}

// ---------------------------------------------------------------------------
// Models (ONNX Runtime Web)
// ---------------------------------------------------------------------------
function getSession(m) {
  if (!state.sessions[m.id]) {
    state.sessions[m.id] = ort.InferenceSession.create("models/" + m.file, { executionProviders: ["wasm"] })
      .catch((e) => { delete state.sessions[m.id]; throw e; });
  }
  return state.sessions[m.id];
}

async function infer(m, spec) {
  const session = await getSession(m);
  const x = new Float32Array(spec.data.length);
  for (let i = 0; i < x.length; i++) x[i] = (spec.data[i] - m.mu) / m.sd;
  const input = new ort.Tensor("float32", x, [1, 1, spec.nMels, spec.nFrames]);
  const out = await session.run({ [session.inputNames[0]]: input });
  const logits = Array.from(out[session.outputNames[0]].data);
  const mx = Math.max(...logits);
  const ex = logits.map((v) => Math.exp(v - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return m.classes.map((c, i) => ({ label: c, p: ex[i] / sum })).sort((a, b) => b.p - a.p);
}

async function preloadModels() {
  const list = state.manifest.models.filter((m) => state.selected.has(m.id));
  for (let i = 0; i < list.length; i++) {
    state.loading = t("loadingModels", { i: i + 1, n: list.length });
    $("#loadState").textContent = state.loading;
    try { await getSession(list[i]); } catch (e) { console.error(e); }
  }
  state.loading = "";
  $("#loadState").textContent = "";
}

async function runModels() {
  const run = state.runId, pre = state.pre;
  if (!pre) return;
  renderResults();
  for (const m of state.manifest.models) {
    if (!state.selected.has(m.id) || state.results[m.id]) continue;
    let res;
    try { res = await infer(m, pre.spec); } catch (e) { console.error(e); res = { error: true }; }
    if (run !== state.runId) return;                    // a newer recording replaced this one
    state.results[m.id] = res;
    renderResults();
  }
}

// ---------------------------------------------------------------------------
// Rendering: model picker + result cards
// ---------------------------------------------------------------------------
function modelMeta(m) {
  return `${esc(m.corpus)} · ${esc(langName(m.lang))} · ${m.classes.length} ${esc(t("emotionsWord"))}`;
}
function uarPill(m) {
  return m.uar != null ? `<span class="pill" title="${esc(t("uarHelp"))}">UAR ${pct(m.uar, 1)}</span>` : "";
}

function renderPicker() {
  const box = $("#picker");
  if (!state.manifest) { box.innerHTML = ""; return; }
  const groups = {};
  state.manifest.models.forEach((m) => { (groups[m.corpus + "|" + m.lang] ||= []).push(m); });
  box.innerHTML = Object.values(groups).map((ms) => `
    <div class="group">
      <div class="group-title">${modelMeta(ms[0])}</div>
      <div class="chips">${ms.map((m) => `
        <label class="chip">
          <input type="checkbox" data-id="${esc(m.id)}" ${state.selected.has(m.id) ? "checked" : ""}>
          <span class="chip-name">${esc(archName(m))}</span>
          ${m.uar != null ? `<span class="chip-uar">${pct(m.uar, 1)}</span>` : ""}
          ${m.size_mb ? `<span class="chip-size">${m.size_mb} MB</span>` : ""}
        </label>`).join("")}
      </div>
    </div>`).join("");
}

function renderResults() {
  const box = $("#results"), summary = $("#agreement");
  summary.textContent = "";
  if (!state.manifest) { box.innerHTML = ""; return; }
  const models = state.manifest.models.filter((m) => state.selected.has(m.id));
  if (!models.length) { box.innerHTML = `<p class="muted">${esc(t("noModel"))}</p>`; return; }

  box.innerHTML = models.map((m) => {
    const r = state.results[m.id];
    let body;
    if (!state.pre) body = `<p class="muted placeholder">${esc(t("waitingModel"))}</p>`;
    else if (!r) body = `<p class="muted placeholder"><span class="spinner"></span>${esc(state.sessions[m.id] ? t("processing") : t("loadingModel"))}</p>`;
    else if (r.error) body = `<p class="error-text">${esc(t("modelError"))}</p>`;
    else body = `
      <div class="top-pred">
        <span class="big-emoji">${emoEmoji(r[0].label)}</span>
        <div><div class="pred-name">${esc(emoName(r[0].label))}</div><div class="muted">${pct(r[0].p)}</div></div>
      </div>
      <div class="bars">${r.map((c, i) => `
        <div class="bar-row${i === 0 ? " top" : ""}">
          <span class="bar-label">${emoEmoji(c.label)} ${esc(emoName(c.label))}</span>
          <span class="bar"><span style="width:${(c.p * 100).toFixed(1)}%"></span></span>
          <span class="bar-val">${pct(c.p)}</span>
        </div>`).join("")}
      </div>`;
    return `<article class="result">
      <header><div><h3>${esc(archName(m))}</h3><p class="muted small">${modelMeta(m)}</p></div>${uarPill(m)}</header>
      ${body}</article>`;
  }).join("");

  const done = models.map((m) => state.results[m.id]).filter((r) => r && !r.error);
  if (done.length >= 2 && done.length === models.length) {
    const votes = {};
    done.forEach((r) => { const k = canon(r[0].label); votes[k] = (votes[k] || 0) + 1; });
    const [best, k] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
    summary.textContent = k >= 2
      ? `${emoEmoji(best)} ${t("agreement", { k, n: done.length, emo: emoName(best) })}`
      : `🤔 ${t("disagree")}`;
  }
}

// ---------------------------------------------------------------------------
// Rendering: waveform + spectrogram
// ---------------------------------------------------------------------------
function prepCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return { g, w, h };
}

function drawAxisX(g, x0, width, y, maxS, step) {
  g.strokeStyle = cssVar("--line"); g.fillStyle = cssVar("--muted");
  g.font = "11px system-ui, sans-serif"; g.textAlign = "center"; g.textBaseline = "top";
  g.beginPath(); g.moveTo(x0, y + 0.5); g.lineTo(x0 + width, y + 0.5); g.stroke();
  for (let s = 0; s <= maxS + 1e-9; s += step) {
    const x = x0 + (s / maxS) * width;
    g.beginPath(); g.moveTo(x + 0.5, y); g.lineTo(x + 0.5, y + 4); g.stroke();
    g.fillText(Number(s.toFixed(1)).toLocaleString(state.lang), x, y + 6);
  }
}

function drawSignal() {
  const has = !!state.pre;
  $("#signalEmpty").hidden = has;
  $("#signalPlots").hidden = !has;
  if (!has) return;
  const { sr, n_samples: N } = state.manifest;
  const sig = state.pre.signal;
  $("#info").textContent = t("duration", { d: (sig.length / sr).toLocaleString(state.lang, { maximumFractionDigits: 1 }) })
    + (sig.length > N ? t("windowNote") : ".");

  // --- waveform ---
  {
    const { g, w, h } = prepCanvas($("#wave"));
    const L = { l: 10, r: 10, t: 20, b: 24 }, pw = w - L.l - L.r, ph = h - L.t - L.b;
    const dur = Math.max(sig.length, N) / sr;
    const X = (s) => L.l + (s / dur) * pw;
    if (sig.length > N) {
      const s0 = state.pre.windowStart / sr, x0 = X(s0), x1 = X(s0 + N / sr);
      g.fillStyle = cssVar("--accent-soft");
      g.fillRect(x0, 2, x1 - x0, L.t - 2 + ph);
      g.fillStyle = cssVar("--accent"); g.font = "600 11px system-ui, sans-serif";
      g.textAlign = "center"; g.textBaseline = "top";
      g.fillText(t("windowLabel"), (x0 + x1) / 2, 4);          // label above the waveform
    }
    g.strokeStyle = cssVar("--wave"); g.lineWidth = 1;
    const mid = L.t + ph / 2, spp = (dur * sr) / pw;
    g.beginPath();
    for (let px = 0; px < pw; px++) {
      const a = Math.floor(px * spp), b = Math.min(sig.length, Math.floor((px + 1) * spp) + 1);
      if (a >= sig.length) break;
      let mn = 1, mx = -1;
      for (let i = a; i < b; i++) { if (sig[i] < mn) mn = sig[i]; if (sig[i] > mx) mx = sig[i]; }
      g.moveTo(L.l + px + 0.5, mid - mx * ph / 2);
      g.lineTo(L.l + px + 0.5, mid - mn * ph / 2 + 0.5);
    }
    g.stroke();
    drawAxisX(g, L.l, pw, L.t + ph, dur, dur > 6 ? 2 : 1);
  }

  // --- log-mel spectrogram ---
  {
    const { g, w, h } = prepCanvas($("#spec"));
    const { data, nMels, nFrames } = state.pre.spec;
    const L = { l: 40, r: 72, t: 6, b: 38 }, pw = w - L.l - L.r, ph = h - L.t - L.b;
    let mn = Infinity, mx = -Infinity;
    for (const v of data) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const off = document.createElement("canvas");
    off.width = nFrames; off.height = nMels;
    const og = off.getContext("2d"), img = og.createImageData(nFrames, nMels);
    for (let m = 0; m < nMels; m++) {
      const row = nMels - 1 - m;                          // low frequencies at the bottom
      for (let t_ = 0; t_ < nFrames; t_++) {
        const v = (data[m * nFrames + t_] - mn) / (mx - mn || 1);
        const c = MAGMA[Math.min(63, Math.floor(v * 64))], k = (row * nFrames + t_) * 4;
        img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
      }
    }
    og.putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true;
    g.drawImage(off, L.l, L.t, pw, ph);

    drawAxisX(g, L.l, pw, L.t + ph, N / sr, 0.5);
    g.fillStyle = cssVar("--muted"); g.font = "11px system-ui, sans-serif";
    g.textAlign = "center"; g.textBaseline = "bottom";
    g.fillText(t("time"), L.l + pw / 2, h - 1);
    g.textAlign = "right"; g.textBaseline = "middle";
    for (let m = 0; m <= nMels; m += 16) g.fillText(String(m), L.l - 6, L.t + ph - (m / nMels) * ph);
    g.save(); g.translate(11, L.t + ph / 2); g.rotate(-Math.PI / 2);
    g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(t("mel"), 0, 0); g.restore();

    // colorbar
    const cx = L.l + pw + 10, cw = 10;
    for (let i = 0; i < 64; i++) {
      const c = MAGMA[i];
      g.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      g.fillRect(cx, L.t + ph - ((i + 1) / 64) * ph, cw, ph / 64 + 0.6);
    }
    g.fillStyle = cssVar("--muted"); g.textAlign = "left"; g.textBaseline = "top";
    g.fillText(`${Math.round(mx)} dB`, cx + cw + 4, L.t);
    g.textBaseline = "bottom";
    g.fillText(`${Math.round(mn)} dB`, cx + cw + 4, L.t + ph);
  }
}

// ---------------------------------------------------------------------------
// Audio input: recording (with live spectrogram) and file upload
// ---------------------------------------------------------------------------
async function decodeTo16k(arrayBuffer) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  let buf;
  try { buf = await ctx.decodeAudioData(arrayBuffer); } finally { ctx.close(); }
  const sr = state.manifest.sr;
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(buf.duration * sr)), sr);
  const src = off.createBufferSource();
  src.buffer = buf; src.connect(off.destination); src.start();
  const out = await off.startRendering();                 // resampled to 16 kHz, mixed to mono
  return new Float32Array(out.getChannelData(0));
}

async function handleAudio(blob) {
  $("#player").src = URL.createObjectURL(blob);
  $("#player").hidden = false;
  setStatus(t("processing"));
  let y;
  try { y = await decodeTo16k(await blob.arrayBuffer()); }
  catch (e) { console.error(e); setStatus(t("decodeError"), "error"); return; }
  analyze(y);
}

function analyze(y) {
  const pre = DSP.preprocess(y, state.manifest);
  state.runId++;
  state.results = {};
  if (pre.tooShort) {
    state.pre = null;
    setStatus(t("tooShort"), "warn");
    drawSignal(); renderResults();
    return;
  }
  setStatus("");
  state.pre = pre;
  drawSignal();
  runModels();
}

function clearLive() {
  const c = $("#live"), dpr = window.devicePixelRatio || 1;
  c.width = Math.round(c.clientWidth * dpr); c.height = Math.round(c.clientHeight * dpr);
  const g = c.getContext("2d");
  g.fillStyle = "rgb(0,0,4)"; g.fillRect(0, 0, c.width, c.height);
}

async function startRecording() {
  if (!navigator.mediaDevices || !window.MediaRecorder) { setStatus(t("micError"), "error"); return; }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { console.error(e); setStatus(t("micError"), "error"); return; }

  const AC = window.AudioContext || window.webkitAudioContext;
  const ac = new AC();
  const an = ac.createAnalyser();
  an.fftSize = 2048; an.smoothingTimeConstant = 0; an.minDecibels = -95; an.maxDecibels = -25;
  ac.createMediaStreamSource(stream).connect(an);

  const chunks = [], rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  const r = { rec, an, ac, stream, t0: performance.now(), raf: 0, bins: new Uint8Array(an.frequencyBinCount) };
  rec.onstop = () => {
    cancelAnimationFrame(r.raf);
    stream.getTracks().forEach((tr) => tr.stop());
    ac.close();
    $("#timer").textContent = "";
    handleAudio(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
  };

  clearLive();
  rec.start();
  state.recording = r;
  $("#recBtn").classList.add("recording");
  $("#recLabel").textContent = t("stop");
  setStatus("");
  liveLoop();
}

function stopRecording() {
  const r = state.recording;
  if (!r) return;
  state.recording = null;
  $("#recBtn").classList.remove("recording");
  $("#recLabel").textContent = t("record");
  if (r.rec.state !== "inactive") r.rec.stop();
}

function liveLoop() {
  const r = state.recording;
  if (!r) return;
  const c = $("#live"), g = c.getContext("2d"), w = c.width, h = c.height;
  const step = Math.max(1, Math.round(2 * (window.devicePixelRatio || 1)));
  r.an.getByteFrequencyData(r.bins);
  g.drawImage(c, -step, 0);                                 // scroll left
  const col = g.createImageData(step, h), nyq = r.ac.sampleRate / 2, melMax = hzToMel(8000);
  for (let y = 0; y < h; y++) {
    const f = melToHz((1 - (y + 0.5) / h) * melMax);        // mel-spaced rows, 0–8 kHz
    const v = r.bins[Math.min(r.bins.length - 1, Math.round((f / nyq) * r.bins.length))];
    const cc = MAGMA[v >> 2];
    for (let x = 0; x < step; x++) {
      const k = (y * step + x) * 4;
      col.data[k] = cc[0]; col.data[k + 1] = cc[1]; col.data[k + 2] = cc[2]; col.data[k + 3] = 255;
    }
  }
  g.putImageData(col, w - step, 0);
  const s = (performance.now() - r.t0) / 1000;
  $("#timer").textContent = t("recording", { s: s.toLocaleString(state.lang, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) });
  if (s >= MAX_REC_S) { stopRecording(); return; }
  r.raf = requestAnimationFrame(liveLoop);
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------
async function init() {
  let lang = "en";
  try { lang = localStorage.getItem("ser-lang") || ""; } catch (e) { lang = ""; }
  if (!I18N[lang]) lang = (navigator.language || "en").slice(0, 2);
  state.lang = I18N[lang] ? lang : "en";

  document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
  $("#recBtn").addEventListener("click", () => (state.recording ? stopRecording() : startRecording()));
  $("#file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (f) handleAudio(f);
    e.target.value = "";
  });
  $("#picker").addEventListener("change", (e) => {
    const id = e.target.dataset.id;
    if (!id) return;
    if (e.target.checked) state.selected.add(id); else state.selected.delete(id);
    renderResults();
    if (e.target.checked) {
      const m = state.manifest.models.find((x) => x.id === id);
      getSession(m).catch(() => {});
      runModels();
    }
  });
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawSignal, 150); });
  if (REPO_URL) { $("#repo").href = REPO_URL; $("#repo").hidden = false; }

  clearLive();
  applyLang();

  if (typeof ort !== "undefined") {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
  }
  try {
    const res = await fetch("models/manifest.json", { cache: "no-cache" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    state.manifest = await res.json();
  } catch (e) {
    console.error(e);
    setStatus(t("manifestError"), "error");
    return;
  }
  state.manifest.models.forEach((m) => state.selected.add(m.id));
  $("#recBtn").disabled = false;
  $("#file").disabled = false;
  applyLang();
  preloadModels();
}

document.addEventListener("DOMContentLoaded", init);
