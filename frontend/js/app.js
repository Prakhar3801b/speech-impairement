import { Recorder, startLiveSpectrogram, drawMatrix } from './audio.js';
import { SignClassifier, SignStabilizer, LETTERS, CONTROLS } from './signs.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || 'POST';
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  const res = await fetch(path, init);
  if (res.status === 401) { location.replace('/login.html'); throw new Error('Please sign in'); }
  const type = res.headers.get('content-type') || '';
  if (!res.ok) {
    const body = type.includes('json') ? await res.json() : { detail: await res.text() };
    const detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body);
    throw new Error(detail);
  }
  return type.includes('json') ? res.json() : res;
}

let toastTimer;
function toast(msg, ms = 3000) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

// ------------------------------------------------------------------ state
const state = {
  session: localStorage.getItem('ps16.session') || newSessionId(),
  status: null, phrases: [], spelled: '', signPhrases: [],
  candidates: [], selected: 0, shownAt: 0, actions: 0, lastResult: null,
  recMode: null, vision: null, cameraOn: false,
  signLog: [], lastSignFeatures: null, scanTimer: null, lastPartnerTurn: 0, partnerPoll: null,
};
function newSessionId() {
  const id = 's-' + Math.random().toString(36).slice(2, 10);
  try { localStorage.setItem('ps16.session', id); } catch { /* storage unavailable */ }
  return id;
}
// Per-account profile from the first-run setup (language, "yes" signal, voice choice), kept on this device.
const profile = { name: '', lang: 'auto', yes: 'thumbs', voice: 'default', done: false };
let profileKey = 'clio.profile';
function loadProfile(user) {
  profileKey = `clio.profile.${user.id}`;
  try { Object.assign(profile, JSON.parse(localStorage.getItem(profileKey) || '{}')); } catch { /* storage unavailable */ }
  if (!profile.name) profile.name = user.name || '';
}
function saveProfile() {
  try { localStorage.setItem(profileKey, JSON.stringify(profile)); } catch { /* storage unavailable */ }
}

const recorder = new Recorder();
const classifier = new SignClassifier();

// ------------------------------------------------------------------ tabs
document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.id === 'tab-' + b.dataset.tab));
  ({ signs: loadSigns, spectro: loadTemplates, voice: loadVoiceTab, phrases: loadPhrases, metrics: loadMetrics })[b.dataset.tab]?.();
}));

// ------------------------------------------------------------------ status
async function loadStatus() {
  state.status = await api('/api/status');
  const s = state.status;
  $('#status').innerHTML = [
    `<span class="chip ${s.elevenlabs ? 'on' : 'off'}">ElevenLabs: ${s.elevenlabs ? 'live' : 'mock'}</span>`,
    `<span class="chip ${s.llm ? 'on' : 'off'}">LLM: ${s.llm ? 'on' : 'off (heuristic)'}</span>`,
    `<span class="chip">templates: ${s.templates}</span>`,
    `<span class="chip">TTS cache: ${s.cache.tts.n}</span>`,
    s.elevenlabs ? `<span class="chip ${s.budget.fraction_left < 0.1 ? 'danger' : s.budget.saver_mode ? 'off' : 'on'}"
      title="${s.budget.keys.length} key(s)${s.budget.saver_mode ? ' · saver mode' : ''}">credits left: ${Math.round(s.budget.fraction_left * 100)}%</span>` : '',
  ].join('');
}

async function loadPhraseCache() { state.phrases = await api('/api/phrases'); }

// ------------------------------------------------------------------ media
$('#startCamera').addEventListener('click', startMedia);
async function startMedia() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (e) {
    try {   // no camera (or camera denied): keep going with the microphone only
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      toast('Camera unavailable; running with microphone only', 5000);
    } catch (e2) {
      toast('Could not access the microphone: ' + e2.message, 6000);
      return;
    }
  }
  if (stream.getVideoTracks().length) {
    $('#video').srcObject = stream; await $('#video').play();
    $('#cameraOff').hidden = true; state.cameraOn = true;
  }
  await recorder.init(stream);
  startLiveSpectrogram($('#liveSpec'), recorder.analyser);
  if (!state.cameraOn) return;
  toast('Loading hand and face models…');
  try {
    const { Vision } = await import('./vision.js');
    state.vision = new Vision($('#video'), $('#overlay'));
    state.vision.on('frame', onVisionFrame);
    await state.vision.init();
    toast('Camera, microphone and tracking ready');
  } catch (e) {
    console.error(e);
    state.vision = null;
    toast('Hand/face tracking could not load (needs access to cdn.jsdelivr.net and storage.googleapis.com). Voice features still work.', 7000);
  }
}

function requireMic() {
  if (!recorder.ctx) { toast('Start the camera & mic first (Talk tab)'); return false; }
  return true;
}

// ------------------------------------------------------------------ vision frame handling
const stabilizer = new SignStabilizer(onSignEntered);
let thumbHold = { dir: null, n: 0, fired: false };
let lastHead = null, palmSince = 0, lastBrowDouble = 0;

function onVisionFrame(s) {
  $('#cueExpr').textContent = 'face: ' + s.expression;
  $('#cueExpr').classList.toggle('danger', s.expression === 'pain');
  $('#cueHead').textContent = 'head: ' + (s.head || '–');
  $('#cueHead').classList.toggle('on', !!s.head);
  $('#cueThumb').textContent = 'thumb: ' + (s.thumbs || '–');
  $('#cueThumb').classList.toggle('on', !!s.thumbs);
  if (state.vision?.utter) $('#cueSyl').textContent = 'syllables: ' + state.vision.utter.syllables;

  // emergency: an open palm held for 2.5 s, or two quick eyebrow raises
  if ($('#emergencyGestures').checked) {
    const now = performance.now();
    palmSince = s.openPalm ? (palmSince || now) : 0;
    if (palmSince && now - palmSince > 2500) { palmSince = 0; triggerEmergency('open palm'); }
    if (s.browDouble && s.browDouble !== lastBrowDouble) { lastBrowDouble = s.browDouble; triggerEmergency('double brow raise'); }
  }

  // nod / shake: confirm / next, edge-triggered (vision keeps the label for 1.5 s)
  if (profile.yes === 'nod' && s.head !== lastHead) {
    lastHead = s.head;
    if (state.candidates.length && s.head === 'nod') { state.actions++; confirmCandidate(state.selected); }
    if (state.candidates.length && s.head === 'shake') { state.actions++; select((state.selected + 1) % state.candidates.length); }
  }

  // thumbs: confirm / next while suggestions are on screen
  if (profile.yes !== 'thumbs') { /* thumbs gestures off for this user */ }
  else if (state.candidates.length && s.thumbs) {
    if (thumbHold.dir === s.thumbs) thumbHold.n++; else thumbHold = { dir: s.thumbs, n: 0, fired: false };
    if (!thumbHold.fired && s.thumbs === 'up' && thumbHold.n > 15) { thumbHold.fired = true; state.actions++; confirmCandidate(state.selected); }
    if (!thumbHold.fired && s.thumbs === 'down' && thumbHold.n > 10) { thumbHold.fired = true; state.actions++; select((state.selected + 1) % state.candidates.length); }
  } else if (!s.thumbs) thumbHold = { dir: null, n: 0, fired: false };

  // ISL signs (skip frames that are a thumbs gesture)
  const pred = classifier.size && s.features && !s.thumbs ? classifier.predict(s.features) : null;
  $('#cueSign').textContent = 'sign: ' + (pred?.known ? `${pred.label} ${(pred.confidence * 100) | 0}%` : '–');
  $('#cueSign').classList.toggle('on', !!pred?.known);
  if ($('#tab-signs').classList.contains('active')) {
    $('#livePred').textContent = pred ? `${pred.label} (${(pred.confidence * 100) | 0}%, d=${pred.distance.toFixed(2)}${pred.known ? '' : ', unknown'})` : '–';
  }
  if (pred?.known) state.lastSignFeatures = s.features;
  if ($('#signEnabled').checked && !signCapture.active) stabilizer.push(pred);
  if (signCapture.active && s.features) signCapture.push(s.features);
}

function onSignEntered(pred) {
  // remembered so a confirmed message can teach the classifier these exact hand shapes
  if (state.lastSignFeatures && pred.kind !== 'control') {
    state.signLog.push({ label: pred.label, kind: pred.kind,
      phrase_id: pred.phrase_id || null, features: state.lastSignFeatures });
  }
  if (pred.kind === 'letter') state.spelled += pred.label;
  else if (pred.kind === 'control') {
    if (pred.label === 'SPACE') state.spelled += ' ';
    if (pred.label === 'DEL') state.spelled = state.spelled.slice(0, -1);
    if (pred.label === 'SEND') { sendPatient(null); return; }
  } else if (pred.kind === 'phrase' && pred.phrase_id) {
    if (!state.signPhrases.includes(pred.phrase_id)) state.signPhrases.push(pred.phrase_id);
    const p = state.phrases.find((x) => x.id === pred.phrase_id);
    if (p?.is_emergency) { renderSpell(); sendPatient(null); return; }   // fast path
  }
  renderSpell();
}

function renderSpell() {
  $('#spellBuffer').textContent = state.spelled;
  $('#signPhrases').innerHTML = state.signPhrases.map((id) => {
    const p = state.phrases.find((x) => x.id === id);
    return `<span class="chip on">sign: ${esc(p?.text || id)}</span>`;
  }).join('');
}
$('#spellSpace').onclick = () => { state.spelled += ' '; renderSpell(); };
$('#spellBack').onclick = () => { state.spelled = state.spelled.slice(0, -1); renderSpell(); };
$('#spellClear').onclick = () => { state.spelled = ''; state.signPhrases = []; state.signLog = []; renderSpell(); };
$('#spellSend').onclick = () => sendPatient(null);

// ------------------------------------------------------------------ recording controls
function setRecButton(btn, on) { btn.setAttribute('aria-pressed', on ? 'true' : 'false'); }

async function toggleRecord(mode, btn, onDone) {
  if (!requireMic()) return;
  if (state.recMode && state.recMode !== mode) { toast('Already recording: ' + state.recMode); return; }
  if (state.recMode === mode) {
    state.recMode = null; setRecButton(btn, false);
    const { blob, seconds } = recorder.stop();
    const cues = state.vision ? state.vision.endUtterance() : {};
    $('#cueSyl').textContent = 'syllables: ' + (cues.syllables ?? '–');
    if (seconds < 0.3) { toast('Recording too short'); return; }
    onDone(blob, cues);
  } else {
    state.recMode = mode; setRecButton(btn, true);
    state.vision?.beginUtterance();
    recorder.start();
  }
}

$('#partnerBtn').onclick = () => toggleRecord('partner', $('#partnerBtn'), (blob) => sendPartner({ blob }));
$('#patientBtn').onclick = () => toggleRecord('patient', $('#patientBtn'), (blob, cues) => sendPatient(blob, cues));
$('#partnerTextForm').onsubmit = (e) => {
  e.preventDefault();
  const t = $('#partnerText').value.trim(); if (!t) return;
  $('#partnerText').value = ''; sendPartner({ text: t });
};

document.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
  if (!$('#tab-converse').classList.contains('active')) return;
  const k = e.key.toLowerCase();
  if (k === 'o') { e.preventDefault(); $('#partnerBtn').click(); }
  else if (k === 'p') { e.preventDefault(); $('#patientBtn').click(); }
  else if (['1', '2', '3'].includes(k) && state.candidates[+k - 1]) { state.actions++; confirmCandidate(+k - 1); }
  else if ((k === 'arrowdown' || k === 'arrowright') && state.candidates.length) { e.preventDefault(); state.actions++; select((state.selected + 1) % state.candidates.length); }
  else if ((k === 'arrowup' || k === 'arrowleft') && state.candidates.length) { e.preventDefault(); state.actions++; select((state.selected - 1 + state.candidates.length) % state.candidates.length); }
  else if ((k === 'enter' || (k === ' ' && profile.yes === 'scan')) && state.candidates.length) {
    e.preventDefault(); state.actions++; confirmCandidate(state.selected);
  }
});

// ------------------------------------------------------------------ turns
function addChat(speaker, text, note = '') {
  const li = document.createElement('li');
  li.className = speaker;
  li.innerHTML = `<small>${speaker}${note ? ' · ' + esc(note) : ''}</small>${esc(text)}`;
  $('#chat').appendChild(li); li.scrollIntoView({ block: 'nearest' });
}

async function sendPartner({ blob, text }) {
  const fd = new FormData();
  fd.append('session_id', state.session); fd.append('language', $('#language').value);
  fd.append('use_llm', $('#useLLM').checked);
  if (text) fd.append('text', text);
  if (blob) fd.append('audio', blob, 'partner.wav');
  try {
    const r = await api('/api/partner', { method: 'POST', body: fd });
    if (!r.transcript) { toast(r.note || 'No transcript', 5000); return; }
    state.lastPartnerTurn = Math.max(state.lastPartnerTurn, r.turn_id || 0);
    addChat('partner', r.transcript, r.stt?.cached ? 'cached STT' : '');
    showCandidates(r.suggestions, { ranker: 'reply suggestions', latency: r.latency, question: r.question });
  } catch (e) { toast('Partner turn failed: ' + e.message, 5000); }
}

async function sendPatient(blob, cues) {
  const fd = new FormData();
  fd.append('session_id', state.session); fd.append('language', $('#language').value);
  fd.append('cues', JSON.stringify(cues || state.vision?.snapshot() || {}));
  fd.append('spelled', state.spelled.trim());
  fd.append('sign_phrase_ids', JSON.stringify(state.signPhrases));
  fd.append('use_llm', $('#useLLM').checked);
  fd.append('use_scribe', $('#useScribe').checked);
  fd.append('use_spectrogram', $('#useSpectro').checked);
  if (blob) fd.append('audio', blob, 'patient.wav');
  $('#rankerInfo').textContent = 'thinking…';
  try {
    const r = await api('/api/patient', { method: 'POST', body: fd });
    state.lastResult = r;
    showCandidates(r.candidates, { ranker: r.ranker, latency: r.latency, transcript: r.transcript,
      note: r.stt?.skipped ? 'Scribe skipped: ' + r.stt.skipped : r.stt?.cached === 'fuzzy' ? 'transcript re-used'
        : r.stt?.budget === 'exhausted' ? 'Scribe credits used up' : '' });
    if (r.stt?.budget === 'exhausted') loadStatus();
    showSpectrogramCompare(r);
  } catch (e) { toast('Patient turn failed: ' + e.message, 5000); $('#rankerInfo').textContent = ''; }
}

async function showSpectrogramCompare(r) {
  if (!r.spectrogram?.length) { $('#specCompare').hidden = true; return; }
  $('#specCompare').hidden = false;
  drawMatrix($('#specInput'), r.spectrogram);
  const best = r.spectrogram_matches?.[0];
  if (best) {
    $('#specMatchLabel').textContent = `best template: “${best.text}” (${(best.confidence * 100) | 0}%)`;
    const t = await api('/api/templates/' + best.phrase_id);
    drawMatrix($('#specMatch'), t[0]?.spectrogram);
  } else { $('#specMatchLabel').textContent = 'no templates enrolled'; drawMatrix($('#specMatch'), []); }
}

function showCandidates(cands, meta) {
  state.candidates = cands || []; state.selected = 0; state.shownAt = performance.now(); state.actions = 0;
  const box = $('#candidates');
  // auto-scan: the highlight steps through the suggestions; one key or switch press picks it
  clearInterval(state.scanTimer);
  box.classList.toggle('scanning', profile.yes === 'scan' && state.candidates.length > 1);
  if (profile.yes === 'scan' && state.candidates.length > 1) {
    state.scanTimer = setInterval(() => select((state.selected + 1) % state.candidates.length), 2000);
  }
  // yes/no and either/or questions: the first two cards are the answers, shown side by side
  box.classList.toggle('binary', ['yes_no', 'choice'].includes(meta.question?.type) && state.candidates.length >= 2);
  $('#rankerInfo').textContent = meta.ranker ? `ranker: ${meta.ranker}` + (meta.transcript ? ` · heard: “${meta.transcript}”` : '')
    + (meta.note ? ` · ${meta.note}` : '') : '';
  if (!state.candidates.length) { box.innerHTML = `<p class="empty">${meta.ranker ? 'No suggestions. Try again or type the message below.' : emptyHint()}</p>`; }
  else {
    box.innerHTML = state.candidates.map((c, i) => {
      const p = state.phrases.find((x) => x.id === c.phrase_id);
      return `<button class="cand ${i === 0 ? 'selected' : ''} ${p?.is_emergency ? 'emergency' : ''}" data-i="${i}">
        <span class="num">${i + 1}</span><span class="text">${esc(c.text)}</span>
        <span class="conf">${Math.round((c.confidence || 0) * 100)}%</span>
        <span class="meta">${(c.sources || []).map((s) => `<span class="chip">${esc(s)}</span>`).join('')}
          ${c.emotion && c.emotion !== 'neutral' ? `<span class="chip">${esc(c.emotion)}</span>` : ''}</span>
        ${c.rationale ? `<span class="why">${esc(c.rationale)}</span>` : ''}
        <span class="bar"><i style="width:${Math.round((c.confidence || 0) * 100)}%"></i></span>
      </button>`;
    }).join('');
    box.querySelectorAll('.cand').forEach((b) => b.onclick = () => { state.actions++; confirmCandidate(+b.dataset.i); });
  }
  const lat = meta.latency || {};
  $('#latency').innerHTML = Object.entries(lat).map(([k, v]) =>
    `<span class="chip ${k === 'total_ms' ? (v < 2000 ? 'on' : 'danger') : ''}">${k.replace('_ms', '')}: ${Math.round(v)} ms</span>`).join('');
}

function select(i) {
  state.selected = i;
  document.querySelectorAll('.cand').forEach((b, j) => b.classList.toggle('selected', j === i));
}

async function confirmCandidate(i, customText) {
  const c = customText ? { text: customText, sources: ['typed'], phrase_id: null } : state.candidates[i];
  if (!c) return;
  const elapsed = performance.now() - state.shownAt;
  const rank = customText ? -1 : i;
  const raw = state.lastResult?.transcript || '';
  clearInterval(state.scanTimer); $('#candidates').classList.remove('scanning', 'binary');
  state.candidates = [];
  learnSigns(c);
  $('#candidates').innerHTML = `<p class="empty">Speaking: “${esc(c.text)}”</p>`;
  addChat('patient', c.text, (c.sources || []).join(', '));
  state.spelled = ''; state.signPhrases = []; state.signLog = []; renderSpell();
  api('/api/confirm', { json: { session_id: state.session, text: c.text, phrase_id: c.phrase_id, rank,
    actions: Math.max(1, state.actions), time_to_confirm_ms: elapsed, sources: c.sources || [], raw_text: raw,
    ranker: $('#rankerInfo').textContent } }).catch((e) => console.warn(e));
  await speak(c.text, c.language);
}

$('#customForm').onsubmit = (e) => {
  e.preventDefault();
  const t = $('#customText').value.trim(); if (!t) return;
  $('#customText').value = ''; confirmCandidate(-1, t);
};

async function speak(text, language) {
  const t0 = performance.now();
  try {
    const res = await fetch('/api/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, session_id: state.session }) });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    if ((res.headers.get('content-type') || '').includes('json')) return browserSpeak(text, language);
    const url = URL.createObjectURL(await res.blob());
    const audio = new Audio(url);
    audio.onplaying = () => { $('#latency').insertAdjacentHTML('beforeend',
      `<span class="chip">tts ${res.headers.get('X-Cache') === 'hit' ? '(cached)' : ''}: ${Math.round(performance.now() - t0)} ms</span>`); };
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  } catch (e) {
    toast('ElevenLabs TTS failed (' + e.message + '); using browser voice', 4000);
    browserSpeak(text, language);
  }
}

function browserSpeak(text, language) {
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = language === 'hi' || /[ऀ-ॿ]/.test(text) || $('#language').value === 'hin' ? 'hi-IN' : 'en-IN';
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}

$('#newSession').onclick = () => { state.session = newSessionId(); $('#chat').innerHTML = ''; showCandidates([], {}); toast('New session'); };

// ------------------------------------------------------------------ signs tab
const signCapture = {
  active: false, buf: [], frame: 0,
  push(f) { if (this.frame++ % 2 === 0) this.buf.push(f); },
};

function populateSignLabels() {
  const kind = $('#signKind').value, sel = $('#signLabel');
  if (kind === 'letter') sel.innerHTML = LETTERS.map((l) => `<option>${l}</option>`).join('');
  else if (kind === 'control') sel.innerHTML = CONTROLS.map((l) => `<option>${l}</option>`).join('');
  else sel.innerHTML = state.phrases.map((p) => `<option value="${p.id}">${esc(p.text)}</option>`).join('');
}
$('#signKind').onchange = populateSignLabels;

$('#recordSign').onclick = async () => {
  if (!state.vision) { toast('Start the camera on the Talk tab first'); return; }
  const kind = $('#signKind').value, raw = $('#signLabel').value;
  const label = kind === 'phrase' ? state.phrases.find((p) => p.id === +raw)?.text : raw;
  const btn = $('#recordSign'); btn.disabled = true;
  for (let n = 3; n > 0; n--) { $('#signRecStatus').textContent = `Hold “${label}” in ${n}…`; await sleep(1000); }
  signCapture.buf = []; signCapture.frame = 0; signCapture.active = true;
  $('#signRecStatus').textContent = 'Capturing… keep holding';
  await sleep(2500);
  signCapture.active = false;
  const samples = signCapture.buf.slice(0, 40);
  btn.disabled = false;
  if (samples.length < 8) { $('#signRecStatus').textContent = 'Hands not detected. Try better lighting.'; return; }
  await api('/api/signs', { json: { label, kind, phrase_id: kind === 'phrase' ? +raw : null, samples } });
  $('#signRecStatus').textContent = `Saved ${samples.length} samples for “${label}”`;
  await reloadClassifier(); loadSigns();
};

async function reloadClassifier() { classifier.load(await api('/api/signs')); }

async function loadSigns() {
  populateSignLabels();
  const rows = await api('/api/signs/summary');
  $('#signCount').textContent = `${rows.length} signs · ${rows.reduce((a, r) => a + r.n, 0)} samples`;
  $('#signTable tbody').innerHTML = rows.map((r) => `<tr><td><strong>${esc(r.label)}</strong></td><td>${r.kind}</td><td>${r.n}</td>
    <td><button class="btn ghost danger" data-del="${esc(r.label)}">Delete</button></td></tr>`).join('')
    || '<tr><td colspan="4" class="hint">No signs trained yet</td></tr>';
  document.querySelectorAll('[data-del]').forEach((b) => b.onclick = async () => {
    await api('/api/signs/' + encodeURIComponent(b.dataset.del), { method: 'DELETE' });
    await reloadClassifier(); loadSigns();
  });
}

// ------------------------------------------------------------------ spectrogram tab
async function loadTemplates() {
  await loadPhraseCache();
  const sel = $('#tmplPhrase'), keep = sel.value;
  sel.innerHTML = state.phrases.map((p) => `<option value="${p.id}">${esc(p.text)} (${p.n_templates})</option>`).join('');
  if (keep) sel.value = keep;
  $('#tmplTable tbody').innerHTML = state.phrases.map((p) =>
    `<tr><td>${esc(p.text)}</td><td>${p.language}</td><td>${'●'.repeat(p.n_templates) || '<span class="hint">none</span>'}</td></tr>`).join('');
}

$('#tmplRecord').onclick = () => toggleRecord('tmpl', $('#tmplRecord'), async (blob) => {
  const fd = new FormData(); fd.append('phrase_id', $('#tmplPhrase').value); fd.append('audio', blob, 'tmpl.wav');
  try {
    const r = await api('/api/templates', { method: 'POST', body: fd });
    drawMatrix($('#tmplSpec'), r.spectrogram);
    $('#tmplResult').textContent = `Saved template (${r.frames} frames).`;
    loadTemplates(); loadStatus();
  } catch (e) { toast(e.message, 5000); }
});

$('#tmplTest').onclick = () => toggleRecord('test', $('#tmplTest'), async (blob) => {
  const fd = new FormData();
  fd.append('session_id', state.session + '-test'); fd.append('use_llm', false); fd.append('use_scribe', false);
  fd.append('audio', blob, 'test.wav');
  const r = await api('/api/patient', { method: 'POST', body: fd });
  drawMatrix($('#tmplSpec'), r.spectrogram);
  $('#tmplResult').innerHTML = r.spectrogram_matches.length
    ? 'Matches: ' + r.spectrogram_matches.map((m) => `<span class="chip">${esc(m.text)} ${(m.confidence * 100) | 0}% · d=${m.distance}</span>`).join(' ')
      + ` <span class="hint">(${Math.round(r.latency.spectrogram_ms || 0)} ms)</span>`
    : 'No templates enrolled yet.';
});

// ------------------------------------------------------------------ voice & consent tab
async function loadVoiceTab() {
  const v = await api('/api/voices');
  $('#currentVoice').textContent = v.selected;
  const items = [{ voice_id: v.default, name: 'Default voice', kind: 'default' }, ...v.voices];
  const premade = v.premade.filter((p) => p.voice_id !== v.default);
  $('#premadeList').innerHTML = premade.map((p) => `<div class="voice-item"><span><strong>${esc(p.name)}</strong>
    <span class="hint">${esc(p.labels)}</span></span><span class="row" style="margin:0">
    ${p.preview_url ? `<button class="btn text" data-preview="${esc(p.preview_url)}" aria-label="Play a sample of ${esc(p.name)}"><span class="icon" aria-hidden="true">play_arrow</span></button>` : ''}
    <button class="btn" data-voice="${esc(p.voice_id)}" ${p.voice_id === v.selected ? 'disabled' : ''}>${p.voice_id === v.selected ? 'In use' : 'Use'}</button></span></div>`).join('')
    || '<p class="hint">Add an ElevenLabs key to see the ready-made voices.</p>';
  document.querySelectorAll('[data-preview]').forEach((b) => b.onclick = () => new Audio(b.dataset.preview).play().catch(() => toast('Could not play the sample')));
  // Paid-only features are shown but switched off, with the reason, instead of failing later.
  const freeNote = 'Needs a paid ElevenLabs plan (the free tier can only use the ready-made voices above).';
  $('#designNote').textContent = v.plan.can_design ? 'No recordings needed.' : freeNote;
  $('#designForm').querySelectorAll('textarea, button').forEach((el) => { el.disabled = !v.plan.can_design; });
  $('#cloneNote').hidden = v.plan.can_clone;
  $('#cloneNote').textContent = 'Copying a voice ' + freeNote.charAt(0).toLowerCase() + freeNote.slice(1)
    + ' You can still record consent now.';
  $('#voiceList').innerHTML = items.map((x) => `<div class="voice-item"><span><strong>${esc(x.name)}</strong>
    <span class="chip">${x.kind}</span>${x.voice_owner ? ` <span class="hint">consent: ${esc(x.voice_owner)}</span>` : ''}</span>
    <button class="btn" data-voice="${esc(x.voice_id)}" ${x.voice_id === v.selected ? 'disabled' : ''}>${x.voice_id === v.selected ? 'In use' : 'Use'}</button></div>`).join('');
  document.querySelectorAll('[data-voice]').forEach((b) => b.onclick = async () => {
    try { await api('/api/voices/select', { json: { voice_id: b.dataset.voice } }); loadVoiceTab(); } catch (e) { toast(e.message); }
  });
  const consents = await api('/api/consent');
  $('#consentList').innerHTML = consents.map((c) => `<div class="voice-item"><span>#${c.id} <strong>${esc(c.voice_owner)}</strong>
    <span class="hint">signed by ${esc(c.signer)} (${esc(c.relationship)}) · ${new Date(c.created_at * 1000).toLocaleString()}</span>
    ${c.revoked_at ? '<span class="chip danger">revoked</span>' : '<span class="chip on">active</span>'}</span>
    ${c.revoked_at ? '' : `<button class="btn danger" data-revoke="${c.id}">Revoke</button>`}</div>`).join('') || '<p class="hint">No consent records.</p>';
  document.querySelectorAll('[data-revoke]').forEach((b) => b.onclick = async () => {
    if (!confirm('Revoke consent? Any voice cloned under it will be deleted.')) return;
    const r = await api(`/api/consent/${b.dataset.revoke}/revoke`, { method: 'POST' });
    toast(`Consent revoked; deleted ${r.deleted_voices.length} voice(s)`); loadVoiceTab();
  });
  $('#cloneConsent').innerHTML = consents.filter((c) => !c.revoked_at)
    .map((c) => `<option value="${c.id}">#${c.id}: ${esc(c.voice_owner)}</option>`).join('') || '<option value="">No active consent</option>';
  const audit = await api('/api/audit');
  $('#auditLog').innerHTML = audit.map((e) => `<li>${new Date(e.created_at * 1000).toLocaleString()}: ${esc(e.data.action)}
    ${e.data.consent_id ? '#' + e.data.consent_id : ''} ${e.data.voice_id ? esc(e.data.voice_id) : ''}</li>`).join('');
  updateStatement();
}

// One-screen consent: the owner signs for themself, or a family member / guardian signs for them.
// Pressing "I agree" after reading the statement is the agreement; the signer's name is typed once.
function consentFields() {
  const self = document.querySelector('input[name="cWho"]:checked').value === 'self';
  const owner = $('#cOwner').value.trim();
  return { owner, self, signer: self ? owner : $('#cSigner').value.trim(), relationship: self ? 'self' : $('#cRel').value };
}

async function updateStatement() {
  const f = consentFields();
  $('#cOtherFields').hidden = f.self;
  $('#cSigner').required = !f.self;
  const q = new URLSearchParams({ owner: f.owner || '[name]', signer: f.signer || (f.self ? '[name]' : '[your name]'),
    relationship: f.relationship, purpose: $('#cPurpose').value });
  try { $('#cStatement').textContent = (await api('/api/consent/statement?' + q)).statement; } catch { /* shown on submit */ }
}
['#cOwner', '#cSigner', '#cRel'].forEach((s) => $(s).addEventListener('input', updateStatement));
document.querySelectorAll('input[name="cWho"]').forEach((r) => r.addEventListener('change', updateStatement));

$('#consentForm').onsubmit = async (e) => {
  e.preventDefault();
  const f = consentFields();
  const fd = new FormData();
  fd.append('voice_owner', f.owner); fd.append('signer', f.signer);
  fd.append('relationship', f.relationship); fd.append('purpose', $('#cPurpose').value);
  fd.append('typed_signature', f.signer); fd.append('agreed', true);
  try { await api('/api/consent', { method: 'POST', body: fd }); toast('Consent saved'); e.target.reset(); await loadVoiceTab(); }
  catch (err) { toast('Consent not saved: ' + err.message, 5000); }
};

$('#cloneForm').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append('consent_id', $('#cloneConsent').value); fd.append('name', $('#cloneName').value);
  for (const f of $('#cloneFiles').files) fd.append('files', f);
  try { await api('/api/voices/clone', { method: 'POST', body: fd }); toast('Voice cloned and selected'); loadVoiceTab(); }
  catch (err) { toast('Clone failed: ' + err.message, 6000); }
};

$('#designForm').onsubmit = async (e) => {
  e.preventDefault();
  const description = $('#designDesc').value.trim();
  $('#designPreviews').innerHTML = '<p class="hint">Generating…</p>';
  try {
    const r = await api('/api/voices/design', { json: { description } });
    $('#designPreviews').innerHTML = r.previews.map((p, i) => `<div class="voice-item">
      <audio controls src="data:${p.media_type};base64,${p.audio_base64}"></audio>
      <button class="btn primary" data-gen="${esc(p.generated_voice_id)}" data-key="${esc(p.key_id || '')}">Use preview ${i + 1}</button></div>`).join('');
    document.querySelectorAll('[data-gen]').forEach((b) => b.onclick = async () => {
      await api('/api/voices/design/save', { json: { generated_voice_id: b.dataset.gen, name: 'Designed voice', description,
        key_id: b.dataset.key || null } });
      toast('Designed voice saved and selected'); $('#designPreviews').innerHTML = ''; loadVoiceTab();
    });
  } catch (err) { $('#designPreviews').innerHTML = ''; toast('Voice Design failed: ' + err.message, 6000); }
};

// ------------------------------------------------------------------ phrasebook tab
async function loadPhrases() {
  await loadPhraseCache();
  const chars = state.phrases.reduce((a, p) => a + p.text.length, 0);
  $('#precacheInfo').textContent = `${state.phrases.length} phrases, ${chars} characters in total. Audio is generated once per phrase and voice, then stored in the database; replays cost no credits.`;
  $('#phraseTable tbody').innerHTML = state.phrases.map((p) => `<tr><td>${p.is_emergency ? '🚨 ' : ''}${esc(p.text)}
    <div class="hint">${esc(p.aliases.join(', '))}</div></td><td>${p.language}</td><td>${esc(p.category)}</td><td>${p.use_count}</td>
    <td><button class="btn ghost" data-say="${p.id}">▶</button><button class="btn ghost danger" data-delp="${p.id}">Delete</button></td></tr>`).join('');
  document.querySelectorAll('[data-say]').forEach((b) => b.onclick = () => {
    const p = state.phrases.find((x) => x.id === +b.dataset.say); speak(p.text, p.language);
  });
  document.querySelectorAll('[data-delp]').forEach((b) => b.onclick = async () => {
    await api('/api/phrases/' + b.dataset.delp, { method: 'DELETE' }); loadPhrases();
  });
}

$('#phraseForm').onsubmit = async (e) => {
  e.preventDefault();
  const split = (s) => s.split(',').map((x) => x.trim()).filter(Boolean);
  await api('/api/phrases', { json: { text: $('#pText').value, language: $('#pLang').value, category: $('#pCat').value || 'general',
    aliases: split($('#pAliases').value), keywords: split($('#pKeywords').value), is_emergency: $('#pEmergency').checked } });
  e.target.reset(); loadPhrases();
};

$('#precache').onclick = async () => {
  if (!confirm('Generate ElevenLabs audio for every phrase that is not cached yet? This uses credits once.')) return;
  try {
    const r = await api('/api/phrases/precache', { method: 'POST' });
    toast(r.mock ? 'Mock mode: no ElevenLabs key configured' : `Generated ${r.generated}, already cached ${r.already_cached}`, 5000);
    loadStatus();
  } catch (e) { toast('Pre-generation failed: ' + e.message, 6000); }
};

// ------------------------------------------------------------------ metrics tab
async function loadMetrics() {
  const m = await api('/api/metrics');
  const kv = (o) => `<dl class="kv">${Object.entries(o).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</dd>`).join('')}</dl>`;
  $('#mLatency').innerHTML = Object.entries(m.latency).map(([kind, stages]) => `<h4>${kind}</h4><div class="table-wrap"><table class="table">
    <tr><th>stage</th><th>n</th><th>p50</th><th>p95</th></tr>${Object.entries(stages).map(([s, v]) =>
      `<tr><td>${s.replace('_ms', '')}</td><td>${v.n}</td><td>${v.p50}</td><td>${v.p95}</td></tr>`).join('')}</table></div>`).join('')
    || '<p class="hint">No turns yet.</p>';
  $('#mEffort').innerHTML = Object.keys(m.effort).length ? kv({
    'confirmed messages': m.effort.n, 'mean actions to confirm': m.effort.mean_actions,
    'median time to confirm (ms)': m.effort.median_time_ms, 'top suggestion accepted': (m.effort.top1_rate * 100).toFixed(1) + '%',
    'typed instead': (m.effort.custom_rate * 100).toFixed(1) + '%' }) : '<p class="hint">No confirmations yet.</p>';
  $('#mCache').innerHTML = kv({ 'TTS clips': m.cache.tts.n, 'TTS replays served from cache': m.cache.tts.hits,
    'TTS cache size (KB)': Math.round(m.cache.tts.bytes / 1024), 'transcripts': m.cache.stt.n, 'STT cache hits': m.cache.stt.hits });
  $('#mCredits').innerHTML = kv({ 'TTS characters sent': m.credits.tts_chars, 'STT minutes sent': m.credits.stt_minutes,
    'STT minutes saved (re-used, skipped, trimmed)': m.credits.stt_minutes_saved, 'Scribe calls skipped': m.credits.stt_calls_skipped,
    'Voice Design calls': m.credits.voice_designs })
    + (m.budget.keys.length ? `<h4>API keys</h4><div class="table-wrap"><table class="table"><tr><th>key</th><th>used</th><th>limit</th><th>left</th><th></th></tr>
      ${m.budget.keys.map((k) => `<tr><td><code>${esc(k.key_id)}</code></td><td>${k.used}</td><td>${k.limit}${k.synced ? '' : ' (est.)'}</td>
        <td>${k.remaining}</td><td>${k.exhausted ? '<span class="chip danger">used up</span>' : ''}</td></tr>`).join('')}</table></div>
      ${m.budget.saver_mode ? '<p class="hint">Saver mode: Scribe is skipped whenever a recorded phrase matches clearly.</p>' : ''}` : '');
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ learn signs from confirmed messages
// When a confirmed message used fingerspelling or a phrase sign, the hand shapes seen for those signs
// become extra training samples (marked "learned"; the server keeps only the newest 40 per sign).
function learnSigns(c) {
  const used = (c.sources || []).some((s) => s.startsWith('isl-spelling') || s === 'sign');
  const log = state.signLog; state.signLog = [];
  if (!used || !log.length) return;
  const groups = {};
  for (const e of log) (groups[e.label] ||= { ...e, samples: [] }).samples.push(e.features);
  Promise.all(Object.values(groups).map((g) => api('/api/signs', { json: {
    label: g.label, kind: g.kind, phrase_id: g.phrase_id, samples: g.samples, learned: true } })))
    .then(reloadClassifier).catch((e) => console.warn('sign learning failed', e));
}

// ------------------------------------------------------------------ emergency
let emergencyAt = 0;
async function triggerEmergency(trigger) {
  if (performance.now() - emergencyAt < 10000 && !$('#emergency').hidden) return;   // already showing
  emergencyAt = performance.now();
  const fd = new FormData(); fd.append('session_id', state.session); fd.append('trigger', trigger);
  let text = 'I need help now';
  try { text = (await api('/api/emergency', { method: 'POST', body: fd })).text; } catch { /* still alert */ }
  $('#emergencyText').textContent = text;
  $('#emergency').hidden = false; $('#emergencyDismiss').focus();
  addChat('patient', text, 'emergency · ' + trigger);
  speak(text);
}
$('#helpBtn').onclick = () => triggerEmergency('button');
$('#emergencyDismiss').onclick = () => { $('#emergency').hidden = true; };

// ------------------------------------------------------------------ partner phone
async function openPartnerPhone() {
  const { addresses } = await api('/api/lan').catch(() => ({ addresses: [] }));
  const host = location.hostname === 'localhost' || location.hostname === '127.0.0.1' ? addresses[0] : location.hostname;
  const { key } = await api(`/api/partner/link?session_id=${encodeURIComponent(state.session)}`);
  const url = `${location.protocol}//${host || location.hostname}:${location.port || (location.protocol === 'https:' ? 443 : 80)}`
    + `/partner.html?s=${encodeURIComponent(state.session)}&k=${encodeURIComponent(key)}`;
  $('#partnerUrl').textContent = url;
  $('#partnerNote').textContent = (!host ? 'No network address found: start the server with --host 0.0.0.0. ' : '')
    + (location.protocol !== 'https:' ? 'Over plain http the phone can show captions and take typed words; its microphone needs https.' : '');
  $('#partnerQr').innerHTML = '';
  try {
    if (!window.QRCode) await loadScript('https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js');
    new window.QRCode($('#partnerQr'), { text: url, width: 200, height: 200 });
  } catch { $('#partnerQr').hidden = true; }
  $('#partnerDialog').showModal();
  startPartnerFeed();
}
function loadScript(src) {
  return new Promise((ok, fail) => { const el = document.createElement('script'); el.src = src; el.onload = ok; el.onerror = fail; document.head.appendChild(el); });
}
// Turns spoken on the partner's phone arrive here: poll the server for the session's latest partner turn.
function startPartnerFeed() {
  if (state.partnerPoll) return;
  state.partnerPoll = setInterval(async () => {
    try {
      const r = await api(`/api/partner/latest?session_id=${encodeURIComponent(state.session)}&after=${state.lastPartnerTurn}`);
      if (!r.turn_id) return;
      state.lastPartnerTurn = r.turn_id;
      addChat('partner', r.transcript, 'partner phone');
      showCandidates(r.suggestions, { ranker: 'reply suggestions', latency: r.latency, question: r.question });
    } catch { /* server restarting; try again next tick */ }
  }, 1500);
}
$('#partnerPhone').onclick = openPartnerPhone;
$('#closePartner').onclick = () => $('#partnerDialog').close();
$('#copyPartnerUrl').onclick = () => navigator.clipboard?.writeText($('#partnerUrl').textContent)
  .then(() => toast('Link copied')).catch(() => toast('Copy the link from the box'));

// ------------------------------------------------------------------ first-run setup
function emptyHint() {
  const how = { nod: 'nod to choose, shake your head for the next one',
    thumbs: 'hold a thumbs up to choose, thumbs down for the next one',
    tap: 'click one, or press 1–3 or Enter',
    scan: 'press Space, Enter or your switch when the right one is highlighted' }[profile.yes];
  return `Speak, sign or gesture. Suggestions appear here. Then ${how}.`;
}

function applyProfile() {
  $('#language').value = profile.lang;
  $('#greeting').textContent = profile.name ? `Hello, ${profile.name}` : 'Conversation assistant';
  if (!state.candidates.length) $('#candidates').innerHTML = `<p class="empty">${emptyHint()}</p>`;
}

const setup = { step: 0, last: 3 };
function showStep(n) {
  setup.step = n;
  document.querySelectorAll('#setup .step').forEach((s) => { s.hidden = +s.dataset.step !== n; });
  document.querySelectorAll('#setup .steps i').forEach((d, i) => d.classList.toggle('on', i <= n));
  $('#setupBack').hidden = n === 0;
  $('#setupNext').textContent = n === setup.last ? 'Start talking' : 'Next';
}
function openSetup() {
  $('#sName').value = profile.name;
  for (const [name, v] of [['sLang', profile.lang], ['sYes', profile.yes], ['sVoice', profile.voice]]) {
    const r = document.querySelector(`input[name="${name}"][value="${v}"]`); if (r) r.checked = true;
  }
  showStep(0); $('#setup').showModal();
  loadIntroPrompts();
}
async function finishSetup(skipped) {
  const pick = (name) => document.querySelector(`input[name="${name}"]:checked`)?.value;
  if (!skipped) {
    profile.name = $('#sName').value.trim();
    profile.lang = pick('sLang') || 'auto'; profile.yes = pick('sYes') || 'thumbs'; profile.voice = pick('sVoice') || 'default';
  }
  profile.done = true; saveProfile(); applyProfile();
  $('#setup').close();
  if (skipped) return;
  if (profile.voice !== 'default') {           // voice choices continue on the Voice tab
    document.querySelector('.tabs [data-tab="voice"]').click();
    if (profile.voice === 'own') { $('#consentCard').scrollIntoView({ block: 'start' }); $('#cOwner').value = profile.name; updateStatement(); }
    else $('#designDesc').focus();
  } else if (!recorder.ctx) startMedia();     // this click is the user gesture getUserMedia needs
}
$('#setupNext').onclick = () => (setup.step === setup.last ? finishSetup(false) : showStep(setup.step + 1));
$('#setupBack').onclick = () => showStep(Math.max(0, setup.step - 1));
$('#setupSkip').onclick = () => finishSetup(true);
$('#openSetup').onclick = openSetup;

// ------------------------------------------------------------------ guided voice intro (setup step 3)
// Part 1: free speech about themselves (voice profile + names). Part 2: the prompt words with pauses,
// split by the server into one template per phrase. Both are recorded here and sent together.
const intro = { rec: new Recorder(), prompts: [], about: null, words: null, active: null };

async function loadIntroPrompts() {
  try {
    const lang = document.querySelector('input[name="sLang"]:checked')?.value === 'hin' ? 'hi' : 'en';
    const r = await api(`/api/intro/prompts?language=${lang}`);
    intro.prompts = r.prompts;
    $('#introWords').innerHTML = r.prompts.map((p) => `<span>${esc(p.say)}</span>`).join('');
  } catch { /* shown as an empty list; the step can be skipped */ }
}
async function toggleIntroRecording(which) {
  const btn = which === 'about' ? $('#introAbout') : $('#introWordsRec');
  const note = which === 'about' ? $('#introAboutState') : $('#introWordsState');
  if (intro.active === which) {          // stop
    const { blob, seconds } = intro.rec.stop();
    intro.active = null;
    btn.setAttribute('aria-pressed', 'false'); btn.lastElementChild.textContent = 'Record again';
    intro[which] = seconds >= 1 ? blob : null;
    note.textContent = seconds >= 1 ? `Recorded ${seconds.toFixed(0)} s` : 'Too short, try again';
    $('#introSave').disabled = !intro.words;
    return;
  }
  if (intro.active) return;              // one recording at a time
  try {
    if (!intro.rec.ctx) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      await intro.rec.init(stream);
    }
  } catch (e) { toast('Microphone not available: ' + e.message, 5000); return; }
  intro.rec.start(); intro.active = which;
  btn.setAttribute('aria-pressed', 'true'); btn.lastElementChild.textContent = 'Stop';
  note.textContent = 'Listening…';
}
$('#introAbout').onclick = () => toggleIntroRecording('about');
$('#introWordsRec').onclick = () => toggleIntroRecording('words');
$('#introSave').onclick = async () => {
  const fd = new FormData();
  fd.append('words', intro.words, 'words.wav');
  fd.append('phrase_ids', JSON.stringify(intro.prompts.map((p) => p.phrase_id)));
  fd.append('language', document.querySelector('input[name="sLang"]:checked')?.value || 'auto');
  if (intro.about) fd.append('about', intro.about, 'about.wav');
  $('#introSave').disabled = true; $('#introResult').innerHTML = '<p class="hint">Learning your voice…</p>';
  try {
    const r = await api('/api/intro', { method: 'POST', body: fd });
    const p = r.profile;
    $('#introResult').innerHTML = `<p><b>Learned ${r.saved.length} words.</b> Clio can recognise them from now on.</p>`
      + (p ? `<p class="hint">Your voice: ${p.pitch_hz ? `pitch about ${p.pitch_hz} Hz, ` : ''}${p.syllables_per_s ? `${p.syllables_per_s} syllables a second, ` : ''}background noise ${p.snr_db != null ? `${p.snr_db} dB below your voice` : 'unknown'}.</p>` : '')
      + (r.suggestions.length ? `<p class="hint">Add to your phrases?</p><div class="row wrap">${r.suggestions.map((t) => `<button type="button" class="btn tonal" data-add="${esc(t)}">${esc(t)}</button>`).join('')}</div>` : '');
    $('#introResult').querySelectorAll('[data-add]').forEach((b) => {
      b.onclick = async () => {
        await api('/api/phrases', { json: { text: b.dataset.add, category: 'people' } });
        b.disabled = true; b.textContent = 'Added: ' + b.dataset.add; loadPhraseCache();
      };
    });
    loadStatus();
  } catch (e) {
    $('#introResult').innerHTML = `<p class="hint">${esc(e.message)}</p>`;
  } finally { $('#introSave').disabled = !intro.words; }
};

// ------------------------------------------------------------------ account
$('#signOut').onclick = async () => {
  await fetch('/api/auth/logout', { method: 'POST' });
  location.replace('/login.html');
};

// ------------------------------------------------------------------ boot
(async () => {
  let me;
  try { me = await api('/api/auth/me'); } catch { return; }   // 401 -> login page
  loadProfile(me.user);
  $('#accountName').textContent = me.user.name;
  applyProfile();
  startPartnerFeed();   // picks up turns spoken on the partner's phone
  if (!profile.done) openSetup();
  try {
    await Promise.all([loadStatus(), loadPhraseCache(), reloadClassifier()]);
    populateSignLabels();
  } catch (e) { toast('Backend not reachable: ' + e.message, 6000); }
})();
