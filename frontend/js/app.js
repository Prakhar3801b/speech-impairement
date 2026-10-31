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
};
function newSessionId() {
  const id = 's-' + Math.random().toString(36).slice(2, 10);
  try { localStorage.setItem('ps16.session', id); } catch { /* storage unavailable */ }
  return id;
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
  if (!recorder.ctx) { toast('Start the camera & mic first (Converse tab)'); return false; }
  return true;
}

// ------------------------------------------------------------------ vision frame handling
const stabilizer = new SignStabilizer(onSignEntered);
let thumbHold = { dir: null, n: 0, fired: false };

function onVisionFrame(s) {
  $('#cueExpr').textContent = 'face: ' + s.expression;
  $('#cueExpr').classList.toggle('danger', s.expression === 'pain');
  $('#cueHead').textContent = 'head: ' + (s.head || '–');
  $('#cueHead').classList.toggle('on', !!s.head);
  $('#cueThumb').textContent = 'thumb: ' + (s.thumbs || '–');
  $('#cueThumb').classList.toggle('on', !!s.thumbs);
  if (state.vision?.utter) $('#cueSyl').textContent = 'syllables: ' + state.vision.utter.syllables;

  // thumbs: confirm / next while suggestions are on screen
  if (state.candidates.length && s.thumbs) {
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
  if ($('#signEnabled').checked && !signCapture.active) stabilizer.push(pred);
  if (signCapture.active && s.features) signCapture.push(s.features);
}

function onSignEntered(pred) {
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
$('#spellClear').onclick = () => { state.spelled = ''; state.signPhrases = []; renderSpell(); };
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
  else if (k === 'enter' && state.candidates.length) { e.preventDefault(); state.actions++; confirmCandidate(state.selected); }
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
    addChat('partner', r.transcript, r.stt?.cached ? 'cached STT' : '');
    showCandidates(r.suggestions, { ranker: 'reply suggestions', latency: r.latency });
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
    showCandidates(r.candidates, { ranker: r.ranker, latency: r.latency, transcript: r.transcript });
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
  $('#rankerInfo').textContent = meta.ranker ? `ranker: ${meta.ranker}` + (meta.transcript ? ` · heard: “${meta.transcript}”` : '') : '';
  if (!state.candidates.length) { box.innerHTML = '<p class="empty">No suggestions. Try again or type the message below.</p>'; }
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
  state.candidates = [];
  $('#candidates').innerHTML = `<p class="empty">Speaking: “${esc(c.text)}”</p>`;
  addChat('patient', c.text, (c.sources || []).join(', '));
  state.spelled = ''; state.signPhrases = []; renderSpell();
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
  if (!state.vision) { toast('Start the camera on the Converse tab first'); return; }
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
  const items = [{ voice_id: v.default, name: 'Default library voice', kind: 'default' }, ...v.voices];
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

async function updateStatement() {
  const q = new URLSearchParams({ owner: $('#cOwner').value || '[owner]', signer: $('#cSigner').value || '[signer]',
    relationship: $('#cRel').value, purpose: $('#cPurpose').value || '[purpose]' });
  $('#cStatement').textContent = (await api('/api/consent/statement?' + q)).statement;
}
['#cOwner', '#cSigner', '#cRel', '#cPurpose'].forEach((s) => $(s).addEventListener('input', updateStatement));

$('#consentForm').onsubmit = async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append('voice_owner', $('#cOwner').value); fd.append('signer', $('#cSigner').value);
  fd.append('relationship', $('#cRel').value); fd.append('purpose', $('#cPurpose').value);
  fd.append('typed_signature', $('#cSignature').value); fd.append('agreed', $('#cAgree').checked);
  try { await api('/api/consent', { method: 'POST', body: fd }); toast('Consent recorded'); e.target.reset(); loadVoiceTab(); }
  catch (err) { toast('Consent rejected: ' + err.message, 5000); }
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
      <button class="btn primary" data-gen="${esc(p.generated_voice_id)}">Use preview ${i + 1}</button></div>`).join('');
    document.querySelectorAll('[data-gen]').forEach((b) => b.onclick = async () => {
      await api('/api/voices/design/save', { json: { generated_voice_id: b.dataset.gen, name: 'Designed voice', description } });
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
    'Voice Design calls': m.credits.voice_designs });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ------------------------------------------------------------------ boot
(async () => {
  try {
    await Promise.all([loadStatus(), loadPhraseCache(), reloadClassifier()]);
    populateSignLabels();
  } catch (e) { toast('Backend not reachable: ' + e.message, 6000); }
})();
