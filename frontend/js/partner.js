// Partner's phone: hold-to-talk into the shared session, plus large captions of what the patient says.
// The laptop picks up these turns through /api/partner/latest and shows the reply suggestions there.
import { Recorder } from './audio.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const session = new URLSearchParams(location.search).get('s');
const recorder = new Recorder();
let toastTimer, lastId = 0;

function toast(msg, ms = 3000) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), ms);
}

async function send(fd) {
  fd.append('session_id', session);
  const res = await fetch('/api/partner', { method: 'POST', body: fd });
  const r = await res.json();
  if (!res.ok) throw new Error(r.detail || res.statusText);
  if (!r.transcript) toast(r.note || 'Nothing was heard. Try again or type.', 4000);
  refresh();
}

// ---- hold to talk (needs a secure origin for the microphone)
const canRecord = !!navigator.mediaDevices?.getUserMedia;
if (!canRecord) {
  $('#hold').disabled = true;
  $('#micNote').hidden = false;
  $('#micNote').textContent = 'The microphone needs an https link. You can still type, and captions still work.';
}
async function start(e) {
  e.preventDefault();
  try {
    if (!recorder.ctx) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      await recorder.init(stream);
    }
  } catch (err) { toast('Microphone not available: ' + err.message, 5000); return; }
  recorder.start();
  $('#hold').setAttribute('aria-pressed', 'true'); $('#holdLabel').textContent = 'Listening… release to send';
}
async function stop() {
  if ($('#hold').getAttribute('aria-pressed') !== 'true') return;
  $('#hold').setAttribute('aria-pressed', 'false'); $('#holdLabel').textContent = 'Hold to talk';
  const { blob, seconds } = recorder.stop();
  if (seconds < 0.3) { toast('Hold the button while you speak'); return; }
  const fd = new FormData(); fd.append('audio', blob, 'partner.wav');
  try { await send(fd); } catch (err) { toast('Could not send: ' + err.message, 5000); }
}
$('#hold').addEventListener('pointerdown', start);
['pointerup', 'pointercancel', 'pointerleave'].forEach((ev) => $('#hold').addEventListener(ev, stop));

$('#typeForm').onsubmit = async (e) => {
  e.preventDefault();
  const t = $('#typed').value.trim(); if (!t) return;
  $('#typed').value = '';
  const fd = new FormData(); fd.append('text', t);
  try { await send(fd); } catch (err) { toast('Could not send: ' + err.message, 5000); }
};

// ---- captions: poll the shared conversation
async function refresh() {
  try {
    const turns = await (await fetch(`/api/history/${encodeURIComponent(session)}`)).json();
    const newest = turns.at(-1)?.id || 0;
    if (newest === lastId) return;
    lastId = newest;
    $('#chat').innerHTML = turns.map((t) => `<li class="${t.speaker}"><small>${t.speaker === 'patient' ? 'patient' : 'you'}</small>${esc(t.text)}</li>`).join('');
    $('#chat').lastElementChild?.scrollIntoView({ block: 'nearest' });
    const said = [...turns].reverse().find((t) => t.speaker === 'patient');
    if (said) $('#lastSaid').textContent = said.text;
  } catch { /* laptop asleep or offline; keep trying */ }
}

if (!session) {
  $('#lastSaid').textContent = 'Open this page from the QR code on the patient\'s screen.';
  $('#hold').disabled = true;
} else {
  refresh();
  setInterval(refresh, 1500);
}
