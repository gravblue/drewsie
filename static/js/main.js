const video       = document.getElementById('video');
const videoPanel  = document.getElementById('videoPanel');
const videoCol    = document.getElementById('videoCol');
const statusPanel = document.getElementById('statusPanel');
const overlay     = document.getElementById('overlay');
const ctx         = overlay.getContext('2d');
const placeholder = document.getElementById('placeholder');
const startBtn    = document.getElementById('startBtn');
const stopBtn     = document.getElementById('stopBtn');
const resetBtn    = document.getElementById('resetBtn');
const liveDot     = document.getElementById('liveDot');
const fpsTag      = document.getElementById('fpsTag');
const gaugeFill   = document.getElementById('gaugeFill');
const countLabel  = document.getElementById('countLabel');
const statusLabel = document.getElementById('statusLabel');
const alarmBanner = document.getElementById('alarmBanner');
const alarmAudio  = document.getElementById('alarmAudio');
const logList     = document.getElementById('logList');

const CIRCUMFERENCE = 2 * Math.PI * 64;
const SEND_INTERVAL_MS = 333;   // ~3fps -> DROWSY_FRAMES=6 setara 2 detik mata merem

let DROWSY_FRAMES_TOTAL = 6;
let WARN_THRESHOLD = Math.round(DROWSY_FRAMES_TOTAL * 0.5); // level "waspada" mulai dari sini

function syncDrowsyFramesTotal(drowsyFrames) {
if (!drowsyFrames || drowsyFrames === DROWSY_FRAMES_TOTAL) return;
DROWSY_FRAMES_TOTAL = drowsyFrames;
WARN_THRESHOLD = Math.round(DROWSY_FRAMES_TOTAL * 0.5);
}

let currentAlertLevel = 'idle'; // idle -> warning -> danger

let stream = null;
let loopHandle = null;
let cameraActive = false;
let sendCanvas = document.createElement('canvas');
let sending = false;

// WebSocket state
let ws = null;
let wsReady = false;
let wantReconnect = false;
let reconnectDelay = 1000;
let reconnectTimer = null;

function wsUrl() {
const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
return `${proto}//${location.host}/ws`;
}

function connectWs() {
clearTimeout(reconnectTimer);
ws = new WebSocket(wsUrl());
ws.binaryType = 'arraybuffer';

ws.addEventListener('open', () => {
  wsReady = true;
  reconnectDelay = 1000;
  sending = false;
});

ws.addEventListener('message', (event) => {
  let result;
  try {
    result = JSON.parse(event.data);
  } catch (err) {
    sending = false;
    return;
  }
  sending = false;
  if (result.error) return;
  if (result.status === 'reset') {
    updateGauge(0);
    return;
  }

  // Hindari race condition: kalau kamera udah di-stop sebelum response datang, abaikan hasilnya
  if (!cameraActive) return;

  syncDrowsyFramesTotal(result.drowsy_frames);
  drawOverlay(result);
  updateGauge(result.drowsy_count);
  setStatus(result.label, result.prob);
  if (result.label) addLog(result.label, result.prob);
  updateAlarmLevel(result);
});

ws.addEventListener('close', () => {
  wsReady = false;
  sending = false;
  if (wantReconnect) {
    reconnectTimer = setTimeout(connectWs, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 10000);
  }
});

ws.addEventListener('error', () => {
  sending = false;
});
}

function openWsSession() {
wantReconnect = true;
reconnectDelay = 1000;
connectWs();
}

function closeWsSession() {
wantReconnect = false;
clearTimeout(reconnectTimer);
wsReady = false;
if (ws) {
  ws.close();
  ws = null;
}
}

function addLog(label, prob) {
const item = document.createElement('div');
item.className = 'log-item';
const time = new Date().toLocaleTimeString('id-ID', { hour12: false });
const cls = label === 'DROWSY' ? 'drowsy' : 'nondrowsy';
item.innerHTML = `<span>${time}</span><span class="lbl ${cls}">${label} (${(prob*100).toFixed(0)}%)</span>`;
logList.prepend(item);
while (logList.children.length > 30) logList.removeChild(logList.lastChild);
}

function updateGauge(count) {
const ratio = Math.min(count / DROWSY_FRAMES_TOTAL, 1);
gaugeFill.style.strokeDashoffset = CIRCUMFERENCE * (1 - ratio);
gaugeFill.style.stroke = ratio >= 1 ? 'var(--danger)' : ratio > 0.5 ? 'var(--warn)' : 'var(--safe)';
countLabel.textContent = count;
}

function setStatus(label, prob) {
statusLabel.classList.remove('safe', 'warn', 'danger', 'idle');
if (label === null) {
  statusLabel.textContent = '';
  statusLabel.classList.add('idle');
  statusLabel.style.display = 'none';
  return;
}
statusLabel.style.display = 'inline-block';
if (label === 'DROWSY') {
  statusLabel.textContent = 'DROWSY';
  statusLabel.classList.add('danger');
} else {
  statusLabel.textContent = 'NON-DROWSY';
  statusLabel.classList.add('safe');
}
}

function updateAlarmLevel(result) {
const count = result.drowsy_count;
const dangerActive = result.alarm === true;

if (dangerActive) {
  alarmBanner.classList.remove('warning');
  alarmBanner.classList.add('active');
  alarmBanner.textContent = 'WAKE UP ⚠';
  if (currentAlertLevel !== 'danger') {
    alarmAudio.loop = true;
    alarmAudio.currentTime = 0;
    alarmAudio.play().catch(() => {});
  }
  currentAlertLevel = 'danger';
} else if (count >= WARN_THRESHOLD) {
  alarmBanner.classList.remove('active');
  alarmBanner.classList.add('warning');
  alarmBanner.textContent = 'STAY ALERT';
  if (currentAlertLevel === 'danger') {
    alarmAudio.pause();
    alarmAudio.currentTime = 0;
    alarmAudio.loop = true;
  }
  currentAlertLevel = 'warning';
} else {
  alarmBanner.classList.remove('active', 'warning');
  if (currentAlertLevel !== 'idle') {
    alarmAudio.pause();
    alarmAudio.currentTime = 0;
    alarmAudio.loop = true; // reset default buat siklus alarm penuh berikutnya
  }
  currentAlertLevel = 'idle';
}
}

function resizeOverlayCanvas() {
const dpr = window.devicePixelRatio || 1;
const cssW = overlay.clientWidth;
const cssH = overlay.clientHeight;
if (!cssW || !cssH) return;
overlay.width = Math.round(cssW * dpr);
overlay.height = Math.round(cssH * dpr);
ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawOverlay(result) {
const cssW = overlay.clientWidth;
const cssH = overlay.clientHeight;
ctx.clearRect(0, 0, cssW, cssH);
if (!result.bbox) return;
// Belum ada label & bukan kasus "wajah kejauhan" -> masih nunggu streak/inference,
// jangan gambar kotak dulu (biar gak ada kotak abu-abu netral yang keliatan kayak putih)
if (!result.label && !result.face_too_small) return;

const scaleX = cssW / sendCanvas.width;
const scaleY = cssH / sendCanvas.height;
const [x1, y1, x2, y2] = result.bbox;

// Overlay ga ikut di-mirror kayak video (selfie view), jadi X di-mirror manual
const bx1 = cssW - (x2 * scaleX);
const bx2 = cssW - (x1 * scaleX);
const by1 = y1 * scaleY;
const by2 = y2 * scaleY;

let color = '#d97706';
if (result.label === 'DROWSY') color = '#dc2626';
else if (result.label === 'NON-DROWSY') color = '#16a34a';

ctx.strokeStyle = color;
ctx.lineWidth = 2;
ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);

if (result.label) {
  ctx.fillStyle = color;
  const text = `${result.label} (${result.prob.toFixed(2)})`;
  ctx.font = '600 10px Roboto Mono, monospace';
  const textWidth = ctx.measureText(text).width;
  ctx.fillRect(bx1, by1 - 22, textWidth + 10, 22);
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, bx1 + 5, by1 - 6);
} else if (result.face_too_small) {
  ctx.fillStyle = color;
  ctx.font = '600 10px Roboto Mono, monospace';
  ctx.fillText('Face Too Far', bx1, by1 - 6);
}
}

async function sendFrame() {
if (!cameraActive || sending || video.readyState < 2) return;
if (!wsReady || !ws || ws.readyState !== WebSocket.OPEN) return;
sending = true;
try {
  sendCanvas.width = video.videoWidth;
  sendCanvas.height = video.videoHeight;
  const sctx = sendCanvas.getContext('2d');
  // Frame dikirim TIDAK di-mirror (biar bbox server konsisten); mirroring cuma efek visual overlay
  sctx.drawImage(video, 0, 0, sendCanvas.width, sendCanvas.height);

  // Dikirim sebagai Blob biner (JPEG) langsung lewat WebSocket (bukan lagi
  // multipart/form-data + request HTTP baru tiap frame)
  const blob = await new Promise((resolve) =>
    sendCanvas.toBlob(resolve, 'image/jpeg', 0.85)
  );
  const buf = await blob.arrayBuffer();

  if (!cameraActive || !ws || ws.readyState !== WebSocket.OPEN) {
    sending = false;
    return;
  }
  ws.send(buf);
} catch (err) {
  console.error('Send frame error:', err);
  sending = false;
}
}

async function startCamera() {
try {
  const isNarrowPhone = window.matchMedia('(max-width: 480px)').matches;
  const idealRatio = isNarrowPhone ? 3 / 4 : 4 / 3;
  const constraints = {
    video: {
      facingMode: { exact: 'user' },
      width: { ideal: isNarrowPhone ? 720 : 1280 },
      height: { ideal: isNarrowPhone ? 960 : 720 },
      aspectRatio: { ideal: idealRatio },
    }
  };
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (exactErr) {
    constraints.video.facingMode = { ideal: 'user' };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  }
} catch (err) {
  alert('Ga bisa akses kamera: ' + err.message);
  return;
}

try {
  const wasMuted = alarmAudio.muted;
  alarmAudio.muted = true;
  await alarmAudio.play();
  alarmAudio.pause();
  alarmAudio.currentTime = 0;
  alarmAudio.muted = wasMuted;
} catch (err) {
}

video.srcObject = stream;
video.style.display = 'block';
overlay.style.display = 'block';
placeholder.style.display = 'none';
liveDot.classList.add('live');
fpsTag.textContent = 'LIVE';
cameraActive = true;

video.onloadedmetadata = () => {
  resizeOverlayCanvas();
};

startBtn.disabled = true;
stopBtn.disabled = false;
resetBtn.disabled = false;

openWsSession();
loopHandle = setInterval(sendFrame, SEND_INTERVAL_MS);
}

function stopCamera() {
cameraActive = false;
if (stream) stream.getTracks().forEach(t => t.stop());
if (loopHandle) clearInterval(loopHandle);
closeWsSession();
logList.innerHTML = '';
video.style.display = 'none';
overlay.style.display = 'none';
placeholder.style.display = 'block';
liveDot.classList.remove('live');
fpsTag.textContent = 'CAM OFF';
ctx.clearRect(0, 0, overlay.clientWidth, overlay.clientHeight);
setStatus(null, null);
updateGauge(0);
alarmBanner.classList.remove('active', 'warning');
alarmAudio.pause();
alarmAudio.currentTime = 0;
alarmAudio.loop = true;
currentAlertLevel = 'idle';

startBtn.disabled = false;
stopBtn.disabled = true;
resetBtn.disabled = true;
}

async function resetCounter() {
if (ws && ws.readyState === WebSocket.OPEN) {
  ws.send('reset');
}
updateGauge(0);
}

startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
resetBtn.addEventListener('click', resetCounter);

const logPanel   = document.getElementById('logPanel');
const wrapperEl  = document.querySelector('.wrapper');
const headerEl   = document.querySelector('header');
const iconRow    = document.querySelector('.icon-controls');

function sizeVideoPanel() {
if (!videoPanel || !wrapperEl || !headerEl || !iconRow) return;
if (window.innerWidth <= 760) {
  videoPanel.style.height = '';
  return;
}
const bodyStyle   = getComputedStyle(document.body);
const padTop      = parseFloat(bodyStyle.paddingTop) || 0;
const padBottom   = parseFloat(bodyStyle.paddingBottom) || 0;
const wrapperGap  = parseFloat(getComputedStyle(wrapperEl).rowGap) || 16;
const colGap      = parseFloat(getComputedStyle(videoCol).rowGap) || 14;

const SAFETY_MARGIN = 8;
const SHRINK = 0.99;
const used = padTop + padBottom + headerEl.offsetHeight + wrapperGap +
             iconRow.offsetHeight + colGap + SAFETY_MARGIN;
const available = (window.innerHeight - used) * SHRINK;
videoPanel.style.height = Math.max(Math.min(available, 560), 200) + 'px';
}

function syncHistoryHeight() {
if (!videoCol || !logPanel || !statusPanel) return;
if (window.innerWidth <= 760) {
  logPanel.style.height = '';
  logPanel.classList.remove('height-synced');
  return;
}
// Kolom kanan (status+history) disamain tingginya sama kolom kiri (videoCol)
const SIDE_GAP = 12; // harus sama dengan CSS ".side { gap: 12px; }"
const availableHeight = videoCol.offsetHeight - statusPanel.offsetHeight - SIDE_GAP;
logPanel.style.height = Math.max(availableHeight, 120) + 'px';
logPanel.classList.add('height-synced');
}

function syncLayout() {
sizeVideoPanel();
syncHistoryHeight();
if (cameraActive) resizeOverlayCanvas();
}

window.addEventListener('load', syncLayout);
if (document.fonts && document.fonts.ready) {
document.fonts.ready.then(syncLayout).catch(() => {});
}
window.addEventListener('resize', syncLayout);
if (window.ResizeObserver) {
new ResizeObserver(syncHistoryHeight).observe(statusPanel);
}
syncLayout();