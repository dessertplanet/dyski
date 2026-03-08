const connectBtn = document.getElementById('connectBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const sourceBtn = document.getElementById('sourceBtn');
const sourcePanel = document.getElementById('sourcePanel');
const uploadBtn = document.getElementById('uploadBtn');
const screenBtn = document.getElementById('screenBtn');
const cameraBtn = document.getElementById('cameraBtn');
const followMouseBtn = document.getElementById('followMouseBtn');
const emptyUploadBtn = document.getElementById('emptyUploadBtn');
const emptyUploadHint = document.getElementById('emptyUploadHint');
const imageInput = document.getElementById('imageInput');
const previewCanvas = document.getElementById('previewCanvas');
const serialStatus = document.getElementById('serialStatus');
const slewSlider = document.getElementById('slewSlider');
const settingsBtn = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const borderWeightSlider = document.getElementById('borderWeightSlider');
const borderColorPicker = document.getElementById('borderColorPicker');
const invertBtn = document.getElementById('invertBtn');
const colorsBtn = document.getElementById('colorsBtn');
const colorsPanel = document.getElementById('colorsPanel');
const exposureSlider = document.getElementById('exposureSlider');
const hueSlider = document.getElementById('hueSlider');
const contrastSlider = document.getElementById('contrastSlider');
const previewCtx = previewCanvas.getContext('2d');
const analysisCanvas = document.createElement('canvas');
analysisCanvas.width = 256;
analysisCanvas.height = 256;
const analysisCtx = analysisCanvas.getContext('2d', { willReadFrequently: true });

function resizePreviewCanvas() {
  const rect = previewCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(rect.width * dpr);
  const h = Math.round(rect.height * dpr);
  if (previewCanvas.width !== w || previewCanvas.height !== h) {
    previewCanvas.width = w;
    previewCanvas.height = h;
  }
}

resizePreviewCanvas();
window.addEventListener('resize', () => {
  resizePreviewCanvas();
  renderViewfinder();
});

const FRAME_KNOB_0 = 'K'.charCodeAt(0);
const FRAME_KNOB_1 = 'N'.charCodeAt(0);
const FRAME_CV_0 = 'C'.charCodeAt(0);
const FRAME_CV_1 = 'V'.charCodeAt(0);
const FRAME_RGB_0 = 'R'.charCodeAt(0);
const FRAME_RGB_1 = 'B'.charCodeAt(0);
const KNOB_FRAME_SIZE = 11;
const CV_FRAME_SIZE = 8;
const SYNTH_GRID = 4;
const MAX_SYNTH_GRID = 4;
const RGB_FRAME_SIZE = 2 + SYNTH_GRID * SYNTH_GRID * 3 + 1; // 51 (legacy, not used for send)
const VIEW_SIZE = 48;
const MIN_VIEW_SIZE = 8;
const MAX_VIEW_SIZE = 256;
const MIN_ENHANCE_RES = 2;
const CV_SEND_INTERVAL_MS = 100;

let port;
let openPromise;
let serialState = 'disconnected';
let portIsOpen = false;
let reader;
let writer;
let readLoopPromise;
let readBuffer = new Uint8Array(0);
let sourceImageBitmap = null;
let analysisPixels = null;
let knobState = { main: 2048, x: 2048, y: 2048, sw: 1, seq: 0 };
let smoothedKnobState = { main: 2048, x: 2048, y: 2048, sw: 1, seq: 0 };
let outputState = { cv1: 0, cv2: 0, pulses: 0, seq: 0 };
let brightGateWasHigh = false;
let edgeGateWasHigh = false;
let linkStats = {
  validFrames: 0,
  badChecksums: 0,
  seqGaps: 0,
  lastSeq: null,
};
let rawHistory = [];
let lastCvSendTime = 0;
let followMouse = false;
let mouseOrigin = null;
let viewSize = VIEW_SIZE;
let originalAnalysisPixels = null;
let adjustments = { invert: false, exposure: 0, hue: 0, contrast: 0 };
let viewfinderBorderWeight = 3;
let viewfinderBorderColor = '#6ca6ff';
let screenStream = null;
let screenVideo = null;
let screenAnimFrame = null;
let liveSource = null; // 'screen' | 'camera' | null

function isPortOpen() {
  return portIsOpen;
}

function appendBytes(existing, incoming) {
  const merged = new Uint8Array(existing.length + incoming.length);
  merged.set(existing, 0);
  merged.set(incoming, existing.length);
  return merged;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function mapKnobToViewport(knobValue, maxOrigin) {
  if (maxOrigin <= 0) return 0;
  return Math.round((knobValue / 4095) * maxOrigin);
}

function getEffectiveResolution() {
  const t = smoothedKnobState.main / 4095;
  return clamp(Math.round(MIN_ENHANCE_RES + t * (viewSize - MIN_ENHANCE_RES)), MIN_ENHANCE_RES, viewSize);
}

function getViewportOrigin() {
  const maxOrigin = analysisCanvas.width - viewSize;
  if (followMouse && mouseOrigin) {
    return {
      vx: clamp(mouseOrigin.x, 0, maxOrigin),
      vy: clamp(mouseOrigin.y, 0, maxOrigin),
    };
  }
  return {
    vx: mapKnobToViewport(smoothedKnobState.x, maxOrigin),
    vy: mapKnobToViewport(smoothedKnobState.y, maxOrigin),
  };
}

function toInt16Bytes(value) {
  const normalized = (value & 0xffff);
  return [normalized & 0xff, (normalized >> 8) & 0xff];
}

function fromUint16LE(lo, hi) {
  return lo | (hi << 8);
}

function toSignedCv(value) {
  return clamp(Math.round(value), -2048, 2047);
}

function frameChecksum(bytes, count) {
  let sum = 0;
  for (let i = 0; i < count; i += 1) {
    sum ^= bytes[i];
  }
  return sum & 0xff;
}

function smoothTowards(current, target, alpha, deadband = 0) {
  if (Math.abs(target - current) <= deadband) {
    return current;
  }

  return current + ((target - current) * alpha);
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function medianFromHistory(key, fallback) {
  if (rawHistory.length === 0) return fallback;
  return median(rawHistory.map((entry) => entry[key]));
}

function rgbToHsl(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return [h, s, l];
}

function hue2rgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}

function hslToRgb(h, s, l) {
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
}

function hasActiveAdjustments() {
  return adjustments.invert || adjustments.exposure !== 0 || adjustments.hue !== 0 || adjustments.contrast !== 0;
}

function buildCssFilterString() {
  const parts = [];
  if (adjustments.exposure !== 0) {
    parts.push(`brightness(${Math.pow(2, adjustments.exposure / 100)})`);
  }
  if (adjustments.contrast !== 0) {
    // Map -100..100 to CSS contrast 0..2 (0=grey, 1=normal, 2=double)
    const contVal = adjustments.contrast * 2.55;
    const factor = (259 * (contVal + 255)) / (255 * (259 - contVal));
    parts.push(`contrast(${factor})`);
  }
  if (adjustments.hue !== 0) {
    parts.push(`hue-rotate(${adjustments.hue}deg)`);
  }
  if (adjustments.invert) {
    parts.push('invert(1)');
  }
  return parts.length > 0 ? parts.join(' ') : 'none';
}

function isVideoSourceReady(video) {
  return !!(
    video
    && video.readyState >= video.HAVE_CURRENT_DATA
    && video.videoWidth > 0
    && video.videoHeight > 0
  );
}

function createLiveVideoElement() {
  const video = document.createElement('video');
  video.muted = true;
  video.autoplay = true;
  video.playsInline = true;
  video.setAttribute('aria-hidden', 'true');
  video.style.position = 'fixed';
  video.style.left = '-9999px';
  video.style.top = '0';
  video.style.width = '1px';
  video.style.height = '1px';
  video.style.opacity = '0';
  video.style.pointerEvents = 'none';
  document.body.appendChild(video);
  return video;
}

function tryPlayVideo(video) {
  if (!video) return;
  const playPromise = video.play();
  if (playPromise) {
    playPromise.catch(() => {
      // Some browsers report a transient source-wait error before frames arrive.
      // The capture loop keeps polling and playback is nudged again on metadata events.
    });
  }
}

function applyImageManipulations() {
  if (!originalAnalysisPixels) return;

  if (!hasActiveAdjustments()) {
    // No adjustments — restore original pixels and let renderViewfinder use the full-res bitmap
    const imageData = new ImageData(new Uint8ClampedArray(originalAnalysisPixels), analysisCanvas.width, analysisCanvas.height);
    analysisCtx.putImageData(imageData, 0, 0);
    analysisPixels = imageData.data;
    renderViewfinder();

    if (isPortOpen() && analysisPixels) {
      const { cv1, cv2, pulses } = analyzeViewport();
      updateLocalOutputs(cv1, cv2, pulses);
      sendCvFrame(cv1, cv2);
    }
    return;
  }

  const src = originalAnalysisPixels;
  const len = src.length;
  const dst = new Uint8ClampedArray(len);

  const expFactor = Math.pow(2, adjustments.exposure / 100);
  const contVal = adjustments.contrast * 2.55;
  const contFactor = (259 * (contVal + 255)) / (255 * (259 - contVal));
  const hueShift = adjustments.hue / 360;

  for (let i = 0; i < len; i += 4) {
    let r = src[i];
    let g = src[i + 1];
    let b = src[i + 2];

    // Exposure
    r *= expFactor;
    g *= expFactor;
    b *= expFactor;

    // Contrast
    r = contFactor * (r - 128) + 128;
    g = contFactor * (g - 128) + 128;
    b = contFactor * (b - 128) + 128;

    // Clamp before hue rotation
    r = clamp(r, 0, 255);
    g = clamp(g, 0, 255);
    b = clamp(b, 0, 255);

    // Hue rotation
    if (hueShift !== 0) {
      const [h0, s0, l0] = rgbToHsl(r / 255, g / 255, b / 255);
      let h1 = (h0 + hueShift) % 1;
      if (h1 < 0) h1 += 1;
      const [rr, gg, bb] = hslToRgb(h1, s0, l0);
      r = rr * 255;
      g = gg * 255;
      b = bb * 255;
    }

    // Invert
    if (adjustments.invert) {
      r = 255 - r;
      g = 255 - g;
      b = 255 - b;
    }

    dst[i] = Math.round(r);
    dst[i + 1] = Math.round(g);
    dst[i + 2] = Math.round(b);
    dst[i + 3] = src[i + 3];
  }

  const imageData = new ImageData(dst, analysisCanvas.width, analysisCanvas.height);
  analysisCtx.putImageData(imageData, 0, 0);
  analysisPixels = dst;

  renderViewfinder();

  if (isPortOpen() && analysisPixels) {
    const { cv1, cv2, pulses } = analyzeViewport();
    updateLocalOutputs(cv1, cv2, pulses);
    sendCvFrame(cv1, cv2);
  }
}

function setSerialState(nextState) {
  serialState = nextState;
  refreshButtons();
}

function refreshButtons() {
  const connected = serialState === 'connected';
  const connecting = serialState === 'connecting';

  connectBtn.disabled = connecting;
  disconnectBtn.disabled = connecting || !connected;

  connectBtn.style.display = connected ? 'none' : 'inline-flex';
  disconnectBtn.style.display = connected ? 'inline-flex' : 'none';

  if (serialState === 'connecting') {
    serialStatus.textContent = 'Connecting…';
    serialStatus.style.borderColor = '#6ca6ff';
    return;
  }

  if (connected) {
    serialStatus.textContent = 'Connected';
    serialStatus.style.borderColor = '#5fd38d';
    return;
  }

  if (serialState === 'ready') {
    serialStatus.textContent = 'Permission Granted';
    serialStatus.style.borderColor = '#6ca6ff';
    return;
  }

  if (serialState === 'error') {
    serialStatus.textContent = 'Connection Failed';
    serialStatus.style.borderColor = '#ff7b7b';
    return;
  }

  serialStatus.textContent = 'Disconnected';
  serialStatus.style.borderColor = '#334055';
}

async function ensurePortSelected() {
  if (port) {
    return port;
  }

  const grantedPorts = await navigator.serial.getPorts();
  if (grantedPorts.length > 0) {
    [port] = grantedPorts;
    if (!isPortOpen()) {
      setSerialState('ready');
    }
    return port;
  }

  port = await navigator.serial.requestPort();
  setSerialState('ready');
  return port;
}

async function connectSerial() {
  if (!('serial' in navigator)) {
    window.alert('Web Serial API is not supported in this browser.');
    return;
  }

  if (isPortOpen()) {
    setSerialState('connected');
    return;
  }

  if (openPromise) {
    try {
      await openPromise;
      setSerialState('connected');
      return;
    } catch {
      openPromise = null;
    }
  }

  setSerialState('connecting');

  try {
    openPromise = (async () => {
      await ensurePortSelected();

      if (!port.readable) {
        await port.open({ baudRate: 115200 });
      }

      // Assert DTR so the firmware's tud_cdc_connected() sees the host as present
      await port.setSignals({ dataTerminalReady: true, requestToSend: true });

      portIsOpen = true;
      reader = port.readable.getReader();
      writer = port.writable.getWriter();
      readLoopPromise = readSerialLoop();
    })();

    await openPromise;
    setSerialState('connected');
  } catch (error) {
    portIsOpen = false;
    setSerialState('error');
    window.alert(`Connect failed: ${error.message}`);
  } finally {
    openPromise = null;
  }
}

async function disconnectSerial() {
  try {
    if (reader) {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
      reader = null;
    }
    if (writer) {
      writer.releaseLock();
      writer = null;
    }
    if (port) {
      await port.close();
    }

    portIsOpen = false;
    setSerialState('disconnected');
  } catch (error) {
    setSerialState('error');
    window.alert(`Disconnect failed: ${error.message}`);
  }
}

function updatePreview(file) {
  stopLiveCapture();
  if (!file) {
    sourceImageBitmap = null;
    analysisPixels = null;
    originalAnalysisPixels = null;
    renderViewfinder();
    return;
  }

  loadImage(file).catch((error) => {
    window.alert(`Image load failed: ${error.message}`);
  });
}

function openImagePickerAndUpload() {
  imageInput.value = '';
  imageInput.click();
}

connectBtn.addEventListener('click', connectSerial);
disconnectBtn.addEventListener('click', disconnectSerial);
uploadBtn.addEventListener('click', () => {
  closeSourcePanel();
  openImagePickerAndUpload();
});
emptyUploadBtn.addEventListener('click', openImagePickerAndUpload);

sourceBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  sourcePanel.classList.toggle('source-panel-open');
});

sourcePanel.addEventListener('click', (e) => {
  e.stopPropagation();
});

document.addEventListener('click', () => {
  closeSourcePanel();
  if (colorsPanel.classList.contains('colors-panel-open')) {
    colorsPanel.classList.remove('colors-panel-open');
    colorsBtn.classList.remove('toggle-active');
  }
  if (settingsPanel.classList.contains('settings-panel-open')) {
    settingsPanel.classList.remove('settings-panel-open');
    settingsBtn.classList.remove('toggle-active');
  }
});

function closeSourcePanel() {
  sourcePanel.classList.remove('source-panel-open');
  const cameraList = document.getElementById('cameraList');
  cameraList.classList.remove('camera-list-open');
  cameraList.innerHTML = '';
}

function updateEmptyState() {
  const hasSource = !!(sourceImageBitmap || screenStream);
  emptyUploadBtn.style.display = hasSource ? 'none' : '';
  emptyUploadHint.style.display = hasSource ? 'none' : '';
}

screenBtn.addEventListener('click', async () => {
  closeSourcePanel();
  if (liveSource === 'screen') {
    stopLiveCapture();
    return;
  }
  try {
    await startLiveCapture('screen');
  } catch (error) {
    if (error.name !== 'NotAllowedError') {
      window.alert(`Screen capture failed: ${error.message}`);
    }
  }
});

cameraBtn.addEventListener('click', async () => {
  if (liveSource === 'camera') {
    closeSourcePanel();
    stopLiveCapture();
    return;
  }
  try {
    // Request temporary permission to enumerate labeled devices
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    tempStream.getTracks().forEach((t) => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === 'videoinput');
    const cameraList = document.getElementById('cameraList');
    cameraList.innerHTML = '';

    if (videoDevices.length <= 1) {
      // Only one camera — start immediately
      closeSourcePanel();
      cameraList.classList.remove('camera-list-open');
      await startLiveCapture('camera');
    } else {
      // Multiple cameras — show picker
      videoDevices.forEach((device, i) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = device.label || `Camera ${i + 1}`;
        btn.addEventListener('click', async () => {
          closeSourcePanel();
          cameraList.classList.remove('camera-list-open');
          try {
            await startLiveCapture('camera', device.deviceId);
          } catch (err) {
            if (err.name !== 'NotAllowedError') {
              window.alert(`Camera failed: ${err.message}`);
            }
          }
        });
        cameraList.appendChild(btn);
      });
      cameraList.classList.add('camera-list-open');
    }
  } catch (error) {
    closeSourcePanel();
    if (error.name !== 'NotAllowedError') {
      window.alert(`Camera failed: ${error.message}`);
    }
  }
});

followMouseBtn.addEventListener('click', () => {
  followMouse = !followMouse;
  followMouseBtn.textContent = followMouse ? 'Mouse Mode: On' : 'Mouse Mode: Off';
  followMouseBtn.classList.toggle('toggle-active', followMouse);
  previewCanvas.classList.toggle('hide-cursor', followMouse);
  if (!followMouse) {
    mouseOrigin = null;
    renderViewfinder();
  }
});

previewCanvas.addEventListener('mousemove', (event) => {
  if (!followMouse || (!sourceImageBitmap && !screenStream)) return;
  const rect = previewCanvas.getBoundingClientRect();
  const canvasX = ((event.clientX - rect.left) / rect.width) * analysisCanvas.width;
  const canvasY = ((event.clientY - rect.top) / rect.height) * analysisCanvas.height;
  mouseOrigin = {
    x: Math.round(canvasX - viewSize / 2),
    y: Math.round(canvasY - viewSize / 2),
  };
  renderViewfinder();
  if (analysisPixels) {
    const { cv1, cv2, pulses } = analyzeViewport();
    updateLocalOutputs(cv1, cv2, pulses);
    sendCvFrame(cv1, cv2);
  }
});

previewCanvas.addEventListener('mouseleave', () => {
  // keep mouseOrigin so viewfinder stays at last position
});

// Scroll to resize viewfinder (up = bigger, down = smaller)
previewCanvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  const delta = event.deltaY < 0 ? 2 : -2;
  viewSize = clamp(viewSize + delta, MIN_VIEW_SIZE, MAX_VIEW_SIZE);
  renderViewfinder();
}, { passive: false });

imageInput.addEventListener('change', (event) => {
  const [file] = event.target.files;
  updatePreview(file);
});

// ---- Image adjustment controls ----
invertBtn.addEventListener('click', () => {
  adjustments.invert = !adjustments.invert;
  invertBtn.classList.toggle('toggle-active', adjustments.invert);
  applyImageManipulations();
});

colorsBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const isOpen = colorsPanel.classList.toggle('colors-panel-open');
  colorsBtn.classList.toggle('toggle-active', isOpen);
});

colorsPanel.addEventListener('click', (e) => {
  e.stopPropagation();
});

function syncAdjustmentControls() {
  exposureSlider.value = adjustments.exposure;
  hueSlider.value = adjustments.hue;
  contrastSlider.value = adjustments.contrast;
  invertBtn.classList.toggle('toggle-active', adjustments.invert);
}

function resetAdjustments(applyChanges = true) {
  adjustments = { invert: false, exposure: 0, hue: 0, contrast: 0 };
  syncAdjustmentControls();
  if (applyChanges) {
    applyImageManipulations();
  }
}

function onAdjustmentSliderChange() {
  adjustments.exposure = parseInt(exposureSlider.value, 10);
  adjustments.hue = parseInt(hueSlider.value, 10);
  adjustments.contrast = parseInt(contrastSlider.value, 10);
  applyImageManipulations();
}

exposureSlider.addEventListener('input', onAdjustmentSliderChange);
hueSlider.addEventListener('input', onAdjustmentSliderChange);
contrastSlider.addEventListener('input', onAdjustmentSliderChange);

document.getElementById('colorsResetBtn').addEventListener('click', () => {
  resetAdjustments();
});

// ---- Settings panel ----
settingsBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const isOpen = settingsPanel.classList.toggle('settings-panel-open');
  settingsBtn.classList.toggle('toggle-active', isOpen);
});

settingsPanel.addEventListener('click', (e) => {
  e.stopPropagation();
});

borderWeightSlider.addEventListener('input', () => {
  viewfinderBorderWeight = parseFloat(borderWeightSlider.value);
  renderViewfinder();
});

borderColorPicker.addEventListener('input', () => {
  viewfinderBorderColor = borderColorPicker.value;
  renderViewfinder();
});

// ---- Drag-and-drop image loading ----
const stage = document.getElementById('stage');

stage.addEventListener('dragover', (event) => {
  event.preventDefault();
  event.dataTransfer.dropEffect = 'copy';
  stage.classList.add('drag-over');
});

stage.addEventListener('dragleave', (event) => {
  // Only remove highlight when leaving the stage entirely
  if (!stage.contains(event.relatedTarget)) {
    stage.classList.remove('drag-over');
  }
});

stage.addEventListener('drop', (event) => {
  event.preventDefault();
  stage.classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    loadImage(file).catch((error) => {
      window.alert(`Image load failed: ${error.message}`);
    });
  }
});

window.addEventListener('beforeunload', () => {
  stopLiveCapture();
  if (!port) return;
  if (isPortOpen()) {
    portIsOpen = false;
    port.close().catch(() => {});
  }
});

if ('serial' in navigator) {
  navigator.serial.addEventListener('disconnect', (event) => {
    if (event.target === port) {
      portIsOpen = false;
      port = null;
      setSerialState('disconnected');
    }
  });

  navigator.serial.addEventListener('connect', () => {
    if (port && !isPortOpen()) {
      setSerialState('ready');
    }
  });
}

function stopLiveCapture() {
  if (screenAnimFrame) {
    cancelAnimationFrame(screenAnimFrame);
    screenAnimFrame = null;
  }
  if (screenStream) {
    screenStream.getTracks().forEach((t) => t.stop());
    screenStream = null;
  }
  if (screenVideo) {
    screenVideo.pause();
    screenVideo.srcObject = null;
    screenVideo.remove();
    screenVideo = null;
  }
  screenBtn.classList.remove('toggle-active');
  cameraBtn.classList.remove('toggle-active');
  liveSource = null;
}

async function startLiveCapture(kind, deviceId) {
  stopLiveCapture();
  resetAdjustments(false);

  let stream;
  if (kind === 'screen') {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: { cursor: 'never' },
      audio: false,
    });
  } else {
    const constraints = deviceId
      ? { video: { deviceId: { exact: deviceId } }, audio: false }
      : { video: { facingMode: 'environment' }, audio: false };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  }

  const videoTrack = stream.getVideoTracks()[0];

  // If user stops sharing via browser chrome, clean up
  videoTrack.addEventListener('ended', () => {
    if (screenStream !== stream) return;
    stopLiveCapture();
    if (!sourceImageBitmap) {
      analysisPixels = null;
      originalAnalysisPixels = null;
      renderViewfinder();
    }
  });

  const video = createLiveVideoElement();
  video.srcObject = stream;

  screenStream = stream;
  screenVideo = video;

  // Clear any previous static image so live feed takes over
  sourceImageBitmap = null;
  liveSource = kind;
  if (kind === 'screen') {
    screenBtn.classList.add('toggle-active');
  } else {
    cameraBtn.classList.add('toggle-active');
  }
  renderViewfinder();

  let captureStarted = false;

  const startCaptureLoop = () => {
    if (captureStarted || screenStream !== stream || screenVideo !== video) return;
    captureStarted = true;
    screenAnimFrame = requestAnimationFrame(captureFrame);
  };

  const nudgePlayback = () => {
    if (screenStream !== stream || screenVideo !== video) return;
    tryPlayVideo(video);
    startCaptureLoop();
  };

  video.addEventListener('loadedmetadata', nudgePlayback);
  video.addEventListener('loadeddata', nudgePlayback);
  video.addEventListener('canplay', nudgePlayback);
  video.addEventListener('playing', nudgePlayback);
  videoTrack.addEventListener('unmute', nudgePlayback);

  tryPlayVideo(video);
  startCaptureLoop();

  function captureFrame() {
    if (screenStream !== stream || screenVideo !== video) return;

    // Skip frames until the video is actually delivering pixels
    if (!isVideoSourceReady(video)) {
      screenAnimFrame = requestAnimationFrame(captureFrame);
      return;
    }

    analysisCtx.clearRect(0, 0, analysisCanvas.width, analysisCanvas.height);
    analysisCtx.drawImage(video, 0, 0, analysisCanvas.width, analysisCanvas.height);

    const imageData = analysisCtx.getImageData(0, 0, analysisCanvas.width, analysisCanvas.height);
    originalAnalysisPixels = new Uint8ClampedArray(imageData.data);

    if (hasActiveAdjustments()) {
      applyImageManipulations();
    } else {
      analysisPixels = imageData.data;
      renderViewfinder();

      if (isPortOpen() && analysisPixels) {
        const { cv1, cv2, pulses } = analyzeViewport();
        updateLocalOutputs(cv1, cv2, pulses);
        sendCvFrame(cv1, cv2);
      }
    }

    screenAnimFrame = requestAnimationFrame(captureFrame);
  }

  captureFrame();
}

async function loadImage(file) {
  stopLiveCapture();
  resetAdjustments(false);
  const bitmap = await createImageBitmap(file);
  sourceImageBitmap = bitmap;

  analysisCtx.clearRect(0, 0, analysisCanvas.width, analysisCanvas.height);
  analysisCtx.drawImage(bitmap, 0, 0, analysisCanvas.width, analysisCanvas.height);

  const imageData = analysisCtx.getImageData(0, 0, analysisCanvas.width, analysisCanvas.height);
  originalAnalysisPixels = new Uint8ClampedArray(imageData.data);
  analysisPixels = imageData.data;
  applyImageManipulations();
}

function renderViewfinder() {
  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);

  if (!sourceImageBitmap && !screenStream) {
    previewCtx.fillStyle = '#0b0e14';
    previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
    updateEmptyState();
    return;
  }

  updateEmptyState();

  if (hasActiveAdjustments() && sourceImageBitmap) {
    previewCtx.save();
    previewCtx.filter = buildCssFilterString();
    previewCtx.drawImage(sourceImageBitmap, 0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.restore();
  } else if (screenStream && screenVideo && screenVideo.readyState >= screenVideo.HAVE_CURRENT_DATA) {
    if (hasActiveAdjustments()) {
      previewCtx.save();
      previewCtx.filter = buildCssFilterString();
      previewCtx.drawImage(screenVideo, 0, 0, previewCanvas.width, previewCanvas.height);
      previewCtx.restore();
    } else {
      previewCtx.drawImage(screenVideo, 0, 0, previewCanvas.width, previewCanvas.height);
    }
  } else if (sourceImageBitmap) {
    previewCtx.drawImage(sourceImageBitmap, 0, 0, previewCanvas.width, previewCanvas.height);
  }

  const { vx, vy } = getViewportOrigin();

  const scaleX = previewCanvas.width / analysisCanvas.width;
  const scaleY = previewCanvas.height / analysisCanvas.height;

  const effectiveRes = getEffectiveResolution();

  // Draw pixelated blocks inside the viewfinder area
  if (analysisPixels && effectiveRes < viewSize) {
    const step = viewSize / effectiveRes;
    const blockW = (viewSize * scaleX) / effectiveRes;
    const blockH = (viewSize * scaleY) / effectiveRes;

    for (let gy = 0; gy < effectiveRes; gy += 1) {
      for (let gx = 0; gx < effectiveRes; gx += 1) {
        const sampleX = Math.min(Math.floor(vx + (gx + 0.5) * step), analysisCanvas.width - 1);
        const sampleY = Math.min(Math.floor(vy + (gy + 0.5) * step), analysisCanvas.height - 1);
        const idx = (sampleY * analysisCanvas.width + sampleX) * 4;
        const r = analysisPixels[idx];
        const g = analysisPixels[idx + 1];
        const b = analysisPixels[idx + 2];

        previewCtx.fillStyle = `rgb(${r},${g},${b})`;
        previewCtx.fillRect(
          vx * scaleX + gx * blockW,
          vy * scaleY + gy * blockH,
          Math.ceil(blockW),
          Math.ceil(blockH),
        );
      }
    }
  }

  if (viewfinderBorderWeight > 0) {
    previewCtx.strokeStyle = viewfinderBorderColor;
    previewCtx.lineWidth = viewfinderBorderWeight;
    previewCtx.strokeRect(vx * scaleX, vy * scaleY, viewSize * scaleX, viewSize * scaleY);
  }

  if (effectiveRes >= viewSize) {
    previewCtx.fillStyle = 'rgba(108, 166, 255, 0.12)';
    previewCtx.fillRect(vx * scaleX, vy * scaleY, viewSize * scaleX, viewSize * scaleY);
  }

  const mode = ['Edge', 'Texture', 'Brightness'][clamp(smoothedKnobState.sw, 0, 2)] ?? 'Edge';
}

function analyzeViewport() {
  if (!analysisPixels) {
    return { cv1: 0, cv2: 0, pulses: 0 };
  }

  const { vx: originX, vy: originY } = getViewportOrigin();
  const mode = clamp(smoothedKnobState.sw, 0, 2);
  const effectiveRes = getEffectiveResolution();
  const step = viewSize / effectiveRes;

  // Build a downsampled grayscale grid at the effective resolution
  const gray = new Float32Array(effectiveRes * effectiveRes);
  for (let gy = 0; gy < effectiveRes; gy += 1) {
    for (let gx = 0; gx < effectiveRes; gx += 1) {
      const sampleX = Math.min(Math.floor(originX + (gx + 0.5) * step), analysisCanvas.width - 1);
      const sampleY = Math.min(Math.floor(originY + (gy + 0.5) * step), analysisCanvas.height - 1);
      const index = (sampleY * analysisCanvas.width + sampleX) * 4;
      const r = analysisPixels[index];
      const g = analysisPixels[index + 1];
      const b = analysisPixels[index + 2];
      gray[gy * effectiveRes + gx] = Math.round((r * 77 + g * 150 + b * 29) / 256);
    }
  }

  const count = effectiveRes * effectiveRes;
  const halfRes = effectiveRes / 2;
  let sum = 0;
  let leftSum = 0;
  let rightSum = 0;
  let topSum = 0;
  let bottomSum = 0;
  let edgeSum = 0;
  let brightest = 0;

  for (let gy = 0; gy < effectiveRes; gy += 1) {
    for (let gx = 0; gx < effectiveRes; gx += 1) {
      const val = gray[gy * effectiveRes + gx];
      sum += val;
      brightest = Math.max(brightest, val);

      if (gx < halfRes) leftSum += val;
      else rightSum += val;

      if (gy < halfRes) topSum += val;
      else bottomSum += val;

      if (gx > 0) {
        edgeSum += Math.abs(val - gray[gy * effectiveRes + (gx - 1)]);
      }
      if (gy > 0) {
        edgeSum += Math.abs(val - gray[(gy - 1) * effectiveRes + gx]);
      }
    }
  }

  const mean = sum / count;
  let varianceAccum = 0;
  for (let i = 0; i < count; i += 1) {
    const diff = gray[i] - mean;
    varianceAccum += diff * diff;
  }

  const leftMean = leftSum / (count / 2);
  const rightMean = rightSum / (count / 2);
  const topMean = topSum / (count / 2);
  const bottomMean = bottomSum / (count / 2);
  const variance = varianceAccum / count;
  const normalizedEdge = edgeSum / (count * 2);

  let cv1 = 0;
  let cv2 = 0;
  let pulses = 0;

  if (mode === 0) {
    cv1 = toSignedCv((rightMean - leftMean) * 24);
    cv2 = toSignedCv((bottomMean - topMean) * 24);
    if (normalizedEdge > 28 && !edgeGateWasHigh) pulses |= 0x01;
    edgeGateWasHigh = normalizedEdge > 28;
    if (mean > 96 && !brightGateWasHigh) pulses |= 0x02;
    brightGateWasHigh = mean > 96;
  } else if (mode === 1) {
    cv1 = toSignedCv((variance - 800) * 1.5);
    cv2 = toSignedCv((brightest - 127.5) * 16);
    if (variance > 1200 && !edgeGateWasHigh) pulses |= 0x01;
    edgeGateWasHigh = variance > 1200;
    if (brightest > 220 && !brightGateWasHigh) pulses |= 0x02;
    brightGateWasHigh = brightest > 220;
  } else {
    cv1 = toSignedCv((mean - 127.5) * 16);
    cv2 = toSignedCv((rightMean - leftMean) * 16);
    const brightGate = mean > 150;
    if (brightGate && !brightGateWasHigh) pulses |= 0x01;
    brightGateWasHigh = brightGate;
    if (normalizedEdge > 20 && !edgeGateWasHigh) pulses |= 0x02;
    edgeGateWasHigh = normalizedEdge > 20;
  }

  return { cv1, cv2, pulses };
}

function updateLocalOutputs(cv1, cv2, pulses) {
  outputState.cv1 = cv1;
  outputState.cv2 = cv2;
  outputState.pulses = pulses;
  renderViewfinder();
}

async function sendCvFrame(cv1, cv2) {
  if (!writer || !isPortOpen()) {
    return;
  }

  const now = performance.now();
  if (now - lastCvSendTime < CV_SEND_INTERVAL_MS) {
    return;
  }
  lastCvSendTime = now;

  const [cv1Lo, cv1Hi] = toInt16Bytes(cv1);
  const [cv2Lo, cv2Hi] = toInt16Bytes(cv2);
  const frame = new Uint8Array([
    FRAME_CV_0,
    FRAME_CV_1,
    cv1Lo,
    cv1Hi,
    cv2Lo,
    cv2Hi,
    outputState.pulses & 0xff,
    0,
  ]);
  frame[CV_FRAME_SIZE - 1] = frameChecksum(frame, CV_FRAME_SIZE - 1);
  outputState.seq = (outputState.seq + 1) & 0xff;
  await writer.write(frame);

  // Also send current RGB grid for additive synthesis
  await sendRgbFrame();
}

function buildRgbFrame() {
  if (!analysisPixels) {
    console.log('[RGB] no analysisPixels');
    return null;
  }

  const { vx: originX, vy: originY } = getViewportOrigin();
  const effectiveRes = getEffectiveResolution();
  const synthRes = Math.min(effectiveRes, MAX_SYNTH_GRID);
  const numCells = synthRes * synthRes;
  const frameSize = 4 + numCells * 3 + 1; // header(2) + grid_size(1) + slew(1) + cells + chk(1)

  const frame = new Uint8Array(frameSize);
  frame[0] = FRAME_RGB_0;
  frame[1] = FRAME_RGB_1;
  frame[2] = synthRes; // grid_size byte
  frame[3] = parseInt(slewSlider.value, 10) & 0xff; // slew byte

  // Sample at the full effective resolution, then average down to synthRes
  const srcStep = viewSize / effectiveRes;

  for (let sy = 0; sy < synthRes; sy += 1) {
    for (let sx = 0; sx < synthRes; sx += 1) {
      // Determine which source cells map into this synth cell
      const srcY0 = Math.floor((sy / synthRes) * effectiveRes);
      const srcY1 = Math.floor(((sy + 1) / synthRes) * effectiveRes);
      const srcX0 = Math.floor((sx / synthRes) * effectiveRes);
      const srcX1 = Math.floor(((sx + 1) / synthRes) * effectiveRes);

      let rSum = 0, gSum = 0, bSum = 0, count = 0;
      for (let ey = srcY0; ey < srcY1; ey += 1) {
        for (let ex = srcX0; ex < srcX1; ex += 1) {
          const px = Math.min(
            Math.floor(originX + (ex + 0.5) * srcStep),
            analysisCanvas.width - 1
          );
          const py = Math.min(
            Math.floor(originY + (ey + 0.5) * srcStep),
            analysisCanvas.height - 1
          );
          const idx = (py * analysisCanvas.width + px) * 4;
          rSum += analysisPixels[idx];
          gSum += analysisPixels[idx + 1];
          bSum += analysisPixels[idx + 2];
          count += 1;
        }
      }

      const cell = sy * synthRes + sx;
      frame[4 + cell * 3]     = Math.round(rSum / count);
      frame[4 + cell * 3 + 1] = Math.round(gSum / count);
      frame[4 + cell * 3 + 2] = Math.round(bSum / count);
    }
  }

  let chk = 0;
  for (let i = 0; i < frameSize - 1; i += 1) chk ^= frame[i];
  frame[frameSize - 1] = chk & 0xff;
  return frame;
}

let rgbSendCount = 0;
async function sendRgbFrame() {
  if (!writer || !isPortOpen()) {
    console.log('[RGB] no writer or port closed');
    return;
  }
  const frame = buildRgbFrame();
  if (frame) {
    rgbSendCount++;
    if (rgbSendCount <= 5 || rgbSendCount % 50 === 0) {
      const preview = Array.from(frame.slice(0, 10)).map(b => b.toString(16).padStart(2, '0')).join(' ');
      console.log(`[RGB] send #${rgbSendCount} len=${frame.length} hdr=[${preview}...]`);
    }
    await writer.write(frame);
  }
}

async function parseFrames() {
  while (readBuffer.length >= KNOB_FRAME_SIZE) {
    let start = -1;
    for (let i = 0; i <= readBuffer.length - 2; i += 1) {
      if (readBuffer[i] === FRAME_KNOB_0 && readBuffer[i + 1] === FRAME_KNOB_1) {
        start = i;
        break;
      }
    }

    if (start < 0) {
      readBuffer = new Uint8Array(0);
      return;
    }

    if (readBuffer.length - start < KNOB_FRAME_SIZE) {
      readBuffer = readBuffer.slice(start);
      return;
    }

    const frame = readBuffer.slice(start, start + KNOB_FRAME_SIZE);
    readBuffer = readBuffer.slice(start + KNOB_FRAME_SIZE);

    if (frame[KNOB_FRAME_SIZE - 1] !== frameChecksum(frame, KNOB_FRAME_SIZE - 1)) {
      linkStats.badChecksums += 1;
      continue;
    }

    await handleKnobFrame(frame);
  }
}

async function handleKnobFrame(frame) {
  const seq = frame[9];
  if (linkStats.lastSeq !== null) {
    const expectedSeq = (linkStats.lastSeq + 1) & 0xff;
    if (seq !== expectedSeq) {
      linkStats.seqGaps += 1;
    }
  }
  linkStats.lastSeq = seq;
  linkStats.validFrames += 1;

  knobState = {
    main: fromUint16LE(frame[2], frame[3]),
    x: fromUint16LE(frame[4], frame[5]),
    y: fromUint16LE(frame[6], frame[7]),
    sw: frame[8],
    seq: frame[9],
  };

  rawHistory.push({ main: knobState.main, x: knobState.x, y: knobState.y });
  if (rawHistory.length > 8) {
    rawHistory = rawHistory.slice(rawHistory.length - 8);
  }

  smoothedKnobState = {
    main: medianFromHistory('main', knobState.main),
    x: medianFromHistory('x', knobState.x),
    y: medianFromHistory('y', knobState.y),
    sw: knobState.sw,
    seq: knobState.seq,
  };

  renderViewfinder();

  if (!analysisPixels) {
    return;
  }

  const { cv1, cv2, pulses } = analyzeViewport();
  updateLocalOutputs(cv1, cv2, pulses);
  await sendCvFrame(cv1, cv2);
}

async function readSerialLoop() {
  try {
    while (reader && isPortOpen()) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      if (!value) {
        continue;
      }

      readBuffer = appendBytes(readBuffer, value);
      await parseFrames();
    }
  } catch (error) {
    if (isPortOpen()) {
      setSerialState('error');
      window.alert(`Serial read failed: ${error.message}`);
    }
  }
}

async function initializeSerialUi() {
  if (!('serial' in navigator)) {
    setSerialState('disconnected');
    return;
  }

  try {
    const grantedPorts = await navigator.serial.getPorts();
    if (grantedPorts.length > 0) {
      [port] = grantedPorts;
      setSerialState('ready');
      return;
    }
  } catch {
  }

  setSerialState('disconnected');
}

initializeSerialUi();
renderViewfinder();
