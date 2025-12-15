const outEl = document.getElementById("out");

function show(obj) {
  outEl.textContent = JSON.stringify(obj, null, 2);
}

async function postJSON(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await resp.json();
  if (!resp.ok) throw data;
  return data;
}

async function postFile(url, file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(url, { method: "POST", body: form });
  const data = await resp.json();
  if (!resp.ok) throw data;
  return data;
}

/* ====== TEXT MODE ====== */
document.getElementById("btnText").addEventListener("click", async () => {
  try {
    show({status: "running..."});
    const label = document.getElementById("label").value;
    const data = await postJSON("/predict_text", {label});
    show(data);
  } catch (e) {
    console.error(e);
    show({error: e});
  }
});

/* ====== FILE MODE ====== */
document.getElementById("btnFile").addEventListener("click", async () => {
  try {
    const input = document.getElementById("file");
    if (!input.files || !input.files[0]) {
      show({error: "Select a .wav file first"});
      return;
    }
    show({status: "uploading..."});
    const data = await postFile("/predict_audio", input.files[0]);
    show(data);
  } catch (e) {
    console.error(e);
    show({error: e});
  }
});

/* ====== MIC MODE (record -> WAV -> upload) ====== */
const btnRec = document.getElementById("btnRec");
const btnStop = document.getElementById("btnStop");

let audioCtx = null;
let stream = null;
let source = null;
let processor = null;
let chunks = [];
let recording = false;

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = (s < 0 ? s * 32768 : s * 32767) | 0;
  }
  return out;
}

function encodeWAV(samplesInt16, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample; // mono
  const buffer = new ArrayBuffer(44 + samplesInt16.length * bytesPerSample);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samplesInt16.length * bytesPerSample, true);
  writeString(8, "WAVE");

  writeString(12, "fmt ");
  view.setUint32(16, 16, true);  // PCM
  view.setUint16(20, 1, true);   // format
  view.setUint16(22, 1, true);   // channels
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);  // bits

  writeString(36, "data");
  view.setUint32(40, samplesInt16.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < samplesInt16.length; i++, offset += 2) {
    view.setInt16(offset, samplesInt16[i], true);
  }
  return new Blob([view], { type: "audio/wav" });
}

async function startRec() {
  try {
    show({status: "Requesting microphone permission..."});

    stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    source = audioCtx.createMediaStreamSource(stream);

    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    chunks = [];
    recording = true;

    processor.onaudioprocess = (e) => {
      if (!recording) return;
      const input = e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(input));
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    btnRec.disabled = true;
    btnStop.disabled = false;

    show({status: "Recording... press Stop"});
  } catch (err) {
    console.error(err);
    show({error: "Mic error", details: String(err)});
    btnRec.disabled = false;
    btnStop.disabled = true;
  }
}

async function stopRec() {
  try {
    recording = false;
    btnStop.disabled = true;
    show({status: "Stopping..."});

    if (processor) processor.disconnect();
    if (source) source.disconnect();

    if (stream) stream.getTracks().forEach(t => t.stop());

    const totalLen = chunks.reduce((s, a) => s + a.length, 0);
    const merged = new Float32Array(totalLen);
    let off = 0;
    for (const c of chunks) { merged.set(c, off); off += c.length; }

    const sr = audioCtx.sampleRate;

    if (audioCtx) await audioCtx.close();

    const pcm16 = floatTo16BitPCM(merged);
    const wavBlob = encodeWAV(pcm16, sr);

    // надсилаємо як файл "mic.wav"
    const file = new File([wavBlob], "mic.wav", { type: "audio/wav" });

    show({status: "Uploading mic.wav to /predict_audio ..."});
    const data = await postFile("/predict_audio", file);
    show(data);

  } catch (err) {
    console.error(err);
    show({error: "Stop error", details: String(err)});
  } finally {
    btnRec.disabled = false;
    btnStop.disabled = true;
  }
}

btnRec.addEventListener("click", startRec);
btnStop.addEventListener("click", stopRec);
