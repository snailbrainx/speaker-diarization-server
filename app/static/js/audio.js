// Audio in the browser: microphone capture (AudioWorklet), files decoded to 16 kHz mono and played out at real-time
// pace, 16-bit PCM conversion and WAV encoding.

export const STREAM_RATE = 16000;
const FRAME_SECONDS = 0.1;

export const micAvailable = () => !!(window.isSecureContext && navigator.mediaDevices?.getUserMedia);

export async function listMicrophones() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "audioinput");
}

export function rms(samples) {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / (samples.length || 1));
}

// Level for a meter, 0..1, on a log scale (about -60 dBFS to 0)
export const meterLevel = (r) => Math.max(0, Math.min(1, 1 + Math.log10(Math.max(r, 1e-6)) / 3));

export function toInt16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

// Microphone frames of 0.1 s. Asks for a 16 kHz context so the browser resamples properly; a browser that cannot
// record at that rate keeps the device's rate (the server accepts 8-48 kHz). onFrame(float32, sampleRate)
export class MicCapture {
  constructor({ deviceId, processing = false, onFrame }) {
    this.deviceId = deviceId;
    this.processing = processing;
    this.onFrame = onFrame;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: this.deviceId ? { exact: this.deviceId } : undefined,
        channelCount: 1,
        echoCancellation: this.processing,
        noiseSuppression: this.processing,
        autoGainControl: this.processing,
      },
    });
    try {
      this.ctx = new AudioContext({ sampleRate: STREAM_RATE });
      this.source = this.ctx.createMediaStreamSource(this.stream);
    } catch {
      await this.ctx?.close();
      this.ctx = new AudioContext();
      this.source = this.ctx.createMediaStreamSource(this.stream);
    }
    this.sampleRate = this.ctx.sampleRate;
    await this.ctx.audioWorklet.addModule(new URL("./pcm-worklet.js", import.meta.url));
    this.node = new AudioWorkletNode(this.ctx, "pcm-capture", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: 1,
      channelCountMode: "explicit",
      processorOptions: { frameSamples: Math.round(this.sampleRate * FRAME_SECONDS) },
    });
    this.node.port.onmessage = (e) => this.onFrame(e.data, this.sampleRate);
    // A silent path to the output keeps the graph running in every browser
    const mute = this.ctx.createGain();
    mute.gain.value = 0;
    this.source.connect(this.node).connect(mute).connect(this.ctx.destination);
    if (this.ctx.state === "suspended") await this.ctx.resume();
    this.label = this.stream.getAudioTracks()[0]?.label || "Microphone";
    return this;
  }

  stop() {
    if (this.node) this.node.port.onmessage = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close().catch(() => {});
    this.node = this.ctx = this.stream = null;
  }
}

// Decodes any file the browser can play into 16 kHz mono samples
export async function decodeFile(file) {
  const data = await file.arrayBuffer();
  const ctx = new AudioContext();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(data);
  } finally {
    ctx.close().catch(() => {});
  }
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * STREAM_RATE)), STREAM_RATE);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0);
}

// Plays decoded samples out as 0.1 s frames at `speed` times real time, as a microphone would deliver them.
export class FileFeeder {
  constructor(samples, { speed = 1, onFrame, onEnd }) {
    this.samples = samples;
    this.speed = speed;
    this.onFrame = onFrame;
    this.onEnd = onEnd;
    this.sent = 0;
    this.frame = Math.round(STREAM_RATE * FRAME_SECONDS);
    this.sampleRate = STREAM_RATE;
  }

  get duration() { return this.samples.length / STREAM_RATE; }
  get position() { return this.sent / STREAM_RATE; }

  start() {
    this.t0 = performance.now();
    this.timer = setInterval(() => this.tick(), 40);
    return this;
  }

  tick() {
    const due = Math.min(this.samples.length, Math.floor(((performance.now() - this.t0) / 1000) * STREAM_RATE * this.speed));
    while (this.sent < due) {
      const end = Math.min(this.sent + this.frame, this.samples.length);
      if (end - this.sent < this.frame && end < this.samples.length) break;  // wait for a whole frame
      this.onFrame(this.samples.slice(this.sent, end), STREAM_RATE);
      this.sent = end;
    }
    if (this.sent >= this.samples.length) {
      this.stop();
      this.onEnd?.();
    }
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
  }
}

export function encodeWav(chunks, sampleRate) {
  const length = chunks.reduce((a, c) => a + c.length, 0);
  const buf = new ArrayBuffer(44 + length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, length * 2, true);
  let o = 44;
  for (const c of chunks) {
    const pcm = toInt16(c);
    for (let i = 0; i < pcm.length; i++, o += 2) v.setInt16(o, pcm[i], true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
