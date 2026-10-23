// The live session: microphone (or a file played at real-time pace) -> WebSocket -> segments and speaker updates.
// One session at a time, kept outside the page so a recording carries on while other pages are open.
//
// Protocol (README "WS /api/v1/streaming/ws"): {"type": "start", sample_rate, format, diarize}, binary PCM frames,
// {"type": "stop"}. The server answers started, status (per frame), segment, speaker_update, error and completed.

import { api, streamUrl } from "./api.js";
import { MicCapture, FileFeeder, decodeFile, toInt16, rms, meterLevel } from "./audio.js";

const ACTIVE = new Set(["preparing", "connecting", "recording", "stopping"]);

class LiveSession extends EventTarget {
  constructor() {
    super();
    this.reset();
  }

  reset() {
    this.state = "idle";  // idle | preparing | connecting | recording | stopping | done | error
    this.conversationId = null;
    this.segments = [];
    this.byId = new Map();
    this.events = [];
    this.statusMessages = 0;
    this.sentSamples = 0;
    this.sampleRate = 16000;
    this.level = 0;
    this.vad = false;
    this.lags = [];
    this.error = null;
    this.completed = null;
    this.source = null;
    this.sourceLabel = "";
    this.fileDuration = null;
    this.speed = 1;
  }

  get active() { return ACTIVE.has(this.state); }
  get seconds() { return this.sentSamples / this.sampleRate; }

  emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }

  setState(state) {
    this.state = state;
    this.emit("state", state);
  }

  log(type, text) {
    const e = { t: new Date(), type, text };
    this.events.push(e);
    if (this.events.length > 500) this.events.shift();
    this.emit("log", e);
  }

  // opts: {source: "mic" | "file", deviceId, processing, file, speed, hear, diarize}
  async start(opts) {
    if (this.active) return;
    this.reset();
    this.source = opts.source;
    this.diarize = opts.diarize;
    this.setState("preparing");
    const pending = [];
    const onFrame = (samples) => {
      this.level = meterLevel(rms(samples));
      this.emit("level", this.level);
      if (this.state === "recording") this.send(samples);
      else if (this.state === "connecting" && pending.length < 100) pending.push(samples);
    };
    try {
      if (opts.source === "file") {
        this.log("info", `Decoding ${opts.file.name}…`);
        const samples = await decodeFile(opts.file);
        this.speed = opts.speed || 1;
        this.feeder = new FileFeeder(samples, { speed: this.speed, onFrame, onEnd: () => this.stop("end of the file") });
        this.sampleRate = this.feeder.sampleRate;
        this.sourceLabel = opts.file.name;
        this.fileDuration = this.feeder.duration;
        if (opts.hear) {
          this.playerUrl = URL.createObjectURL(opts.file);
          this.player = new Audio(this.playerUrl);
          this.player.playbackRate = this.speed;
        }
      } else {
        this.mic = await new MicCapture({ deviceId: opts.deviceId, processing: opts.processing, onFrame }).start();
        this.sampleRate = this.mic.sampleRate;
        this.sourceLabel = this.mic.label;
      }
    } catch (e) {
      const msg = e.name === "NotAllowedError" ? "Microphone permission was refused"
        : e.name === "NotFoundError" ? "No microphone was found"
        : e.name === "EncodingError" ? "The browser cannot decode this file"
        : e.message || String(e);
      return this.fail(msg);
    }
    if (this.state !== "preparing") return this.release();  // cancelled meanwhile
    this.setState("connecting");
    this.log("info", `${this.sourceLabel}: ${this.sampleRate} Hz → server as 16-bit PCM${opts.diarize ? "" : ", one speaker"}`);
    const ws = (this.ws = new WebSocket(streamUrl()));
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify({ type: "start", sample_rate: this.sampleRate, format: "int16", diarize: !!opts.diarize }));
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      this.onMessage(msg, pending);
    };
    ws.onclose = (ev) => this.onClose(ev);
  }

  onMessage({ type, data = {} }, pending) {
    if (type === "started") {
      this.conversationId = data.conversation_id;
      this.startedAt = Date.now();
      this.setState("recording");
      this.log("started", `Conversation #${data.conversation_id}, ${data.sample_rate} Hz`);
      pending.splice(0).forEach((s) => this.send(s));
      if (this.feeder) {
        this.feeder.start();
        this.player?.play().catch(() => {});
      }
    } else if (type === "status") {
      this.statusMessages++;
      this.vad = !!data.vad_active;
      this.processed = data.stats?.segments_processed ?? this.processed;
      this.emit("status", data);
    } else if (type === "segment") {
      const seg = {
        id: data.segment_id, conversation_id: this.conversationId, speaker_id: data.speaker_id ?? null,
        speaker_name: data.speaker_name, text: data.text, start_offset: data.start_offset, end_offset: data.end_offset,
        confidence: data.confidence, emotion_category: data.emotion_category, emotion_confidence: data.emotion_confidence,
        detector_breakdown: data.detector_breakdown, words: data.words, avg_logprob: data.avg_logprob, is_known: data.is_known,
        // how long after its speech ended the line arrived, in seconds of wall-clock time
        lag: Math.max(0, (this.seconds - data.end_offset) / this.speed),
      };
      this.lags.push(seg.lag);
      this.segments.push(seg);
      this.byId.set(seg.id, seg);
      this.emit("segment", seg);
      this.log("segment", `${seg.speaker_name}: ${seg.text}  (+${seg.lag.toFixed(1)} s)`);
    } else if (type === "speaker_update") {
      const changed = [];
      for (const id of data.segment_ids || []) {
        const seg = this.byId.get(id);
        if (!seg) continue;
        changed.push({ seg, before: { speaker_id: seg.speaker_id, speaker_name: seg.speaker_name } });
        Object.assign(seg, { speaker_id: data.speaker_id, speaker_name: data.speaker_name, is_known: data.is_known },
          data.emotions?.[id] || {});   // read again with the speaker's emotion corrections
      }
      this.emit("speaker_update", { update: data, changed });
      this.log("speaker_update", `${data.segment_ids?.length || 0} line(s) → ${data.speaker_name}${data.is_known ? " (known)" : ""}`);
    } else if (type === "error") {
      this.error = data.message;
      this.log("error", data.message || "error");
      this.emit("error", data.message);
    } else if (type === "completed") {
      this.completed = data;
      this.log("completed", `${data.message || "Completed"}: ${data.num_segments} lines, ${data.num_speakers} speakers`);
      this.setState("done");
    } else {
      this.log(type, JSON.stringify(data));
    }
  }

  send(samples) {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(toInt16(samples).buffer);
    this.sentSamples += samples.length;
  }

  stop(reason) {
    if (this.state === "preparing") {
      this.setState("idle");
      return this.release();
    }
    if (this.state !== "recording" && this.state !== "connecting") return;
    this.release();
    this.endedAt = Date.now();
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "stop" }));
    this.setState("stopping");
    this.log("info", `Stopped${reason ? ` (${reason})` : ""}; the server is finishing the last lines`);
  }

  onClose(ev) {
    this.release();
    if (this.state === "stopping") {
      if (!this.completed) this.log("info", `Connection closed (${ev.code}) before the server confirmed the end`);
      this.setState("done");
    } else if (this.active) {
      this.fail(this.error || `The connection closed unexpectedly (code ${ev.code})`);
    }
  }

  fail(message) {
    this.release();
    this.error = message;
    this.log("error", message);
    try { this.ws?.close(); } catch { /* already closed */ }
    this.setState("error");
  }

  // Stops the audio source; the socket stays open for the server's last messages
  release() {
    this.mic?.stop();
    this.feeder?.stop();
    this.player?.pause();
    if (this.playerUrl) URL.revokeObjectURL(this.playerUrl);
    this.mic = this.feeder = this.player = this.playerUrl = null;
  }

  // Pulls names, emotions and corrections back from the server (after naming a voice mid-session, say)
  async sync() {
    if (!this.conversationId) return;
    const conv = await api.conversation(this.conversationId);
    const changed = [];
    for (const s of conv.transcript_segments || []) {
      const seg = this.byId.get(s.id);
      if (!seg) continue;
      if (seg.speaker_id !== s.speaker_id || seg.speaker_name !== s.speaker_name || seg.emotion_category !== s.emotion_category) {
        changed.push({ seg, before: { speaker_id: seg.speaker_id, speaker_name: seg.speaker_name } });
      }
      Object.assign(seg, {
        speaker_id: s.speaker_id, speaker_name: s.speaker_name, confidence: s.confidence, emotion_category: s.emotion_category,
        emotion_confidence: s.emotion_confidence, emotion_corrected: s.emotion_corrected, is_misidentified: s.is_misidentified,
      });
    }
    if (changed.length) this.emit("speaker_update", { update: null, changed });
  }
}

export const live = new LiveSession();
