// The server's REST API (/api/v1). Every call returns parsed JSON or throws an ApiError carrying the server's message.

const BASE = "/api/v1";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

async function errorMessage(res) {
  try {
    const body = await res.json();
    const d = body.detail ?? body.message;
    if (Array.isArray(d)) return d.map((x) => x.msg || JSON.stringify(x)).join("; ");
    if (d) return typeof d === "string" ? d : JSON.stringify(d);
  } catch { /* not JSON */ }
  return `${res.status} ${res.statusText || "error"}`;
}

function url(path, query) {
  const q = query && Object.entries(query).filter(([, v]) => v !== undefined && v !== null);
  return BASE + path + (q && q.length ? "?" + new URLSearchParams(q).toString() : "");
}

async function call(method, path, { json, form, query } = {}) {
  const opts = { method, headers: {} };
  if (json !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(json);
  } else if (form) {
    opts.body = form;
  }
  let res;
  try {
    res = await fetch(url(path, query), opts);
  } catch {
    throw new ApiError(0, "The server cannot be reached");
  }
  if (!res.ok) throw new ApiError(res.status, await errorMessage(res));
  return (res.headers.get("content-type") || "").includes("json") ? res.json() : res;
}

// Upload with progress (fetch cannot report upload progress). onProgress(fraction), then onUploaded() once the
// server has the whole file and is processing it.
function upload(path, form, { onProgress, onUploaded } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url(path));
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.upload.onload = () => onUploaded?.();
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve(xhr.response);
      const d = xhr.response?.detail;
      reject(new ApiError(xhr.status, typeof d === "string" ? d : d ? JSON.stringify(d) : `${xhr.status} error`));
    };
    xhr.onerror = () => reject(new ApiError(0, "The server cannot be reached"));
    xhr.send(form);
  });
}

const formOf = (fields) => {
  const f = new FormData();
  for (const [k, v] of Object.entries(fields)) {
    if (v === undefined || v === null) continue;
    if (Array.isArray(v)) f.append(k, v[0], v[1]);  // [blob, filename]
    else f.append(k, v);
  }
  return f;
};

export const api = {
  status: () => call("GET", "/status"),

  // speakers
  speakers: () => call("GET", "/speakers"),
  enroll: (name, file, filename) => call("POST", "/speakers/enroll", { form: formOf({ name, audio_file: [file, filename || file.name || "sample.wav"] }) }),
  renameSpeaker: (id, newName) => call("PATCH", `/speakers/${id}/rename`, { json: { new_name: newName } }),
  deleteSpeaker: (id) => call("DELETE", `/speakers/${id}`),
  deleteUnknownSpeakers: (maxSeconds) => call("DELETE", "/speakers/unknown/all", { query: { max_seconds: maxSeconds } }),
  unknownMatches: () => call("GET", "/speakers/unknown/matches"),
  speakerSegments: (id, { limit = 20, order = "recent" } = {}) => call("GET", `/speakers/${id}/segments`, { query: { limit, order } }),

  // files
  process: (file, diarize, handlers) => upload("/process", formOf({ audio_file: file, diarize: String(diarize) }), handlers),

  // conversations
  conversations: (query) => call("GET", "/conversations", { query }),
  conversation: (id) => call("GET", `/conversations/${id}`),
  renameConversation: (id, title) => call("PATCH", `/conversations/${id}`, { json: { title } }),
  deleteConversation: (id) => call("DELETE", `/conversations/${id}`),
  reprocess: (id) => call("POST", `/conversations/${id}/reprocess`),
  recalculateEmotions: (id) => call("POST", `/conversations/${id}/recalculate-emotions`),
  conversationAudioUrl: (id) => url(`/conversations/${id}/audio`),
  segmentAudioUrl: (segmentId) => url(`/conversations/segments/${segmentId}/audio`),
  identify: (convId, segId, body) => call("POST", `/conversations/${convId}/segments/${segId}/identify`, { json: { enroll: true, ...body } }),
  setMisidentified: (convId, segId, value) =>
    call("PATCH", `/conversations/${convId}/segments/${segId}/misidentified`, { json: { is_misidentified: value } }),
  setEmotionMisidentified: (convId, segId, value) =>
    call("PATCH", `/conversations/${convId}/segments/${segId}/emotion-misidentified`, { json: { is_misidentified: value } }),
  correctEmotion: (convId, segId, emotion, learn = true) =>
    call("POST", `/conversations/${convId}/segments/${segId}/correct-emotion`, { query: { corrected_emotion: emotion, learn } }),

  // per-speaker emotion profiles
  emotionProfiles: (speakerId) => call("GET", `/conversations/speakers/${speakerId}/emotion-profiles`),
  setEmotionThreshold: (speakerId, threshold) =>
    call("PATCH", `/conversations/speakers/${speakerId}/emotion-threshold`, { query: { threshold } }),
  setProfileThreshold: (speakerId, cat, threshold) =>
    call("PATCH", `/conversations/speakers/${speakerId}/emotion-profiles/${encodeURIComponent(cat)}/threshold`, { query: { threshold } }),
  setProfileVoiceThreshold: (speakerId, cat, threshold) =>
    call("PATCH", `/conversations/speakers/${speakerId}/emotion-profiles/${encodeURIComponent(cat)}/voice-threshold`, { query: { threshold } }),
  resetEmotionProfiles: (speakerId, cat) =>
    call("DELETE", `/conversations/speakers/${speakerId}/emotion-profiles`, { query: { emotion_category: cat } }),

  // settings
  settings: () => call("GET", "/settings/voice"),
  updateSettings: (values) => call("POST", "/settings/voice", { json: values }),
  resetSettings: () => call("POST", "/settings/voice/reset"),

  // backups ("voice profiles" in the API: named snapshots of speakers, segments' speakers and settings)
  profiles: () => call("GET", "/profiles"),
  createProfile: (name, description) => call("POST", "/profiles", { json: { name, description } }),
  saveAsProfile: (name, description) => call("POST", "/profiles/duplicate", { json: { name, description } }),
  updateProfile: (name, description) => call("PATCH", `/profiles/${encodeURIComponent(name)}`, { json: { description } }),
  deleteProfile: (name) => call("DELETE", `/profiles/${encodeURIComponent(name)}`),
  checkpoints: (name) => call("GET", `/profiles/${encodeURIComponent(name)}/checkpoints`),
  createCheckpoint: (name) => call("POST", `/profiles/${encodeURIComponent(name)}/checkpoints`),
  deleteCheckpoint: (name, ts) => call("DELETE", `/profiles/${encodeURIComponent(name)}/checkpoints/${encodeURIComponent(ts)}`),
  restore: (filename) => call("POST", "/profiles/restore", { query: { filename } }),
  importProfile: (file) => call("POST", "/profiles/import", { form: formOf({ file }) }),
  profileDownloadUrl: (name) => url(`/profiles/download/${encodeURIComponent(name)}`),
  allProfilesDownloadUrl: () => url("/profiles/download-all"),
};

export const streamUrl = () => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${BASE}/streaming/ws`;
