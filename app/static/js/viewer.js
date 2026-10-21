// One conversation: audio player with a speaker timeline, talk time and emotions, and the transcript with every
// correction the API offers (identify a speaker, correct an emotion, exclude a segment from a voice). Used for saved
// conversations, upload results and finished live sessions. segmentRow and the dialogs are shared with the live page.

import { api } from "./api.js";
import {
  h, clear, icon, button, busy, toast, toastError, modal, confirmDialog, promptDialog, openMenu, downloadText,
  fmtClock, fmtDuration, fmtDate, pct, isUnknown, isUnnamedVoice, displayName, Palette, speakerChip, emotionChip, emotionInfo, emotionColor,
  EMOTIONS, loading, idColor,
} from "./ui.js";

export const speakerKey = (s) => (s.speaker_id != null ? `id:${s.speaker_id}` : `name:${s.speaker_name || "Unknown"}`);

// Word times are used for highlighting only when they sit inside the segment (live segments saved before word times
// were stored from the conversation start count from their chunk instead).
function timedWords(seg) {
  const w = seg.words;
  if (!w || !w.length) return null;
  const s = seg.start_offset, e = seg.end_offset;
  for (const x of w) {
    if (typeof x.start !== "number" || typeof x.end !== "number" || x.start < s - 1.5 || x.end > e + 1.5) return null;
  }
  return w;
}

export function segmentText(seg) {
  const p = h("p", { class: "seg-text" });
  const words = seg.words && seg.words.length ? seg.words : null;
  if (!words) {
    p.append(seg.text || "");
    return p;
  }
  const timed = timedWords(seg);
  words.forEach((w, i) => {
    const text = String(w.word ?? "");
    const lead = i === 0 ? text.trimStart() : text;
    const space = lead.match(/^\s+/)?.[0] || (i && !/^[\s,.!?;:'’)]/.test(lead) ? " " : "");
    if (space) p.append(space);
    const low = typeof w.probability === "number" && w.probability < 0.5;
    p.append(h("span", {
      class: `w${low ? " low" : ""}`,
      dataset: timed ? { s: w.start, e: w.end } : null,
      title: typeof w.probability === "number" ? `${pct(w.probability)} sure${timed ? ` · ${fmtClock(w.start, 1)}` : ""}` : null,
    }, lead.trimStart()));
  });
  return p;
}

// A transcript line. ctx: {palette, onPlay(seg), onSpeaker(seg), onEmotion(seg), onMenu(seg, anchor), isNew}
export function segmentRow(seg, ctx) {
  const color = ctx.palette.color(speakerKey(seg));
  const unknown = isUnknown(seg.speaker_name);
  const head = h("div", { class: "seg-head" },
    speakerChip(seg.speaker_name || "Unknown", color, {
      id: seg.speaker_id,
      onClick: ctx.onSpeaker ? () => ctx.onSpeaker(seg) : null,
      title: unknown ? "Not identified yet: click to name this voice" : "Click to reassign this line",
    }),
    emotionChip(seg.emotion_category, seg.emotion_confidence, {
      corrected: seg.emotion_corrected,
      onClick: ctx.onEmotion ? () => ctx.onEmotion(seg) : null,
    }),
    !unknown && seg.confidence != null && seg.confidence < 0.999
      ? h("span", { class: "conf", title: "How closely this voice matched the speaker's profile" }, `match ${pct(seg.confidence)}`) : null,
    seg.is_overlap ? h("span", { class: "badge info", title: "A second voice talking over someone, separated and transcribed" }, "overlap") : null,
    seg.is_misidentified ? h("span", { class: "badge warn", title: "Marked as the wrong speaker: left out of their voice profile" }, "excluded") : null,
    seg.emotion_misidentified ? h("span", { class: "badge warn", title: "Emotion correction marked as wrong" }, "emotion excluded") : null,
  );
  const actions = h("div", { class: "seg-actions" },
    ctx.onPlay ? button("", { kind: "ghost", size: "sm", icon: "play", title: "Play this line", onClick: () => ctx.onPlay(seg) }) : null,
    ctx.onSpeaker ? button("", { kind: "ghost", size: "sm", icon: "identify", title: "Who is speaking?", onClick: () => ctx.onSpeaker(seg) }) : null,
    ctx.onEmotion ? button("", { kind: "ghost", size: "sm", icon: "smile", title: "Correct the emotion", onClick: () => ctx.onEmotion(seg) }) : null,
    ctx.onMenu ? button("", { kind: "ghost", size: "sm", icon: "more", title: "More", onClick: (e) => ctx.onMenu(seg, e.currentTarget) }) : null,
  );
  const row = h("div", {
    class: `seg${ctx.isNew ? " new" : ""}${seg.is_misidentified ? " dim" : ""}`,
    style: { "--c": color }, dataset: { id: seg.id ?? seg.segment_id },
  },
    h("div", { class: "seg-time", title: "Play from here", onclick: ctx.onPlay ? () => ctx.onPlay(seg, true) : null },
      fmtClock(seg.start_offset, 1)),
    h("div", null, head, segmentText(seg)),
    actions);
  return row;
}

// ---------- dialogs (shared with the live page) ----------

function segmentQuote(seg, color) {
  const play = button("", { kind: "ghost", size: "sm", icon: "play", title: "Listen" });
  let audio;
  play.addEventListener("click", () => {
    audio?.pause();
    audio = new Audio(api.segmentAudioUrl(seg.id));
    audio.play().catch(() => toast("This line's audio is not available yet", "err"));
  });
  return h("div", { class: "quote row", style: { "--c": color, flexWrap: "nowrap", alignItems: "flex-start" } },
    play, h("div", null, h("div", { class: "small muted" }, `${fmtClock(seg.start_offset, 1)} – ${fmtClock(seg.end_offset, 1)}`), seg.text || "…"));
}

// Name the voice of a line, or move the line to someone else. Resolves to the API's answer, or null.
export async function identifyDialog(conversationId, seg, color, { sameVoiceCount } = {}) {
  let speakers = [];
  try {
    speakers = (await api.speakers()).filter((s) => !isUnknown(s.name) && !s.name.startsWith("Deleted_"));
  } catch (e) { toastError(e); }
  speakers.sort((a, b) => a.name.localeCompare(b.name));
  const voice = isUnnamedVoice(seg.speaker_name);
  let chosen = null;
  const list = h("div", { class: "pick-list" });
  const nameInput = h("input", { class: "input", placeholder: "Someone new: type their name", maxlength: 80 });
  const filter = h("input", { class: "input sm", placeholder: "Search people…" });
  const draw = () => {
    clear(list);
    const q = filter.value.trim().toLowerCase();
    const shown = speakers.filter((s) => !q || s.name.toLowerCase().includes(q));
    if (!shown.length) list.append(h("div", { class: "small muted", style: { padding: "8px 10px" } }, speakers.length ? "Nobody matches." : "Nobody is enrolled yet: type a name below."));
    for (const s of shown) {
      list.append(h("button", {
        type: "button", class: `pick${chosen?.id === s.id ? " on" : ""}`,
        onclick: () => { chosen = s; nameInput.value = ""; draw(); },
        disabled: s.id === seg.speaker_id,
      }, speakerChip(s.name, idColor(s.id)), h("span", { class: "spacer" }), h("span", { class: "small muted" }, `${s.segment_count} lines`)));
    }
  };
  filter.addEventListener("input", draw);
  nameInput.addEventListener("input", () => { if (nameInput.value) { chosen = null; draw(); } });
  draw();
  const name = displayName(seg.speaker_name, seg.speaker_id);
  const scope = voice && seg.speaker_id != null
    ? `Every line of ${name}${sameVoiceCount ? ` (${sameVoiceCount} here)` : ""}, in every conversation, moves to this person, and their voice profile learns from them. From then on this voice is recognised by name.`
    : voice ? `Every line labelled ${name} in this conversation moves to this person, and their voice profile learns from them.`
    : isUnknown(seg.speaker_name) ? "Only this line moves to this person."
    : `Only this line moves; ${name}'s other lines stay theirs, and both voice profiles are rebuilt.`;
  const body = h("div", { class: "stack" },
    segmentQuote(seg, color),
    h("div", { class: "field" }, h("label", null, "Who is speaking?"), filter, list),
    h("div", { class: "field" }, h("label", null, "Or someone new"), nameInput),
    h("div", { class: "notice info" }, icon("info"), h("div", null, scope)));
  let result = null;
  const closed = new Promise((resolve) => modal({
    title: voice ? "Name this voice" : isUnknown(seg.speaker_name) ? "Who said this?" : "Reassign this line",
    body,
    actions: [{
      label: "Identify", kind: "primary", icon: "identify",
      onClick: async () => {
        const name = nameInput.value.trim();
        if (!chosen && !name) { toast("Pick a person or type a name", "err"); return false; }
        result = await api.identify(conversationId, seg.id, chosen ? { speaker_id: chosen.id } : { speaker_name: name });
        toast(result.message, "ok", 6000);
      },
    }],
    onClose: resolve,
  }));
  setTimeout(() => (speakers.length ? filter : nameInput).focus(), 30);
  await closed;
  return result;
}

export async function emotionDialog(conversationId, seg) {
  let chosen = seg.emotion_category;
  const grid = h("div", { class: "emo-grid" });
  const draw = () => {
    clear(grid);
    for (const [cat, info] of Object.entries(EMOTIONS)) {
      grid.append(h("button", {
        type: "button", class: `emo-pick${chosen === cat ? " on" : ""}`, style: { "--c": emotionColor(cat) },
        onclick: () => { chosen = cat; draw(); },
      }, h("span", { class: "e" }, info.e), info.label));
    }
  };
  draw();
  const canLearn = seg.speaker_id != null;
  const learn = h("label", { class: "switch" }, h("input", { type: "checkbox", checked: canLearn, disabled: !canLearn }),
    h("span", null, `Learn how ${displayName(seg.speaker_name, seg.speaker_id)} sounds when ${chosen ? "like this" : "feeling this"}`,
      h("span", { class: "desc" }, canLearn
        ? "Adds this line to the speaker's personal emotion profile, so their future lines are judged against it."
        : "Identify the speaker first: emotion profiles belong to a person.")));
  const wrong = seg.emotion_corrected ? h("label", { class: "switch" },
    h("input", { type: "checkbox", checked: !!seg.emotion_misidentified }),
    h("span", null, "This correction was a mistake", h("span", { class: "desc" }, "Leaves it out of the speaker's emotion profile."))) : null;
  const bd = seg.detector_breakdown;
  const detected = h("div", { class: "small muted" },
    seg.emotion_category ? ["Detected: ", emotionInfo(seg.emotion_category).label, seg.emotion_confidence != null ? ` (${pct(seg.emotion_confidence)})` : "",
      seg.emotion_corrected ? " · corrected by hand" : ""] : "No emotion was detected for this line.");
  const body = h("div", { class: "stack" }, segmentQuote(seg, "var(--accent)"), detected, grid, learn, wrong,
    bd ? h("details", { class: "collapse" }, h("summary", { class: "small muted" }, "Detector details"), h("pre", { class: "json" }, JSON.stringify(bd, null, 2))) : null);
  let changed = false;
  await new Promise((resolve) => modal({
    title: "Emotion", body,
    actions: [{
      label: "Save", kind: "primary", icon: "check",
      onClick: async () => {
        if (wrong && wrong.querySelector("input").checked !== !!seg.emotion_misidentified) {
          await api.setEmotionMisidentified(conversationId, seg.id, wrong.querySelector("input").checked);
          changed = true;
        }
        if (chosen && (chosen !== seg.emotion_category || !seg.emotion_corrected)) {
          if (!canLearn) { toast("Identify the speaker before correcting the emotion", "err"); return false; }
          const r = await api.correctEmotion(conversationId, seg.id, chosen, learn.querySelector("input").checked);
          toast(r.message || "Emotion corrected", "ok");
          changed = true;
        }
      },
    }],
    onClose: resolve,
  }));
  return changed;
}

export function detailsDialog(seg) {
  const kv = (k, v) => [h("dt", null, k), h("dd", null, v ?? "–")];
  const words = seg.words || [];
  modal({
    title: `Line ${seg.id}`, wide: true, cancelLabel: "Close",
    body: h("div", { class: "stack" },
      h("dl", { class: "kvlist" },
        kv("Time", `${fmtClock(seg.start_offset, 2)} – ${fmtClock(seg.end_offset, 2)} (${(seg.end_offset - seg.start_offset).toFixed(2)} s)`),
        kv("Speaker", `${displayName(seg.speaker_name, seg.speaker_id)}${seg.speaker_name && seg.speaker_name !== displayName(seg.speaker_name, seg.speaker_id) ? ` (${seg.speaker_name})` : seg.speaker_id != null ? ` (#${seg.speaker_id})` : ""}`),
        kv("Voice match", seg.confidence != null ? pct(seg.confidence, 1) : null),
        kv("Speech-to-text", seg.avg_logprob != null ? `average log-probability ${seg.avg_logprob.toFixed(3)}` : null),
        kv("Emotion", seg.emotion_category ? `${emotionInfo(seg.emotion_category).label} ${seg.emotion_confidence != null ? pct(seg.emotion_confidence, 1) : ""}${seg.emotion_corrected ? " · corrected" : ""}` : null),
        kv("Flags", [seg.is_overlap && "overlapping speech", seg.is_misidentified && "excluded from the voice profile",
          seg.emotion_misidentified && "emotion correction excluded"].filter(Boolean).join(", ") || "none")),
      words.length ? h("details", { class: "collapse", open: words.length <= 40 },
        h("summary", { class: "label" }, `Words (${words.length})`),
        h("div", { class: "table-wrap" }, h("table", { class: "table" },
          h("thead", null, h("tr", null, h("th", null, "Word"), h("th", null, "Start"), h("th", null, "End"), h("th", null, "Sure"))),
          h("tbody", null, words.map((w) => h("tr", null, h("td", null, w.word), h("td", { class: "mono" }, w.start?.toFixed?.(2) ?? "–"),
            h("td", { class: "mono" }, w.end?.toFixed?.(2) ?? "–"),
            h("td", { class: "num", style: { color: w.probability < 0.5 ? "var(--warning)" : null } }, w.probability != null ? pct(w.probability) : "–"))))))) : null,
      seg.detector_breakdown ? h("details", { class: "collapse" }, h("summary", { class: "label" }, "Emotion detectors"),
        h("pre", { class: "json" }, JSON.stringify(seg.detector_breakdown, null, 2))) : null),
  });
}

// ---------- exports ----------

const srtTime = (s) => {
  const ms = Math.round(s * 1000);
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(Math.floor(ms / 3600000))}:${p(Math.floor(ms / 60000) % 60)}:${p(Math.floor(ms / 1000) % 60)},${p(ms % 1000, 3)}`;
};

function exportAs(conv, segs, kind) {
  const base = (conv.title || `conversation-${conv.id}`).replace(/[^\w.-]+/g, "_").slice(0, 80);
  if (kind === "json") return downloadText(`${base}.json`, JSON.stringify(conv, null, 2), "application/json");
  if (kind === "srt") {
    return downloadText(`${base}.srt`, segs.map((s, i) =>
      `${i + 1}\n${srtTime(s.start_offset)} --> ${srtTime(s.end_offset)}\n${displayName(s.speaker_name, s.speaker_id)}: ${s.text || ""}\n`).join("\n"));
  }
  const lines = [conv.title || `Conversation ${conv.id}`, `${fmtDate(conv.start_time)} · ${fmtDuration(conv.duration)}`, ""];
  for (const s of segs) {
    const emo = s.emotion_category ? ` (${emotionInfo(s.emotion_category).label.toLowerCase()})` : "";
    lines.push(`[${fmtClock(s.start_offset)}] ${displayName(s.speaker_name, s.speaker_id)}${emo}: ${s.text || ""}`);
  }
  downloadText(`${base}.txt`, lines.join("\n") + "\n");
}

// ---------- the view ----------

function niceStep(duration, ticks) {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
  return steps.find((s) => duration / s <= ticks) || 7200;
}

export function speakerStats(segs) {
  const map = new Map();
  for (const s of segs) {
    const k = speakerKey(s);
    if (!map.has(k)) map.set(k, { key: k, id: s.speaker_id, name: s.speaker_name || "Unknown", segs: [], talk: 0 });
    const st = map.get(k);
    st.segs.push(s);
    st.talk += Math.max(0, s.end_offset - s.start_offset);
  }
  return [...map.values()];
}

// opts: {onDeleted(), title link etc.}. Returns {el, reload(), destroy()}.
export function conversationView(initial, opts = {}) {
  const el = h("div", { class: "stack", style: { gap: "16px" } });
  const palette = new Palette();
  let conv = initial;
  let segs = [];
  let duration = 0;
  let audioOk = false;
  let stopAt = null;
  let raf = 0;
  let follow = true;
  let currentRow = null;
  let currentWord = null;
  let query = "";
  let onlySpeaker = null;
  let segAudio = null;
  const rows = new Map();

  const audio = new Audio();
  audio.preload = "metadata";
  const loadAudio = () => {
    audioOk = false;
    if (["recording", "deleting"].includes(conv.status)) return;
    audio.src = api.conversationAudioUrl(conv.id);
  };
  audio.addEventListener("loadedmetadata", () => {
    audioOk = true;
    if (!duration && isFinite(audio.duration)) duration = audio.duration;
    renderPlayer();
  });
  audio.addEventListener("error", () => { audioOk = false; renderPlayer(); });
  audio.addEventListener("play", () => { tick(); syncPlayButton(); });
  audio.addEventListener("pause", syncPlayButton);
  audio.addEventListener("ended", syncPlayButton);

  // Parts that re-render independently
  const headEl = h("div");
  const playerEl = h("div", { class: "card player" });
  const summaryEl = h("div", { class: "summary" });
  const transcriptEl = h("div", { class: "card" });
  el.append(headEl, playerEl, summaryEl, transcriptEl);

  let playBtn, timeEl, playheadEl, tracksEl;

  function prepare() {
    segs = (conv.transcript_segments || []).slice().sort((a, b) => a.start_offset - b.start_offset || a.id - b.id);
    duration = Math.max(conv.duration || 0, ...segs.map((s) => s.end_offset), audioOk ? audio.duration || 0 : 0);
    // colours in order of first appearance, stable across re-renders
    segs.forEach((s) => palette.color(speakerKey(s)));
  }

  async function reload() {
    try {
      conv = await api.conversation(conv.id);
      prepare();
      renderAll();
      opts.onUpdated?.(conv);
    } catch (e) { toastError(e); }
  }

  // ----- header -----
  function renderHead() {
    const statusBadge = {
      completed: null, recording: h("span", { class: "badge err" }, "● recording"), processing: h("span", { class: "badge warn" }, "processing"),
      failed: h("span", { class: "badge err" }, "failed"), finalizing: h("span", { class: "badge warn" }, "finalizing"),
    }[conv.status] ?? h("span", { class: "badge" }, conv.status);
    const speakers = speakerStats(segs);
    const more = button("", { icon: "more", title: "More actions" });
    more.addEventListener("click", () => openMenu(more, [
      { label: "Recalculate emotions", icon: "smile", onClick: recalcEmotions },
      { label: "Reprocess the recording", icon: "refresh", onClick: reprocess },
      "-",
      { label: "Delete conversation", icon: "trash", danger: true, onClick: remove },
    ]));
    const exp = button("Export", { icon: "download" });
    exp.addEventListener("click", () => openMenu(exp, [
      { label: "Transcript (.txt)", icon: "file", onClick: () => exportAs(conv, segs, "txt") },
      { label: "Subtitles (.srt)", icon: "file", onClick: () => exportAs(conv, segs, "srt") },
      { label: "Everything (.json)", icon: "code", onClick: () => exportAs(conv, segs, "json") },
      audioOk ? { label: "Recording (audio)", icon: "volume", onClick: () => { location.href = api.conversationAudioUrl(conv.id); } } : null,
    ]));
    clear(headEl).append(h("div", { class: "conv-head" },
      h("div", { style: { minWidth: 0 } },
        opts.backLink !== false ? h("a", { class: "back", href: "#/conversations" }, icon("chevronLeft"), "Conversations") : null,
        h("div", { class: "conv-title" }, h("h1", null, conv.title || `Conversation ${conv.id}`),
          button("", { kind: "ghost", size: "sm", icon: "edit", title: "Rename", onClick: rename }), statusBadge),
        h("div", { class: "meta-row" },
          h("span", null, icon("calendar"), fmtDate(conv.start_time)),
          h("span", null, icon("clock"), fmtDuration(duration || conv.duration)),
          h("span", null, icon("users"), `${speakers.length} ${speakers.length === 1 ? "speaker" : "speakers"}`),
          h("span", null, icon("chat"), `${segs.length} lines`),
          h("span", { class: "mono" }, `#${conv.id}`))),
      h("div", { class: "row" }, exp, more)));
  }

  async function rename() {
    const title = await promptDialog({ title: "Rename conversation", label: "Title", value: conv.title || "" });
    if (title == null || title === conv.title) return;
    try {
      await api.renameConversation(conv.id, title);
      conv.title = title;
      renderHead();
      opts.onUpdated?.(conv);
    } catch (e) { toastError(e); }
  }

  async function recalcEmotions() {
    toast("Recalculating emotions…", "info", 2500);
    try {
      const r = await api.recalculateEmotions(conv.id);
      toast(`${r.message}: ${r.updated} updated, ${r.skipped} skipped`, "ok");
      await reload();
    } catch (e) { toastError(e); }
  }

  async function reprocess() {
    const ok = await confirmDialog({
      title: "Reprocess this recording?", confirmLabel: "Reprocess", icon: "refresh",
      message: h("div", { class: "stack" },
        h("p", null, "The whole pipeline runs again on the recording and every line is replaced: transcript, speakers and emotions."),
        h("p", { class: "muted" }, "People you named stay enrolled, so they are normally recognised again; corrections made to single lines of this conversation are lost.")),
    });
    if (!ok) return;
    toast("Reprocessing… this takes about as long as the first time", "info", 5000);
    try {
      const r = await api.reprocess(conv.id);
      toast(`${r.message}: ${r.segments} lines`, "ok");
      await reload();
    } catch (e) { toastError(e); }
  }

  async function remove() {
    const ok = await confirmDialog({
      title: "Delete this conversation?", danger: true, confirmLabel: "Delete", icon: "trash",
      message: "Its recording and transcript are removed. Speakers stay enrolled.",
    });
    if (!ok) return;
    try {
      await api.deleteConversation(conv.id);
      toast("Conversation deleted", "ok");
      destroy();
      opts.onDeleted?.();
    } catch (e) { toastError(e); }
  }

  // ----- player + timeline -----
  function renderPlayer() {
    clear(playerEl);
    playBtn = h("button", { class: "play-btn", title: "Play / pause (space)", disabled: !audioOk, onclick: togglePlay });
    timeEl = h("span", { class: "time" });
    const speed = h("select", { class: "select sm", style: { width: "84px" }, title: "Playback speed", onchange: (e) => { audio.playbackRate = Number(e.target.value); } },
      [0.75, 1, 1.25, 1.5, 2].map((r) => h("option", { value: r, selected: audio.playbackRate === r }, `${r}×`)));
    const followSw = h("label", { class: "switch small", title: "Keep the line being played in view" },
      h("input", { type: "checkbox", checked: follow, onchange: (e) => { follow = e.target.checked; } }), h("span", null, "Follow"));
    playerEl.append(h("div", { class: "player-bar" }, playBtn, timeEl, h("span", { class: "spacer" }),
      audioOk ? null : h("span", { class: "small muted" }, conv.status === "recording" ? "Audio is available once the recording ends" : "No recording to play; lines play from their own clips"),
      speed, followSw));
    syncPlayButton();

    const stats = speakerStats(segs);
    playerEl.classList.toggle("sticky", stats.length <= 4);
    const labels = h("div", { class: "tl-labels" });
    tracksEl = h("div", { class: `tl-tracks${audioOk ? "" : " no-audio"}` });
    const dur = duration || 1;
    for (const st of stats) {
      const color = palette.color(st.key);
      labels.append(h("div", null, speakerChip(st.name, color, { id: st.id, onClick: () => setOnlySpeaker(onlySpeaker === st.key ? null : st.key) })));
      const lane = h("div", { class: "lane" });
      for (const s of st.segs) {
        lane.append(h("div", {
          class: `blk${isUnknown(st.name) ? " unknown" : ""}${s.is_overlap ? " overlap" : ""}`,
          style: { left: `${(s.start_offset / dur) * 100}%`, width: `${Math.max(0.15, ((s.end_offset - s.start_offset) / dur) * 100)}%`, "--c": color },
          title: `${fmtClock(s.start_offset)} ${displayName(st.name, st.id)}: ${s.text || ""}`,
          onclick: (e) => { e.stopPropagation(); playSegment(s, true); scrollToRow(s.id); },
        }));
      }
      tracksEl.append(lane);
    }
    const ruler = h("div", { class: "ruler" });
    // about one time label per 80 px of timeline
    const step = niceStep(dur, Math.max(2, Math.floor((playerEl.clientWidth - 200) / 80)));
    for (let t = 0; t <= dur; t += step) ruler.append(h("span", { style: { left: `${(t / dur) * 100}%` } }, fmtClock(t)));
    tracksEl.append(ruler);
    playheadEl = h("div", { class: "playhead", style: { left: "0%" } });
    const hover = h("div", { class: "hover-time hidden" });
    tracksEl.append(playheadEl, hover);
    const timeAt = (e) => {
      const r = tracksEl.getBoundingClientRect();
      return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * dur;
    };
    tracksEl.addEventListener("mousemove", (e) => {
      const r = tracksEl.getBoundingClientRect();
      hover.classList.remove("hidden");
      hover.style.left = `${e.clientX - r.left}px`;
      hover.textContent = fmtClock(timeAt(e), 1);
    });
    tracksEl.addEventListener("mouseleave", () => hover.classList.add("hidden"));
    tracksEl.addEventListener("click", (e) => {
      if (!audioOk) return;
      stopAt = null;
      audio.currentTime = timeAt(e);
      updatePlayhead();
    });
    if (!stats.length) {
      playerEl.append(h("div", { class: "small muted", style: { marginTop: "12px" } }, "No speech was found in this recording."));
    } else {
      playerEl.append(h("div", { class: "timeline" }, labels, tracksEl));
    }
    updatePlayhead();
  }

  function syncPlayButton() {
    if (!playBtn) return;
    clear(playBtn).append(icon(audio.paused ? "play" : "pause"));
  }

  function togglePlay() {
    if (!audioOk) return;
    if (audio.paused) { stopAt = null; audio.play().catch(toastError); } else audio.pause();
  }

  function playSegment(s, fromHereOn = false) {
    if (audioOk) {
      segAudio?.pause();
      audio.currentTime = s.start_offset;
      stopAt = fromHereOn ? null : s.end_offset + 0.15;
      audio.play().catch(toastError);
    } else {
      segAudio?.pause();
      segAudio = new Audio(api.segmentAudioUrl(s.id));
      segAudio.play().catch(() => toast("This line's audio is not available", "err"));
    }
  }

  function updatePlayhead() {
    const t = audio.currentTime || 0;
    if (timeEl) timeEl.textContent = `${fmtClock(t, 1)} / ${fmtClock(duration)}`;
    if (playheadEl) playheadEl.style.left = `${(t / (duration || 1)) * 100}%`;
  }

  // Highlights the line and word being played (binary search over the sorted lines)
  function tick() {
    cancelAnimationFrame(raf);
    const step = () => {
      const t = audio.currentTime;
      if (stopAt != null && t >= stopAt) { audio.pause(); stopAt = null; }
      updatePlayhead();
      let lo = 0, hi = segs.length - 1, idx = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (segs[mid].start_offset <= t) { idx = mid; lo = mid + 1; } else hi = mid - 1;
      }
      const seg = idx >= 0 && t <= segs[idx].end_offset + 0.05 ? segs[idx] : null;
      const row = seg ? rows.get(seg.id) : null;
      if (row !== currentRow) {
        currentRow?.classList.remove("current");
        currentRow = row;
        if (row) {
          row.classList.add("current");
          if (follow && row.offsetParent) row.scrollIntoView({ block: "center", behavior: "smooth" });
        }
      }
      let word = null;
      if (row) {
        for (const w of row.querySelectorAll(".w[data-s]")) {
          if (+w.dataset.s <= t && t < +w.dataset.e) { word = w; break; }
        }
      }
      if (word !== currentWord) {
        currentWord?.classList.remove("now");
        currentWord = word;
        word?.classList.add("now");
      }
      if (!audio.paused) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }

  function scrollToRow(id) {
    const row = rows.get(id);
    if (row && row.offsetParent) row.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  // ----- summary -----
  function renderSummary() {
    clear(summaryEl);
    const stats = speakerStats(segs).sort((a, b) => b.talk - a.talk);
    const total = stats.reduce((a, s) => a + s.talk, 0) || 1;
    const talk = h("div", { class: "talk" });
    for (const st of stats) {
      const color = palette.color(st.key);
      const longest = st.segs.reduce((a, s) => (s.end_offset - s.start_offset > a.end_offset - a.start_offset ? s : a), st.segs[0]);
      talk.append(h("div", { class: "talk-row" },
        h("div", { class: "row", style: { flexWrap: "nowrap", minWidth: 0 } },
          speakerChip(st.name, color, { id: st.id, onClick: () => setOnlySpeaker(onlySpeaker === st.key ? null : st.key), title: "Show only their lines" }),
          isUnnamedVoice(st.name) && longest?.id != null ? button("Name", { size: "sm", kind: "ghost", icon: "identify", title: "Name this voice", onClick: () => onSpeaker(longest, st.segs.length) }) : null),
        h("div", { class: "bar", style: { "--c": color } }, h("i", { style: { width: `${(st.talk / total) * 100}%` } })),
        h("div", { class: "n" }, `${fmtDuration(st.talk)} · ${pct(st.talk / total)} · ${st.segs.length}`)));
    }
    const emo = new Map();
    let emoTotal = 0;
    for (const s of segs) {
      if (!s.emotion_category) continue;
      const d = Math.max(0.01, s.end_offset - s.start_offset);
      emo.set(s.emotion_category, (emo.get(s.emotion_category) || 0) + d);
      emoTotal += d;
    }
    const emoSorted = [...emo.entries()].sort((a, b) => b[1] - a[1]);
    summaryEl.append(
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, icon("users"), "Talk time"),
        h("span", { class: "small muted" }, "time · share · lines")), h("div", { class: "card-body" },
        stats.length ? talk : h("div", { class: "muted" }, "Nobody spoke."))),
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, icon("smile"), "Emotions"),
        h("span", { class: "small muted" }, "share of speaking time")), h("div", { class: "card-body" },
        emoTotal ? [
          h("div", { class: "emo-bar" }, emoSorted.map(([c, d]) => h("i", { style: { width: `${(d / emoTotal) * 100}%`, "--c": emotionColor(c) }, title: `${emotionInfo(c).label} ${pct(d / emoTotal)}` }))),
          h("div", { class: "emo-legend" }, emoSorted.map(([c, d]) => emotionChip(c, d / emoTotal, { title: `${emotionInfo(c).label}: ${pct(d / emoTotal)} of speaking time` }))),
        ] : h("div", { class: "muted" }, "No emotions detected."))));
  }

  // ----- transcript -----
  function setOnlySpeaker(key) {
    onlySpeaker = key;
    renderTranscript();
  }

  function renderTranscript() {
    clear(transcriptEl);
    rows.clear();
    currentRow = currentWord = null;
    const search = h("div", { class: "search" }, icon("search"),
      h("input", { class: "input sm", placeholder: "Search the transcript…", value: query,
        oninput: (e) => { query = e.target.value; applyFilter(); } }));
    const filterInfo = h("span", { class: "small muted" });
    const stats = speakerStats(segs);
    const chips = h("div", { class: "row", style: { gap: "6px" } },
      stats.length > 1 ? stats.map((st) => {
        const c = speakerChip(st.name, palette.color(st.key), { id: st.id, onClick: () => setOnlySpeaker(onlySpeaker === st.key ? null : st.key), title: "Show only their lines" });
        if (onlySpeaker && onlySpeaker !== st.key) c.style.opacity = ".4";
        return c;
      }) : null,
      onlySpeaker ? button("Everyone", { size: "sm", kind: "ghost", icon: "x", onClick: () => setOnlySpeaker(null) }) : null);
    const list = h("div", { class: "transcript" });
    const ctx = {
      palette,
      onPlay: playSegment,
      onSpeaker: (s) => onSpeaker(s),
      onEmotion: async (s) => { if (await emotionDialog(conv.id, s)) await reload(); },
      onMenu: (s, anchor) => openMenu(anchor, [
        { label: "Play this line", icon: "play", onClick: () => playSegment(s) },
        { label: "Who is speaking?", icon: "identify", onClick: () => onSpeaker(s) },
        { label: "Correct the emotion", icon: "smile", onClick: () => ctx.onEmotion(s) },
        s.speaker_id != null ? {
          label: s.is_misidentified ? "Use for their voice profile again" : "Wrong speaker: leave out of their voice",
          icon: "flag", onClick: () => toggleMisidentified(s),
        } : null,
        "-",
        { label: "Copy text", icon: "copy", onClick: () => navigator.clipboard?.writeText(s.text || "").then(() => toast("Copied", "ok", 1500)) },
        { label: "Details", icon: "info", onClick: () => detailsDialog(s) },
      ]),
    };
    for (const s of segs) {
      const row = segmentRow(s, ctx);
      rows.set(s.id, row);
      list.append(row);
    }
    transcriptEl.append(
      h("div", { class: "card-head" }, h("h2", null, icon("chat"), "Transcript"), filterInfo),
      h("div", { class: "transcript-tools" }, search, chips),
      segs.length ? list : h("div", { class: "empty" }, icon("chat"), h("h3", null, "No lines"), h("p", null, "Nothing was transcribed.")));
    function applyFilter() {
      const q = query.trim().toLowerCase();
      let shown = 0;
      for (const s of segs) {
        const ok = (!onlySpeaker || speakerKey(s) === onlySpeaker) && (!q || (s.text || "").toLowerCase().includes(q) || (s.speaker_name || "").toLowerCase().includes(q));
        rows.get(s.id).classList.toggle("hidden", !ok);
        if (ok) shown++;
      }
      filterInfo.textContent = shown === segs.length ? `${segs.length} lines` : `${shown} of ${segs.length} lines`;
    }
    applyFilter();
  }

  async function onSpeaker(s, count) {
    const same = count ?? segs.filter((x) => speakerKey(x) === speakerKey(s)).length;
    const r = await identifyDialog(conv.id, s, palette.color(speakerKey(s)), { sameVoiceCount: same });
    if (r) {
      palette.rename(speakerKey(s), `id:${r.speaker_id}`);
      await reload();
    }
  }

  async function toggleMisidentified(s) {
    try {
      const r = await api.setMisidentified(conv.id, s.id, !s.is_misidentified);
      toast(r.message, "ok");
      await reload();
    } catch (e) { toastError(e); }
  }

  function renderAll() {
    renderHead();
    renderPlayer();
    renderSummary();
    renderTranscript();
  }

  const onKey = (e) => {
    if (e.code !== "Space" || e.target.closest("input, textarea, select, button, dialog") || !el.isConnected) return;
    e.preventDefault();
    togglePlay();
  };
  document.addEventListener("keydown", onKey);

  function destroy() {
    cancelAnimationFrame(raf);
    audio.pause();
    audio.removeAttribute("src");
    segAudio?.pause();
    document.removeEventListener("keydown", onKey);
  }

  prepare();
  renderAll();
  loadAudio();
  return { el, reload, destroy, get conversation() { return conv; } };
}

// A conversation page by id (used by the conversations list and after a live session)
export function conversationPage(root, id) {
  root.append(loading("Loading conversation…"));
  let view = null;
  let gone = false;
  api.conversation(id).then((conv) => {
    if (gone) return;
    clear(root);
    view = conversationView(conv, { onDeleted: () => { location.hash = "#/conversations"; } });
    root.append(h("div", { class: "page" }, view.el));
  }).catch((e) => {
    clear(root).append(h("div", { class: "page" }, h("a", { class: "back", href: "#/conversations" }, icon("chevronLeft"), "Conversations"),
      h("div", { class: "card" }, h("div", { class: "empty" }, icon("alert"), h("h3", null, "Conversation not available"), h("p", null, e.message)))));
  });
  return () => { gone = true; view?.destroy(); };
}

