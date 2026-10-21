// Speakers: everyone the server knows by voice, the Unknown voices waiting for a name, enrolment from a sample or a
// recording made here, and each person's emotion profile.

import { api } from "../api.js";
import { MicCapture, encodeWav, rms, meterLevel, micAvailable } from "../audio.js";
import {
  h, clear, icon, button, toast, toastError, modal, confirmDialog, promptDialog, loading, emptyState, segmented,
  fmtRelative, fmtClock, fmtDuration, pct, count, isUnknown, isDeleted, displayName, initials, idColor, speakerChip,
  emotionChip,
} from "../ui.js";
import { identifyDialog } from "../viewer.js";

// How alike two stored voices must be to suggest they are one person: a voice heard over a few lines scores about
// 0.6-0.8 against its owner's profile, other people stay under 0.5
const SUGGEST = 0.5;
const LIKELY = 0.6;

export function render(root) {
  let speakers = [];
  let matches = new Map();   // Unknown's id -> the closest person and the closest other Unknown
  let tab = "people";
  let query = "";
  let playing = null;
  const grid = h("div", { class: "speaker-grid" });
  const tabsEl = h("div", { class: "tabs" });
  const body = h("div", null, loading());

  async function load() {
    try {
      const [list, close] = await Promise.all([api.speakers(), api.unknownMatches().catch(() => [])]);
      speakers = list;
      matches = new Map(close.map((m) => [m.speaker_id, m]));
      draw();
    } catch (e) {
      clear(body).append(h("div", { class: "card" }, emptyState("alert", "Cannot load speakers", e.message)));
    }
  }

  function draw() {
    const people = speakers.filter((s) => !isUnknown(s.name));
    const unknown = speakers.filter((s) => isUnknown(s.name));
    clear(tabsEl);
    for (const [key, label, n] of [["people", "People", people.length], ["unknown", "Unknown voices", unknown.length], ["all", "All", speakers.length]]) {
      tabsEl.append(h("button", { type: "button", class: tab === key ? "on" : "", onclick: () => { tab = key; draw(); } },
        label, h("span", { class: "count" }, n)));
    }
    const q = query.trim().toLowerCase();
    const list = (tab === "people" ? people : tab === "unknown" ? unknown : speakers)
      .filter((s) => !q || s.name.toLowerCase().includes(q))
      .sort((a, b) => (isUnknown(a.name) !== isUnknown(b.name) ? (isUnknown(a.name) ? 1 : -1)
        : isUnknown(a.name) ? String(b.last_heard || "").localeCompare(String(a.last_heard || "")) : a.name.localeCompare(b.name)));
    clear(grid);
    for (const s of list) grid.append(card(s));
    clear(body);
    if (!list.length) {
      body.append(h("div", { class: "card" }, tab === "unknown"
        ? emptyState("users", "No Unknown voices", "Every voice nobody has named yet is kept here, and recognised as the same Unknown when it speaks again. Name one here or from any of its lines.")
        : emptyState("users", q ? "Nobody matches" : "Nobody enrolled yet", q ? null : "Enrol someone from a voice sample, or name an Unknown voice from a conversation.",
          q ? null : button("Enrol a speaker", { kind: "primary", icon: "userPlus", onClick: enroll }))));
    } else body.append(grid);
  }

  function card(s) {
    const unknown = isUnknown(s.name);
    const color = idColor(s.id);
    const close = unknown ? matches.get(s.id) : null;
    const suggestion = (m, label) => (m && m.similarity >= SUGGEST ? h("div", { class: `suggest${m.similarity >= LIKELY ? " likely" : ""}` },
      h("span", { class: "small muted" }, label), speakerChip(m.speaker_name, idColor(m.speaker_id), { id: m.speaker_id }),
      h("span", { class: "small num", title: "How alike the two stored voices are" }, pct(m.similarity)),
      h("span", { class: "spacer" }),
      button("Merge", { size: "sm", kind: m.similarity >= LIKELY ? "" : "ghost", icon: "identify", onClick: () => merge(s, m) })) : null);
    return h("div", { class: "card speaker-card" },
      h("div", { class: "top" },
        h("div", { class: `avatar${unknown ? " unknown" : ""}`, style: { "--c": isDeleted(s.name) ? "var(--muted)" : color } }, initials(s.name)),
        h("div", { style: { minWidth: 0 } }, h("div", { class: "name", title: s.name }, displayName(s.name, s.id)),
          h("div", { class: "small muted" }, s.last_heard ? `Last heard ${fmtRelative(s.last_heard)}` : unknown ? "Not named yet" : `Enrolled ${fmtRelative(s.created_at)}`))),
      h("div", { class: "kv" },
        h("span", { title: "Speech in all conversations" }, h("b", null, fmtDuration(s.speech_seconds || 0))),
        h("span", null, h("b", null, s.segment_count), s.segment_count === 1 ? " line" : " lines"),
        h("span", { class: "mono" }, `#${s.id}`)),
      close ? suggestion(close.person, "Sounds like") : null,
      close ? suggestion(close.unknown, "Same voice as") : null,
      h("div", { class: "foot" },
        button(unknown ? "Name" : "Rename", { size: "sm", icon: unknown ? "identify" : "edit", onClick: () => (unknown ? nameVoice(s) : rename(s)) }),
        s.segment_count ? button("", { size: "sm", kind: "ghost", icon: "play", title: "Listen to their longest line", onClick: (e) => listen(s, e.currentTarget) }) : null,
        !unknown ? button("Emotions", { size: "sm", kind: "ghost", icon: "smile", onClick: () => emotionProfiles(s) }) : null,
        h("span", { class: "spacer" }),
        button("", { size: "sm", kind: "ghost", icon: "trash", title: "Delete", onClick: () => remove(s) })));
  }

  const longestLine = async (s) => (await api.speakerSegments(s.id, { limit: 1, order: "longest" }))[0];

  async function listen(s, btn) {
    if (playing) {
      playing.pause();
      const same = playing.speakerId === s.id;
      playing = null;
      if (same) return;
    }
    try {
      const seg = await longestLine(s);
      if (!seg) return toast("No lines to play", "info");
      playing = new Audio(api.segmentAudioUrl(seg.id));
      playing.speakerId = s.id;
      await playing.play();
      toast(`“${seg.text || "…"}” (${seg.conversation_title || `conversation ${seg.conversation_id}`})`, "info", 3500);
    } catch (e) { toastError(e); }
  }

  // Naming an Unknown goes through one of its lines (identify): every line of it, in every conversation, moves to the
  // person, new or already enrolled, whose voice is then rebuilt from all of them
  async function nameVoice(s) {
    let seg;
    try { seg = await longestLine(s); } catch (e) { return toastError(e); }
    if (!seg) return rename(s);
    const r = await identifyDialog(seg.conversation_id, { ...seg, speaker_id: s.id, speaker_name: s.name }, idColor(s.id));
    if (r) await load();
  }

  async function merge(s, m) {
    const target = displayName(m.speaker_name, m.speaker_id);
    if (!(await confirmDialog({
      title: `Merge ${displayName(s.name, s.id)} into ${target}?`, confirmLabel: "Merge", icon: "identify",
      message: h("div", { class: "stack" },
        h("p", null, `${s.segment_count === 1 ? "Its line" : `All ${s.segment_count} of its lines`}, in every conversation, become ${target}'s, and ${target}'s voice profile learns from them.`),
        h("p", { class: "muted" }, "Listen to it first if you are not sure: lines given to the wrong person make their voice profile worse.")),
    }))) return;
    try {
      const seg = await longestLine(s);
      if (!seg) return toast("It has no lines to merge", "err");
      const r = await api.identify(seg.conversation_id, seg.id, { speaker_id: m.speaker_id });
      toast(r.message, "ok", 6000);
      await load();
    } catch (e) { toastError(e); }
  }

  async function rename(s) {
    const unknown = isUnknown(s.name);
    const name = await promptDialog({
      title: unknown ? `Name ${displayName(s.name, s.id)}` : `Rename ${s.name}`, label: "Name", value: unknown ? "" : s.name, confirmLabel: unknown ? "Name" : "Rename",
      hint: unknown ? `All ${s.segment_count} of their lines take this name, and the voice is recognised as them from now on. To merge them into someone already enrolled, use “Who is speaking?” on one of their lines.`
        : "Their past lines are renamed too.",
    });
    if (!name || name === s.name) return;
    if (isUnknown(name)) return toast("Names starting with Unknown_ are kept for unnamed voices", "err");
    try {
      await api.renameSpeaker(s.id, name);
      toast(`${displayName(s.name, s.id)} is now ${name}`, "ok");
      await load();
    } catch (e) { toastError(e); }
  }

  async function remove(s) {
    if (!(await confirmDialog({
      title: `Delete ${displayName(s.name, s.id)}?`, danger: true, confirmLabel: "Delete", icon: "trash",
      message: h("div", { class: "stack" }, h("p", null, "Their voice profile and emotion profiles are removed. They will not be recognised any more."),
        h("p", { class: "muted" }, `Their ${count(s.segment_count, "line")} stay in the transcripts, labelled Deleted_${s.name}.`)),
    }))) return;
    try {
      const r = await api.deleteSpeaker(s.id);
      toast(r.message, "ok");
      await load();
    } catch (e) { toastError(e); }
  }

  // Every new voice is kept as an Unknown, so stray ones (a cough, a word from the TV) pile up: delete those heard
  // for only a few seconds, or all of them
  function cleanUp() {
    const unknowns = speakers.filter((s) => isUnknown(s.name));
    if (!unknowns.length) return toast("There are no Unknown voices", "info");
    let seconds = 5;
    let all = false;
    const summary = h("p");
    const secondsInput = h("input", { class: "input sm", type: "number", min: 0, step: 1, value: "5", style: { width: "80px" } });
    const draw = () => {
      const n = all ? unknowns.length : unknowns.filter((s) => (s.speech_seconds || 0) <= seconds).length;
      summary.textContent = `${n} of ${count(unknowns.length, "Unknown voice")} will be deleted.`;
      secondsInput.disabled = all;
    };
    secondsInput.addEventListener("input", () => { seconds = Math.max(0, Number(secondsInput.value) || 0); draw(); });
    const allSw = h("label", { class: "switch" }, h("input", { type: "checkbox", onchange: (e) => { all = e.target.checked; draw(); } }),
      h("span", null, "All of them", h("span", { class: "desc" }, "Every voice nobody has named.")));
    draw();
    modal({
      title: "Clean up Unknown voices",
      body: h("div", { class: "stack" },
        h("div", { class: "row" }, h("span", null, "Delete Unknown voices heard for at most"), secondsInput, h("span", null, "seconds")),
        allSw, summary,
        h("p", { class: "small muted" }, "Their lines stay in the transcripts, without a speaker. A deleted voice that speaks again starts as a new Unknown.")),
      actions: [{
        label: "Delete", kind: "danger solid", icon: "trash",
        onClick: async () => {
          const r = await api.deleteUnknownSpeakers(all ? undefined : seconds);
          toast(r.message, "ok");
          await load();
        },
      }],
    });
  }

  async function enroll() {
    let mode = micAvailable() ? "record" : "upload";
    let file = null;
    let recorded = null;
    const name = h("input", { class: "input", placeholder: "Their name", maxlength: 80 });
    const fileInput = h("input", { type: "file", accept: "audio/*", class: "input", style: { paddingTop: "6px" }, onchange: (e) => { file = e.target.files[0] || null; } });
    const recBox = h("div", { class: "stack" });
    const upBox = h("div", { class: "field" }, h("label", null, "Voice sample"), fileInput,
      h("div", { class: "hint" }, "10-30 seconds of this person alone, without music or other voices."));
    let mic = null, chunks = [], t0 = 0, timer = 0, previewUrl = null;
    const meter = h("i");
    const clock = h("span", { class: "mono" }, "0:00");
    const recBtn = h("button", { class: "rec-btn", type: "button", title: "Record", style: { width: "56px", height: "56px" } });
    const preview = h("div");
    const stopRec = () => {
      clearInterval(timer);
      mic?.stop();
      if (mic && chunks.length) {
        recorded = encodeWav(chunks, mic.sampleRate);
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        previewUrl = URL.createObjectURL(recorded);
        clear(preview).append(h("audio", { controls: true, src: previewUrl, style: { width: "100%" } }));
      }
      mic = null;
      recBtn.classList.remove("on");
      meter.style.width = "0";
    };
    recBtn.addEventListener("click", async () => {
      if (mic) return stopRec();
      chunks = [];
      recorded = null;
      clear(preview);
      try {
        mic = await new MicCapture({ onFrame: (f) => { chunks.push(f); meter.style.width = `${meterLevel(rms(f)) * 100}%`; } }).start();
      } catch (e) { mic = null; return toastError(e); }
      t0 = performance.now();
      recBtn.classList.add("on");
      timer = setInterval(() => {
        const s = (performance.now() - t0) / 1000;
        clock.textContent = fmtClock(s);
        if (s >= 60) stopRec();
      }, 200);
    });
    recBox.append(h("div", { class: "row", style: { flexWrap: "nowrap", gap: "16px" } }, recBtn,
      h("div", { class: "stack", style: { flex: 1, gap: "8px" } }, h("div", { class: "row" }, h("b", null, "Record a sample"), h("span", { class: "spacer" }), clock),
        h("div", { class: "meter" }, meter))),
      h("div", { class: "hint small muted" }, "Have them talk normally for 10-30 seconds, alone; read something aloud if nothing comes to mind. Stops by itself at one minute."),
      preview);
    const draw = () => { recBox.classList.toggle("hidden", mode !== "record"); upBox.classList.toggle("hidden", mode !== "upload"); };
    draw();
    modal({
      title: "Enrol a speaker",
      body: h("div", { class: "stack" },
        h("div", { class: "field" }, h("label", null, "Name"), name),
        segmented([{ value: "record", label: "Record here", icon: "mic" }, { value: "upload", label: "Upload a sample", icon: "upload" }], mode, (v) => { stopRec(); mode = v; draw(); }),
        recBox, upBox,
        micAvailable() ? null : h("div", { class: "notice" }, icon("alert"), h("div", null, "Recording needs the page on HTTPS; uploading a sample works anywhere."))),
      actions: [{
        label: "Enrol", kind: "primary", icon: "userPlus",
        onClick: async () => {
          stopRec();
          const n = name.value.trim();
          if (!n) { toast("Give them a name", "err"); return false; }
          const audio = mode === "record" ? recorded : file;
          if (!audio) { toast(mode === "record" ? "Record a sample first" : "Choose a sample file", "err"); return false; }
          await api.enroll(n, audio, mode === "record" ? "recording.wav" : file.name);
          toast(`${n} enrolled`, "ok");
          await load();
        },
      }],
      onClose: () => { stopRec(); if (previewUrl) URL.revokeObjectURL(previewUrl); },
    });
    setTimeout(() => name.focus(), 30);
  }

  async function emotionProfiles(s) {
    let data, globalThreshold = 0.6;
    try {
      [data, { emotion_threshold: globalThreshold }] = await Promise.all([api.emotionProfiles(s.id), api.settings()]);
    } catch (e) { return toastError(e); }
    const bodyEl = h("div", { class: "stack" });
    const draw = () => {
      clear(bodyEl);
      const custom = data.emotion_threshold != null;
      const value = h("span", { class: "mono" }, (data.emotion_threshold ?? globalThreshold).toFixed(2));
      const slider = h("input", { type: "range", min: 0.3, max: 1, step: 0.01, value: String(data.emotion_threshold ?? globalThreshold), disabled: !custom,
        oninput: (e) => { value.textContent = Number(e.target.value).toFixed(2); },
        onchange: async (e) => {
          try { await api.setEmotionThreshold(s.id, Number(e.target.value)); data.emotion_threshold = Number(e.target.value); toast("Threshold saved", "ok", 1500); } catch (err) { toastError(err); }
        } });
      const useGlobal = h("label", { class: "switch" }, h("input", { type: "checkbox", checked: !custom, onchange: async (e) => {
        try {
          const t = e.target.checked ? null : globalThreshold;
          await api.setEmotionThreshold(s.id, t);
          data.emotion_threshold = t;
          draw();
        } catch (err) { toastError(err); }
      } }), h("span", null, `Use the global threshold (${globalThreshold.toFixed(2)})`,
        h("span", { class: "desc" }, "How closely a line must match one of their emotion profiles to take it over the generic detector.")));
      bodyEl.append(h("div", { class: "field" }, useGlobal, h("div", { class: "row", style: { flexWrap: "nowrap" } }, slider, value)));
      bodyEl.append(h("div", { class: "divider" }));
      if (!data.profiles.length) {
        bodyEl.append(h("div", { class: "empty", style: { padding: "20px" } }, icon("smile"), h("h3", null, "No emotion profile yet"),
          h("p", null, `Correct the emotion of some of ${s.name}'s lines (with “learn” on) and their personal profile builds up here.`)));
        return;
      }
      const numInput = (val, min, max, save) => h("input", { class: "input sm", type: "number", step: 0.01, min, max, placeholder: "default",
        value: val == null ? "" : String(val), style: { width: "90px" },
        onchange: async (e) => {
          const v = e.target.value === "" ? null : Number(e.target.value);
          try { await save(v); toast("Saved", "ok", 1200); } catch (err) { toastError(err); }
        } });
      bodyEl.append(h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", null, h("tr", null, ["Emotion", "Learned from", "Match threshold", "Voice threshold", ""].map((t) => h("th", null, t)))),
        h("tbody", null, data.profiles.map((p) => h("tr", null,
          h("td", null, emotionChip(p.emotion_category, null) || p.emotion_category),
          h("td", { class: "num" }, count(p.sample_count, "line"), p.voice_sample_count != null ? h("div", { class: "small muted" }, `${p.voice_sample_count} voice samples`) : null),
          h("td", null, numInput(p.confidence_threshold, 0.3, 1, (v) => api.setProfileThreshold(s.id, p.emotion_category, v))),
          h("td", null, numInput(p.voice_threshold, 0, 1, (v) => api.setProfileVoiceThreshold(s.id, p.emotion_category, v))),
          h("td", { style: { textAlign: "right" } }, button("", { size: "sm", kind: "ghost", icon: "trash", title: "Forget this emotion",
            onClick: async () => {
              try { await api.resetEmotionProfiles(s.id, p.emotion_category); data = await api.emotionProfiles(s.id); draw(); } catch (e) { toastError(e); }
            } }))))))));
    };
    draw();
    modal({
      title: `${s.name}: emotion profile`, wide: true, cancelLabel: "Close", body: bodyEl,
      actions: [{
        label: "Forget all", kind: "danger", icon: "trash",
        onClick: async () => {
          if (!data.profiles.length) return false;
          await api.resetEmotionProfiles(s.id);
          data = await api.emotionProfiles(s.id);
          draw();
          toast("Emotion profile cleared", "ok");
          return false;
        },
      }],
    });
  }

  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Speakers"),
        h("p", { class: "sub" }, "Everyone recognised by voice. A voice nobody has named is kept as one Unknown and recognised in later conversations: name it once and it is known by name from then on.")),
      h("div", { class: "actions" },
        h("div", { class: "search", style: { width: "220px" } }, icon("search"),
          h("input", { class: "input", placeholder: "Search…", oninput: (e) => { query = e.target.value; draw(); } })),
        button("Clean up Unknowns", { icon: "trash", onClick: cleanUp }),
        button("Enrol a speaker", { kind: "primary", icon: "userPlus", onClick: enroll }))),
    tabsEl, body));
  load();
}

