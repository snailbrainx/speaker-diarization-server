// Speakers: everyone the server knows by voice, the Unknown voices waiting for a name, enrolment from a sample or a
// recording made here, and each person's emotion profile.

import { api } from "../api.js";
import { MicCapture, encodeWav, rms, meterLevel, micAvailable } from "../audio.js";
import {
  h, clear, icon, button, toast, toastError, modal, confirmDialog, promptDialog, loading, emptyState, segmented,
  fmtRelative, fmtDate, fmtClock, isUnknown, isDeleted, displayName, initials, idColor, emotionChip,
} from "../ui.js";

export function render(root) {
  let speakers = [];
  let tab = "people";
  let query = "";
  const grid = h("div", { class: "speaker-grid" });
  const tabsEl = h("div", { class: "tabs" });
  const body = h("div", null, loading());

  async function load() {
    try {
      speakers = await api.speakers();
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
      .sort((a, b) => (tab === "unknown" ? b.segment_count - a.segment_count : a.name.localeCompare(b.name)));
    clear(grid);
    for (const s of list) grid.append(card(s));
    clear(body);
    if (!list.length) {
      body.append(h("div", { class: "card" }, tab === "unknown"
        ? emptyState("users", "No Unknown voices", "Voices nobody has named yet collect here. Name one from any of its lines, or here.")
        : emptyState("users", q ? "Nobody matches" : "Nobody enrolled yet", q ? null : "Enrol someone from a voice sample, or name an Unknown voice from a conversation.",
          q ? null : button("Enrol a speaker", { kind: "primary", icon: "userPlus", onClick: enroll }))));
    } else body.append(grid);
  }

  function card(s) {
    const unknown = isUnknown(s.name);
    const color = idColor(s.id);
    return h("div", { class: "card speaker-card" },
      h("div", { class: "top" },
        h("div", { class: `avatar${unknown ? " unknown" : ""}`, style: { "--c": isDeleted(s.name) ? "var(--muted)" : color } }, initials(s.name)),
        h("div", { style: { minWidth: 0 } }, h("div", { class: "name", title: s.name }, displayName(s.name, s.id)),
          h("div", { class: "small muted" }, unknown ? "Not named yet" : `Enrolled ${fmtRelative(s.created_at)}`))),
      h("div", { class: "kv" },
        h("span", null, h("b", null, s.segment_count), " lines"),
        h("span", { title: fmtDate(s.updated_at) }, "updated ", h("b", null, fmtRelative(s.updated_at))),
        h("span", { class: "mono" }, `#${s.id}`)),
      h("div", { class: "foot" },
        button(unknown ? "Name" : "Rename", { size: "sm", icon: unknown ? "identify" : "edit", onClick: () => rename(s) }),
        !unknown ? button("Emotions", { size: "sm", kind: "ghost", icon: "smile", onClick: () => emotionProfiles(s) }) : null,
        h("span", { class: "spacer" }),
        button("", { size: "sm", kind: "ghost", icon: "trash", title: "Delete", onClick: () => remove(s) })));
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
        h("p", { class: "muted" }, `Their ${s.segment_count} lines stay in the transcripts, labelled Deleted_${s.name}.`)),
    }))) return;
    try {
      const r = await api.deleteSpeaker(s.id);
      toast(r.message, "ok");
      await load();
    } catch (e) { toastError(e); }
  }

  async function removeUnknowns() {
    const n = speakers.filter((s) => isUnknown(s.name)).length;
    if (!n) return toast("There are no Unknown voices", "info");
    if (!(await confirmDialog({
      title: `Delete all ${n} Unknown voices?`, danger: true, confirmLabel: "Delete them", icon: "trash",
      message: "Voices nobody has named are forgotten; if they speak again they start as new Unknowns.",
    }))) return;
    try {
      const r = await api.deleteUnknownSpeakers();
      toast(r.message, "ok");
      await load();
    } catch (e) { toastError(e); }
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
          h("td", { class: "num" }, `${p.sample_count} lines`, p.voice_sample_count != null ? h("div", { class: "small muted" }, `${p.voice_sample_count} voice samples`) : null),
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
        h("p", { class: "sub" }, "Everyone recognised by voice. A voice nobody has named stays one Unknown across conversations: name it once and it is known from then on.")),
      h("div", { class: "actions" },
        h("div", { class: "search", style: { width: "220px" } }, icon("search"),
          h("input", { class: "input", placeholder: "Search…", oninput: (e) => { query = e.target.value; draw(); } })),
        button("Delete Unknowns", { kind: "danger", icon: "trash", onClick: removeUnknowns }),
        button("Enrol a speaker", { kind: "primary", icon: "userPlus", onClick: enroll }))),
    tabsEl, body));
  load();
}

