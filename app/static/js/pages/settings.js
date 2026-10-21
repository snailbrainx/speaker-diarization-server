// Settings the server lets you change while it runs (/settings/voice), and what it is running on.

import { api, streamUrl } from "../api.js";
import { h, clear, icon, button, toast, toastError, confirmDialog, loading } from "../ui.js";
import { micAvailable } from "../audio.js";

const FIELDS = [
  { key: "speaker_threshold", label: "Speaker match threshold", min: 0.1, max: 0.9, step: 0.01,
    desc: "How close a single line's voice must be to a stored speaker to take their name. Only used when lines are matched one by one (SPEAKER_ASSIGNMENT=segment, or live tracking off); the default whole-speaker matching has its own threshold (CLUSTER_SPEAKER_THRESHOLD)." },
  { key: "emotion_threshold", label: "Emotion profile threshold", min: 0.3, max: 1, step: 0.01,
    desc: "How closely a line must match someone's personal emotion profile to use it instead of the generic detector. Higher means fewer, surer personal matches." },
  { key: "silence_duration", label: "Live pause length", min: 0.1, max: 5, step: 0.05, unit: "s",
    desc: "How long a pause in a live stream must be before the speech so far is processed. Shorter gives lines sooner, longer gives each chunk more context." },
  { key: "context_padding", label: "Voice context padding", min: 0.05, max: 2, step: 0.05, unit: "s",
    desc: "Audio added before and after a line when its voice is measured." },
  { key: "filter_hallucinations", label: "Filter hallucinations", bool: true,
    desc: "Drops lines speech-to-text tends to invent in silence or music, such as “Thank you.” or “Thanks for watching”." },
];

export function render(root) {
  const form = h("div", { class: "card" }, loading());
  const serverBox = h("div", { class: "card-body" }, loading());
  let saved = null;
  const values = {};

  async function load() {
    try {
      saved = await api.settings();
      Object.assign(values, saved);
      draw();
    } catch (e) {
      clear(form).append(h("div", { class: "card-body muted" }, e.message));
    }
  }

  function draw() {
    clear(form);
    for (const f of FIELDS) {
      let ctl;
      if (f.bool) {
        ctl = h("label", { class: "switch" }, h("input", { type: "checkbox", checked: !!values[f.key], onchange: (e) => { values[f.key] = e.target.checked; dirty(); } }),
          h("span", null, values[f.key] ? "On" : "Off"));
        ctl.querySelector("input").addEventListener("change", (e) => { ctl.lastChild.textContent = e.target.checked ? "On" : "Off"; });
      } else {
        const num = h("input", { class: "input sm", type: "number", min: f.min, max: f.max, step: f.step, value: String(values[f.key]) });
        const range = h("input", { type: "range", min: f.min, max: f.max, step: f.step, value: String(values[f.key]) });
        range.addEventListener("input", () => { num.value = range.value; values[f.key] = Number(range.value); dirty(); });
        num.addEventListener("change", () => {
          const v = Math.min(f.max, Math.max(f.min, Number(num.value)));
          num.value = range.value = String(v);
          values[f.key] = v;
          dirty();
        });
        ctl = h("div", { class: "ctl" }, range, num, f.unit ? h("span", { class: "small muted" }, f.unit) : null);
      }
      form.append(h("div", { class: "setting" },
        h("div", null, h("h3", null, f.label), h("p", { class: "desc" }, f.desc), h("code", { class: "small muted" }, f.key)), ctl));
    }
    form.append(h("div", { class: "modal-foot", style: { borderRadius: "0 0 var(--radius) var(--radius)" } },
      button("Reset to defaults", { kind: "ghost", icon: "refresh", onClick: reset }),
      h("span", { class: "spacer" }),
      (saveBtn = button("Save", { kind: "primary", icon: "check", onClick: save, disabled: true }))));
  }

  let saveBtn;
  const changed = () => FIELDS.filter((f) => values[f.key] !== saved[f.key]);
  const dirty = () => { if (saveBtn) saveBtn.disabled = !changed().length; };

  async function save() {
    const update = Object.fromEntries(changed().map((f) => [f.key, values[f.key]]));
    try {
      saved = await api.updateSettings(update);
      Object.assign(values, saved);
      toast("Settings saved", "ok");
      draw();
    } catch (e) { toastError(e); }
  }

  async function reset() {
    if (!(await confirmDialog({ title: "Reset settings to defaults?", confirmLabel: "Reset", message: "All the settings on this page go back to their defaults." }))) return;
    try {
      saved = await api.resetSettings();
      Object.assign(values, saved);
      toast("Settings reset", "ok");
      draw();
    } catch (e) { toastError(e); }
  }

  async function loadServer() {
    const kv = (k, v) => [h("dt", null, k), h("dd", null, v)];
    try {
      const s = await api.status();
      clear(serverBox).append(h("dl", { class: "kvlist" },
        kv("Status", h("span", { class: `badge ${s.status === "online" ? "ok" : "warn"}` }, s.status)),
        kv("Compute", s.gpu_available ? `GPU (${s.device})` : `CPU (${s.device})`),
        kv("Live stream", h("code", null, streamUrl())),
        kv("Microphone here", micAvailable() ? "available (secure page)" : "unavailable: open the page over HTTPS"),
        kv("API", h("a", { href: "/docs", target: "_blank", rel: "noopener" }, "/docs ", icon("external"))),
        kv("MCP", h("code", null, `${location.origin}/mcp`))));
    } catch (e) {
      clear(serverBox).append(h("p", { class: "muted" }, e.message));
    }
  }

  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Settings"),
      h("p", { class: "sub" }, "Changes apply at once and are kept across restarts. The pipeline itself (speech-to-text engine, diarization, voice model, live tracking, overlap separation) is set with environment variables when the server starts."))),
    h("div", { class: "grid", style: { gridTemplateColumns: "minmax(0, 1fr)" } }, form,
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, icon("server"), "Server")), serverBox))));
  load();
  loadServer();
}
