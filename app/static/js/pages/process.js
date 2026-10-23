// Transcribe a file: upload it, wait for the whole pipeline, then show the result in the conversation viewer.

import { api } from "../api.js";
import { h, clear, icon, button, toastError, switchInput, fmtBytes, fmtClock, fmtDuration } from "../ui.js";
import { conversationView } from "../viewer.js";

export function render(root) {
  let file = null;
  let fileDuration = null;
  let diarize = true;
  let view = null;
  let timer = 0;
  let gone = false;

  const input = h("input", { type: "file", accept: "audio/*,video/*", class: "hidden", onchange: (e) => pick(e.target.files[0]) });
  const drop = h("div", { class: "dropzone", tabindex: 0, role: "button", onclick: () => input.click(),
    onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } } },
    h("div", { class: "big-icon" }, icon("upload")),
    h("h2", null, "Drop an audio file here"),
    h("p", { class: "muted" }, "or click to choose one: WAV, MP3, M4A, FLAC, OGG, or a video's sound track"));
  for (const ev of ["dragenter", "dragover"]) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); });
  for (const ev of ["dragleave", "drop"]) drop.addEventListener(ev, () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); pick(e.dataTransfer.files[0]); });

  const fileBox = h("div");
  const progressBox = h("div");
  const resultBox = h("div");
  const diarizeSw = switchInput("Several people", { checked: diarize, onChange: (v) => (diarize = v),
    desc: "Off when one person talks: skips diarization, faster; the whole file is one voice." });
  const startBtn = button("Transcribe", { kind: "primary", icon: "wave", onClick: start, disabled: true });

  function pick(f) {
    if (!f) return;
    file = f;
    fileDuration = null;
    const probe = new Audio(URL.createObjectURL(f));
    probe.preload = "metadata";
    probe.addEventListener("loadedmetadata", () => {
      if (isFinite(probe.duration)) { fileDuration = probe.duration; drawFile(); }
      URL.revokeObjectURL(probe.src);
    });
    probe.addEventListener("error", () => URL.revokeObjectURL(probe.src));
    startBtn.disabled = false;
    drawFile();
  }

  function drawFile() {
    clear(fileBox);
    if (!file) return;
    fileBox.append(h("div", { class: "file-card" },
      h("div", { class: "ic" }, icon("file")),
      h("div", { style: { minWidth: 0, flex: 1 } }, h("div", { class: "name" }, file.name),
        h("div", { class: "small muted" }, `${fmtBytes(file.size)}${fileDuration ? ` · ${fmtClock(fileDuration)}` : ""}`)),
      button("", { kind: "ghost", size: "sm", icon: "x", title: "Remove", onClick: () => { file = null; startBtn.disabled = true; input.value = ""; drawFile(); } })));
  }

  async function start() {
    if (!file) return;
    view?.destroy();
    view = null;
    clear(resultBox);
    startBtn.disabled = true;
    const bar = h("i");
    const label = h("span");
    const elapsed = h("span", { class: "mono small muted" });
    const prog = h("div", { class: "progress" }, bar);
    clear(progressBox).append(h("div", { class: "card card-pad stack" },
      h("div", { class: "row" }, h("span", { class: "spinner" }), label, h("span", { class: "spacer" }), elapsed), prog));
    label.textContent = "Uploading…";
    const t0 = performance.now();
    let tProc = null;
    clearInterval(timer);
    timer = setInterval(() => {
      const s = (performance.now() - (tProc ?? t0)) / 1000;
      elapsed.textContent = `${s.toFixed(1)} s`;
    }, 100);
    try {
      const conv = await api.process(file, diarize, {
        onProgress: (f) => { bar.style.width = `${f * 100}%`; label.textContent = `Uploading… ${Math.round(f * 100)}%`; },
        onUploaded: () => {
          tProc = performance.now();
          prog.classList.add("indeterminate");
          label.textContent = diarize ? "Transcribing, finding speakers and emotions…" : "Transcribing and finding emotions…";
        },
      });
      clearInterval(timer);
      if (gone) return;
      const procSeconds = (performance.now() - (tProc ?? t0)) / 1000;
      const audioSeconds = conv.duration || fileDuration;
      clear(progressBox).append(h("div", { class: "stats" },
        h("div", { class: "stat" }, h("div", { class: "k" }, "Audio"), h("div", { class: "v" }, fmtDuration(audioSeconds))),
        h("div", { class: "stat" }, h("div", { class: "k" }, "Processing"), h("div", { class: "v" }, `${procSeconds.toFixed(1)}`, h("small", null, "s"))),
        h("div", { class: "stat", title: "Seconds of audio per second of processing (upload excluded)" },
          h("div", { class: "k" }, "Speed"), h("div", { class: "v" }, audioSeconds ? `${(audioSeconds / procSeconds).toFixed(1)}` : "–", h("small", null, "× real time"))),
        h("div", { class: "stat" }, h("div", { class: "k" }, "Speakers"), h("div", { class: "v" }, conv.num_speakers ?? "–")),
        h("div", { class: "stat" }, h("div", { class: "k" }, "Lines"), h("div", { class: "v" }, conv.num_segments ?? "–"))));
      view = conversationView(conv, { backLink: false, onDeleted: () => { clear(resultBox); clear(progressBox); } });
      resultBox.append(view.el);
    } catch (e) {
      clearInterval(timer);
      clear(progressBox).append(h("div", { class: "notice" }, icon("alert"), h("div", null, h("b", null, "Processing failed. "), e.message)));
      toastError(e);
    } finally {
      startBtn.disabled = !file;
    }
  }

  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Transcribe a file"),
      h("p", { class: "sub" }, "The whole file goes through the pipeline at once: speech-to-text, who spoke when, who each voice is, and emotions. People you have named are recognised; new voices become Unknowns you can name."))),
    h("div", { class: "card card-pad stack" }, drop, input, fileBox, h("div", { class: "row" }, diarizeSw, h("span", { class: "spacer" }), startBtn)),
    progressBox, resultBox));

  return () => {
    gone = true;
    clearInterval(timer);
    view?.destroy();
  };
}
