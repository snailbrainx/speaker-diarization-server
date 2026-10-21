// Live: stream the microphone, or a file at real-time pace, and watch lines arrive with their speakers. Voices start
// as Unknown and are renamed as the session learns them (speaker_update); naming an Unknown here makes it known.

import { live } from "../live-session.js";
import { listMicrophones, micAvailable } from "../audio.js";
import {
  h, clear, icon, button, toast, segmented, switchInput, Palette, speakerChip, fmtClock, fmtDuration, fmtBytes, isUnknown,
  isUnnamedVoice,
} from "../ui.js";
import { segmentRow, identifyDialog, detailsDialog, speakerKey, speakerStats } from "../viewer.js";
import { api } from "../api.js";

// Options are remembered for this page only (not across reloads)
const opts = { source: micAvailable() ? "mic" : "file", deviceId: "", processing: false, file: null, speed: 1, hear: true, diarize: true };

export function render(root) {
  const palette = new Palette();
  const rows = new Map();
  let autoScroll = true;

  // ----- recorder -----
  const recBtn = h("button", { class: "rec-btn", title: "Start / stop", onclick: onRecord });
  const clock = h("span", { class: "clock" }, "0:00.0");
  const what = h("span", { class: "what" });
  const meterFill = h("i");
  const vad = h("span", { class: "vad", title: "The server hears speech" }, h("i"), "speech");
  const sourceEl = h("span");
  const recorder = h("div", { class: "card recorder" }, recBtn,
    h("div", { class: "rec-info" }, h("div", { class: "rec-state" }, clock, what), h("div", { class: "meter" }, meterFill),
      h("div", { class: "row small muted" }, vad, h("span", { class: "spacer" }), sourceEl)));

  const statEls = {};
  const stat = (k, label) => h("div", { class: "stat" }, h("div", { class: "k" }, label), (statEls[k] = h("div", { class: "v" }, "–")));
  const stats = h("div", { class: "stats" },
    stat("audio", "Audio sent"), stat("lines", "Lines"), stat("speakers", "Speakers"),
    stat("lag", "Last delay"), stat("avg", "Average delay"));

  // ----- options -----
  const micSelect = h("select", { class: "select", onchange: (e) => { opts.deviceId = e.target.value; } },
    h("option", { value: "" }, "Default microphone"));
  const fillMics = async () => {
    try {
      const mics = await listMicrophones();
      clear(micSelect).append(h("option", { value: "" }, "Default microphone"),
        mics.filter((m) => m.deviceId && m.deviceId !== "default").map((m, i) =>
          h("option", { value: m.deviceId, selected: m.deviceId === opts.deviceId }, m.label || `Microphone ${i + 1}`)));
    } catch { /* keep the default */ }
  };
  fillMics();
  const fileInput = h("input", { type: "file", accept: "audio/*,video/*", class: "hidden",
    onchange: (e) => { opts.file = e.target.files[0] || null; drawFile(); } });
  const fileLabel = h("span", { class: "small muted" });
  const drawFile = () => { fileLabel.textContent = opts.file ? `${opts.file.name} · ${fmtBytes(opts.file.size)}` : "No file chosen"; };
  drawFile();
  const micOpts = h("div", { class: "stack" },
    h("div", { class: "field" }, h("label", null, "Microphone"), micSelect),
    switchInput("Browser voice processing", { checked: opts.processing, onChange: (v) => (opts.processing = v),
      desc: "Echo cancelling, noise suppression and automatic gain. Off sends the raw microphone, which voices are matched best on." }));
  const fileOpts = h("div", { class: "stack" },
    h("div", { class: "field" }, h("label", null, "Audio file"),
      h("div", { class: "row" }, button("Choose…", { icon: "file", size: "sm", onClick: () => fileInput.click() }), fileLabel), fileInput),
    h("div", { class: "field" }, h("label", null, "Pace"),
      segmented([{ value: "1", label: "Real time" }, { value: "2", label: "2×" }, { value: "4", label: "4×" }], String(opts.speed), (v) => (opts.speed = Number(v)))),
    switchInput("Play it here too", { checked: opts.hear, onChange: (v) => (opts.hear = v), desc: "Hear the file while it streams." }));
  const sourceSeg = segmented([{ value: "mic", label: "Microphone", icon: "mic" }, { value: "file", label: "Audio file", icon: "file" }],
    opts.source, (v) => { opts.source = v; drawSource(); });
  const drawSource = () => { micOpts.classList.toggle("hidden", opts.source !== "mic"); fileOpts.classList.toggle("hidden", opts.source !== "file"); };
  drawSource();
  const diarizeSw = switchInput("Several people", { checked: opts.diarize, onChange: (v) => (opts.diarize = v),
    desc: "Off when one person talks: no diarization per chunk, everything is one voice." });
  const optionsCard = h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, icon("sliders"), "Source")),
    h("div", { class: "card-body options" }, sourceSeg, micOpts, fileOpts, h("div", { class: "divider" }), diarizeSw));

  // ----- transcript, speakers, log -----
  const feed = h("div", { class: "transcript", style: { maxHeight: "62vh", overflow: "auto" } });
  feed.addEventListener("scroll", () => { autoScroll = feed.scrollTop + feed.clientHeight >= feed.scrollHeight - 40; });
  const feedEmpty = h("div", { class: "empty" }, icon("radio"), h("h3", null, "Nothing yet"),
    h("p", null, "Lines appear here a moment after each pause. New voices start as Unknown and are named once they match someone."));
  const feedCount = h("span", { class: "small muted" });
  const speakersBox = h("div", { class: "speaker-list" });
  const doneBox = h("div");
  const logBox = h("div", { class: "log" });
  const logCount = h("span", { class: "small muted" });

  const ctx = {
    palette,
    onPlay: (seg) => {
      const a = new Audio(api.segmentAudioUrl(seg.id));
      a.play().catch(() => toast("This line's audio is not available", "err"));
    },
    onSpeaker: (seg) => nameVoice(seg),
    onMenu: (seg) => detailsDialog(seg),
  };

  async function nameVoice(seg) {
    const same = live.segments.filter((s) => speakerKey(s) === speakerKey(seg)).length;
    const r = await identifyDialog(live.conversationId, seg, palette.color(speakerKey(seg)), { sameVoiceCount: same });
    if (!r) return;
    palette.rename(speakerKey(seg), `id:${r.speaker_id}`);
    try { await live.sync(); } catch (e) { toast(e.message, "err"); }
  }

  const addRow = (seg, isNew) => {
    const row = segmentRow(seg, { ...ctx, isNew });
    const lag = h("span", { class: "conf", title: "Delay: how long after the speech ended this line arrived" }, `+${seg.lag.toFixed(1)} s`);
    row.querySelector(".seg-head").append(lag);
    rows.set(seg.id, row);
    return row;
  };

  function drawFeed() {
    clear(feed);
    rows.clear();
    if (!live.segments.length) feed.append(feedEmpty);
    live.segments.forEach((s) => feed.append(addRow(s, false)));
    feedCount.textContent = `${live.segments.length} lines`;
  }

  function drawSpeakers() {
    clear(speakersBox);
    const st = speakerStats(live.segments).sort((a, b) => b.talk - a.talk);
    if (!st.length) speakersBox.append(h("div", { class: "small muted" }, "Voices heard in this session show up here."));
    for (const s of st) {
      const unknown = isUnknown(s.name);
      const longest = s.segs.reduce((a, x) => (x.end_offset - x.start_offset > a.end_offset - a.start_offset ? x : a), s.segs[0]);
      speakersBox.append(h("div", { class: "speaker-row" },
        speakerChip(s.name, palette.color(s.key), { id: s.id }),
        isUnnamedVoice(s.name) && live.conversationId ? button("Name", { size: "sm", kind: "ghost", icon: "identify", onClick: () => nameVoice(longest) })
          : unknown ? h("span", { class: "badge", title: "Too little speech so far to tell whose voice this is; these lines join their voice once it has a speaker" }, "not placed yet")
          : h("span", { class: "badge ok" }, "known"),
        h("span", { class: "meta" }, `${s.segs.length} · ${fmtDuration(s.talk)}`)));
    }
    statEls.speakers.textContent = st.length || "–";
  }

  const logLine = (e) => h("div", null, h("span", { class: "ts" }, e.t.toLocaleTimeString()), h("span", { class: `ty ${e.type}` }, e.type), h("span", null, e.text));
  function drawLog() {
    clear(logBox).append(live.events.map(logLine));
    logBox.scrollTop = logBox.scrollHeight;
    logCount.textContent = `${live.events.length} events · ${live.statusMessages} status`;
  }

  function drawStats() {
    const s = live.seconds;
    clock.textContent = fmtClock(s, 1);
    statEls.audio.textContent = live.state === "idle" ? "–" : fmtClock(s, 0);
    if (live.fileDuration) statEls.audio.append(h("small", null, ` / ${fmtClock(live.fileDuration)}`));
    statEls.lines.textContent = live.segments.length || "–";
    const lags = live.lags;
    statEls.lag.textContent = lags.length ? `${lags[lags.length - 1].toFixed(1)}` : "–";
    if (lags.length) statEls.lag.append(h("small", null, "s"));
    statEls.avg.textContent = lags.length ? `${(lags.reduce((a, b) => a + b, 0) / lags.length).toFixed(1)}` : "–";
    if (lags.length) statEls.avg.append(h("small", null, "s"));
  }

  function drawState() {
    const st = live.state;
    recBtn.className = `rec-btn${st === "recording" ? " on" : ""}${["preparing", "connecting", "stopping"].includes(st) ? " busy" : ""}`;
    recBtn.disabled = st === "stopping" || (st !== "recording" && opts.source === "mic" && !micAvailable());
    recBtn.title = st === "recording" ? "Stop" : st === "stopping" ? "Finishing…" : ["preparing", "connecting"].includes(st) ? "Cancel" : "Start";
    what.textContent = {
      idle: "Ready", preparing: live.source === "file" ? "Decoding the file…" : "Opening the microphone…", connecting: "Connecting…",
      recording: live.source === "file" ? `Streaming ${live.speed}×` : "Recording", stopping: "Finishing the last lines…",
      done: "Finished", error: "Stopped with an error",
    }[st];
    sourceEl.textContent = live.sourceLabel || "";
    const locked = live.active;
    optionsCard.querySelectorAll("button, select, input").forEach((x) => (x.disabled = locked));
    clear(doneBox);
    if (st === "done" && live.conversationId) {
      const c = live.completed;
      doneBox.append(h("div", { class: "notice info" }, icon("check"), h("div", { class: "spacer" },
        h("b", null, "Saved. "), c ? `${c.num_segments} lines, ${c.num_speakers} speakers, ${fmtDuration(c.duration)}.` : "",
        c?.processing_errors ? ` ${c.processing_errors} chunk(s) failed.` : "",
        " Open it to play it back, correct speakers and emotions."),
        h("a", { class: "btn primary sm", href: `#/conversations/${live.conversationId}` }, "Open conversation", icon("chevronRight"))));
    } else if (st === "error") {
      doneBox.append(h("div", { class: "notice" }, icon("alert"), h("div", null, h("b", null, "Recording stopped. "), live.error || "")));
    }
    drawStats();
  }

  function onRecord() {
    const st = live.state;
    if (st === "recording" || st === "connecting" || st === "preparing") return live.stop("stopped by you");
    if (opts.source === "file" && !opts.file) { toast("Choose an audio file first", "err"); fileInput.click(); return; }
    live.start({ ...opts }).then(fillMics);
    drawFeed();
    drawSpeakers();
    drawLog();
  }

  // ----- live events -----
  const on = {
    state: () => drawState(),
    level: (e) => { meterFill.style.width = `${e.detail * 100}%`; drawStats(); },
    status: () => vad.classList.toggle("on", live.vad),
    segment: (e) => {
      feedEmpty.remove();
      feed.append(addRow(e.detail, true));
      if (autoScroll) feed.scrollTop = feed.scrollHeight;
      feedCount.textContent = `${live.segments.length} lines`;
      drawSpeakers();
      drawStats();
    },
    speaker_update: (e) => {
      for (const { seg, before } of e.detail.changed) {
        palette.rename(speakerKey(before), speakerKey(seg));
        const old = rows.get(seg.id);
        const row = addRow(seg, false);
        row.querySelector(".chip")?.classList.add("flash");
        old?.replaceWith(row);
      }
      drawSpeakers();
    },
    log: (e) => {
      logBox.append(logLine(e.detail));
      while (logBox.childElementCount > 500) logBox.firstChild.remove();
      logBox.scrollTop = logBox.scrollHeight;
      logCount.textContent = `${live.events.length} events · ${live.statusMessages} status`;
    },
    error: (e) => toast(e.detail || "Server error", "err"),
  };
  for (const [k, fn] of Object.entries(on)) live.addEventListener(k, fn);

  const insecure = !window.isSecureContext ? h("div", { class: "notice" }, icon("alert"), h("div", null,
    h("b", null, "The microphone needs HTTPS. "), "Browsers only allow it on secure pages (or on localhost). Start the server with a certificate in ",
    h("code", null, "certs/"), " (", h("code", null, "./generate_cert.sh"), " makes a self-signed one) and open ",
    h("code", null, `https://${location.host}`), ". Streaming a file works without it.")) : null;

  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" }, h("div", null, h("h1", null, "Live"),
      h("p", { class: "sub" }, "Stream your microphone, or a file at real-time pace, and see who is speaking as it happens. Each voice is followed through the session and named as soon as it clearly matches someone you have named before."))),
    insecure,
    h("div", { class: "live-top" }, h("div", { class: "stack", style: { gap: "16px" } }, recorder, stats, doneBox), optionsCard),
    h("div", { class: "grid", style: { gridTemplateColumns: "minmax(0, 1fr) 320px" }, id: "live-grid" },
      h("div", { class: "card" }, h("div", { class: "card-head" }, h("h2", null, icon("chat"), "Transcript"), feedCount), feed),
      h("div", { class: "card", style: { alignSelf: "start" } }, h("div", { class: "card-head" }, h("h2", null, icon("users"), "Voices")),
        h("div", { class: "card-body" }, speakersBox))),
    h("details", { class: "card collapse" },
      h("summary", { class: "card-head" }, h("h2", null, icon("chevronRight", "chev"), icon("terminal"), "Event log"), logCount),
      logBox)));

  const narrow = matchMedia("(max-width: 1000px)");
  const layout = () => { root.querySelector("#live-grid").style.gridTemplateColumns = narrow.matches ? "1fr" : "minmax(0, 1fr) 320px"; };
  narrow.addEventListener("change", layout);
  layout();

  drawFeed();
  drawSpeakers();
  drawLog();
  drawState();
  vad.classList.toggle("on", live.vad);

  return () => {
    for (const [k, fn] of Object.entries(on)) live.removeEventListener(k, fn);
    narrow.removeEventListener("change", layout);
  };
}
