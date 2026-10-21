// Small DOM toolkit shared by every page: element builder, icons, formatting, speaker colours, emotions,
// toasts, dialogs and menus. Text always goes in as text nodes, never as HTML.

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") {
      for (const [p, val] of Object.entries(v)) {
        if (p.startsWith("--")) el.style.setProperty(p, val);
        else el.style[p] = val;
      }
    } else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (typeof v !== "string" && k in el) el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const c of [children].flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : String(c));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.firstChild.remove();
  return el;
}

// ---------- icons (24x24, stroked) ----------
const ICONS = {
  mic: '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M8.5 21h7"/>',
  upload: '<path d="M12 15V4M7 9l5-5 5 5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
  chat: '<path d="M20.5 12a8.5 8.5 0 0 1-12.3 7.6L3.5 20.5l1-4.4A8.5 8.5 0 1 1 20.5 12z"/><path d="M8.5 10.5h7M8.5 14h4.5"/>',
  users: '<circle cx="9" cy="8" r="3.5"/><path d="M2.5 20a6.5 6.5 0 0 1 13 0"/><path d="M16 4.6a3.5 3.5 0 0 1 0 6.8M18 14.6a6.5 6.5 0 0 1 3.5 5.4"/>',
  archive: '<rect x="3" y="4" width="18" height="5" rx="1.5"/><path d="M5 9v9a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9M10 13h4"/>',
  sliders: '<path d="M4 7h9M18 7h2M4 17h3M12 17h8"/><circle cx="15.5" cy="7" r="2.5"/><circle cx="9.5" cy="17" r="2.5"/>',
  play: '<path d="M8 5.6v12.8a1 1 0 0 0 1.5.86l10.4-6.4a1 1 0 0 0 0-1.72L9.5 4.74A1 1 0 0 0 8 5.6z" fill="currentColor" stroke="none"/>',
  pause: '<rect x="6.5" y="5" width="3.8" height="14" rx="1" fill="currentColor" stroke="none"/><rect x="13.7" y="5" width="3.8" height="14" rx="1" fill="currentColor" stroke="none"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" stroke="none"/>',
  trash: '<path d="M4 7h16M9 7V4.5h6V7M6 7l1 13h10l1-13M10 11v6M14 11v6"/>',
  edit: '<path d="M4 20h4L19 9a2.8 2.8 0 0 0-4-4L4 16v4z"/><path d="M13.5 6.5l4 4"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  refresh: '<path d="M20 11a8 8 0 0 0-14.3-4.9L4 8M4 4v4h4M4 13a8 8 0 0 0 14.3 4.9L20 16M20 20v-4h-4"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M4 19h16"/>',
  smile: '<circle cx="12" cy="12" r="9"/><path d="M8.5 14.5a4.5 4.5 0 0 0 7 0M9 9.5h.01M15 9.5h.01"/>',
  identify: '<circle cx="10" cy="8" r="4"/><path d="M3 20a7 7 0 0 1 11.3-5.5M15.5 18l2.2 2.2 4.3-4.4"/>',
  userPlus: '<circle cx="10" cy="8" r="4"/><path d="M3 20a7 7 0 0 1 14 0M19 8v6M16 11h6"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
  chevronRight: '<path d="M9 6l6 6-6 6"/>',
  chevronLeft: '<path d="M15 6l-6 6 6 6"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13v4M12 11v6M15 14v3"/>',
  wave: '<path d="M3 12h1M7 8v8M11 4v16M15 8v8M19 10v4M21 12h0"/>',
  alert: '<path d="M10.3 4.2L2.6 18a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0z"/><path d="M12 10v4M12 17.5h.01"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 7.5h.01"/>',
  more: '<circle cx="5" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="19" cy="12" r="1.4" fill="currentColor"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  calendar: '<rect x="3.5" y="5" width="17" height="15" rx="2"/><path d="M3.5 10h17M8 3v4M16 3v4"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4"/>',
  save: '<path d="M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
  flag: '<path d="M5 21V4M5 4h11l-2 4 2 4H5"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  history: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5M12 8v4l3 2"/>',
  volume: '<path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/>',
  server: '<rect x="3.5" y="4" width="17" height="7" rx="2"/><rect x="3.5" y="13" width="17" height="7" rx="2"/><path d="M7.5 7.5h.01M7.5 16.5h.01"/>',
  code: '<path d="M8 8l-4 4 4 4M16 8l4 4-4 4M13.5 5l-3 14"/>',
  radio: '<circle cx="12" cy="12" r="2"/><path d="M8.5 15.5a5 5 0 0 1 0-7M15.5 8.5a5 5 0 0 1 0 7M5.6 18.4a9 9 0 0 1 0-12.8M18.4 5.6a9 9 0 0 1 0 12.8"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  target: '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  terminal: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 9l3 3-3 3M12.5 15H17"/>',
};

export function icon(name, cls) {
  const t = document.createElement("template");
  t.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"
    stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ""}</svg>`;
  const svg = t.content.firstElementChild;
  if (cls) svg.setAttribute("class", cls);
  return svg;
}

// ---------- formatting ----------
export function fmtClock(sec, decimals = 0) {
  if (sec == null || !isFinite(sec)) return "–:––";
  sec = Math.max(0, sec);
  const hh = Math.floor(sec / 3600);
  const mm = Math.floor((sec % 3600) / 60);
  const ss = sec % 60;
  const s = decimals ? ss.toFixed(decimals).padStart(3 + decimals, "0") : String(Math.floor(ss)).padStart(2, "0");
  return hh ? `${hh}:${String(mm).padStart(2, "0")}:${s}` : `${mm}:${s}`;
}

export function fmtDuration(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)} s`;
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  if (m < 60) return `${m}m ${String(s).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

// The API's datetimes are UTC; older rows have no zone suffix
export function parseDate(v) {
  if (!v) return null;
  if (typeof v === "string" && !/[zZ]|[+-]\d\d:?\d\d$/.test(v)) v += "Z";
  const d = new Date(v);
  return isNaN(d) ? null : d;
}

export function fmtDate(v) {
  const d = parseDate(v);
  if (!d) return "–";
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function fmtRelative(v) {
  const d = parseDate(v);
  if (!d) return "–";
  const s = (Date.now() - d) / 1000;
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  if (s < 86400 * 7) return `${Math.round(s / 86400)} d ago`;
  return d.toLocaleDateString(undefined, { dateStyle: "medium" });
}

export function fmtBytes(n) {
  if (!n && n !== 0) return "–";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
}

export const pct = (x, digits = 0) => (x == null || !isFinite(x) ? "–" : `${(x * 100).toFixed(digits)}%`);

// ---------- speakers ----------
// Unknown_<n> is a voice nobody has named yet (one stored speaker); a plain "Unknown" line is speech whose voice could
// not be placed (live: too little of it so far; it joins its voice once that has a speaker)
export const isUnknown = (name) => !name || name === "Unknown" || name.startsWith("Unknown_");
export const isUnnamedVoice = (name) => !!name && name.startsWith("Unknown_");
export const isDeleted = (name) => !!name && name.startsWith("Deleted_");
// Unknown_<number> names are long and meaningless to read; their speaker id is short, stable and shown on the
// Speakers page
export const displayName = (name, id) => (name && /^Unknown_\d{5,}$/.test(name) && id != null ? `Unknown #${id}` : name || "Unknown");

// Colours are handed out in order of appearance within one view, so the people in a conversation never share one.
// A renamed speaker keeps its colour (see rename), so a live relabel does not repaint the transcript.
export class Palette {
  constructor() { this.map = new Map(); this.next = 0; }
  color(key) {
    if (!this.map.has(key)) this.map.set(key, `var(--spk-${this.next++ % 10})`);
    return this.map.get(key);
  }
  rename(oldKey, newKey) {
    if (this.map.has(oldKey) && !this.map.has(newKey)) this.map.set(newKey, this.map.get(oldKey));
  }
}

// A stable colour for one speaker id, for pages that list speakers outside any conversation
export const idColor = (id) => `var(--spk-${Math.abs(Number(id) || 0) % 10})`;

export function speakerChip(name, color, { id, onClick, title, flash } = {}) {
  const unknown = isUnknown(name);
  const shown = displayName(name, id);
  const hint = title || (isUnnamedVoice(name) ? "Not identified yet: name this voice once and it is recognised from then on"
    : unknown ? "Voice not placed: too little speech to tell whose it is" : name);
  const el = h(onClick ? "button" : "span", {
    class: `chip${unknown ? " unknown" : ""}${onClick ? " button" : ""}${flash ? " flash" : ""}`,
    style: { "--c": isDeleted(name) ? "var(--muted)" : color },
    title: shown !== name && name ? `${name} · ${hint}` : hint,
    onclick: onClick, type: onClick ? "button" : null,
  }, h("span", { class: "dot" }), h("span", { class: "t" }, shown));
  return el;
}

export function initials(name) {
  if (isUnknown(name)) return "?";
  const parts = name.replace(/[_-]+/g, " ").trim().split(/\s+/);
  return ((parts[0]?.[0] || "") + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase() || "?";
}

// ---------- emotions (emotion2vec categories) ----------
export const EMOTIONS = {
  neutral: { e: "😐", label: "Neutral" },
  happy: { e: "😊", label: "Happy" },
  sad: { e: "😢", label: "Sad" },
  angry: { e: "😠", label: "Angry" },
  fearful: { e: "😨", label: "Fearful" },
  surprised: { e: "😮", label: "Surprised" },
  disgusted: { e: "🤢", label: "Disgusted" },
  other: { e: "🤔", label: "Other" },
};
// emotion2vec answers <unk> when no category fits
export const emotionInfo = (cat) => EMOTIONS[cat] || { e: "❔", label: !cat || /^<?unk/.test(cat) ? "Unclear" : cat };
export const emotionColor = (cat) => `var(--emo-${EMOTIONS[cat] ? cat : "unknown"})`;

export function emotionChip(cat, conf, { corrected, onClick, title } = {}) {
  if (!cat) return null;
  const info = emotionInfo(cat);
  return h(onClick ? "button" : "span", {
    class: `emo${corrected ? " corrected" : ""}${onClick ? " button" : ""}`, type: onClick ? "button" : null,
    style: { "--c": emotionColor(cat) }, onclick: onClick,
    title: title || `${info.label}${conf != null ? ` (${pct(conf)} confidence)` : ""}${corrected ? ", corrected by hand" : ""}`,
  }, h("span", { class: "e" }, info.e), info.label, conf != null && !corrected ? h("span", { class: "p" }, pct(conf)) : null,
     corrected ? icon("check") : null);
}

// ---------- feedback ----------
export function toast(message, kind = "info", timeout = 4500) {
  const box = document.getElementById("toasts");
  const el = h("div", { class: `toast ${kind}`, role: kind === "err" ? "alert" : "status" },
    icon(kind === "ok" ? "check" : kind === "err" ? "alert" : "info"), h("div", null, message));
  box.append(el);
  const remove = () => { el.classList.add("out"); setTimeout(() => el.remove(), 220); };
  setTimeout(remove, timeout);
  el.addEventListener("click", remove);
}

export const toastError = (err) => toast(err?.message || String(err), "err", 7000);

export function spinner(cls = "") { return h("span", { class: `spinner ${cls}` }); }
export function loading(text = "Loading…") { return h("div", { class: "loading" }, spinner(), text); }

export function emptyState(iconName, title, text, action) {
  return h("div", { class: "empty" }, icon(iconName), h("h3", null, title), text ? h("p", null, text) : null, action || null);
}

export function button(label, { kind = "", icon: ic, onClick, title, size = "", disabled, type = "button" } = {}) {
  const cls = ["btn", kind, size, !label && ic ? "icon" : ""].filter(Boolean).join(" ");
  return h("button", { class: cls, type, title: title || (label ? null : ""), "aria-label": title || null, onclick: onClick, disabled },
    ic ? icon(ic) : null, label || null);
}

// Runs an async action from a button: disabled with a spinner meanwhile; errors become a toast
export async function busy(btn, fn) {
  const old = [...btn.childNodes];
  btn.disabled = true;
  const sp = spinner();
  sp.style.width = sp.style.height = "14px";
  btn.prepend(sp);
  old.filter((n) => n.nodeName === "svg").forEach((n) => n.remove());
  try {
    return await fn();
  } catch (e) {
    toastError(e);
  } finally {
    if (btn.isConnected) { clear(btn); append(btn, old); btn.disabled = false; }
  }
}

// ---------- dialogs ----------
// actions: [{label, kind, icon, onClick}] - onClick may be async; returning false keeps the dialog open.
export function modal({ title, body, actions = [], wide = false, onClose, cancelLabel = "Cancel" }) {
  const dlg = h("dialog", { class: `modal${wide ? " wide" : ""}` });
  const close = () => dlg.open && dlg.close();
  const foot = h("div", { class: "modal-foot" });
  if (cancelLabel) foot.append(button(cancelLabel, { kind: "ghost", onClick: close }));
  for (const a of actions) {
    const b = button(a.label, { kind: a.kind, icon: a.icon });
    b.addEventListener("click", async () => {
      const buttons = [...foot.querySelectorAll("button")];
      buttons.forEach((x) => (x.disabled = true));
      try {
        const keep = await busy(b, async () => (await a.onClick?.()) === false);
        if (keep === false) close();
      } finally {
        buttons.forEach((x) => x.isConnected && (x.disabled = false));
      }
    });
    foot.append(b);
  }
  append(dlg, [
    h("div", { class: "modal-head" }, h("h2", null, title), button("", { kind: "ghost", size: "sm", icon: "x", title: "Close", onClick: close })),
    h("div", { class: "modal-body" }, body),
    actions.length || cancelLabel ? foot : null,
  ]);
  dlg.addEventListener("close", () => { dlg.remove(); onClose?.(); });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });  // click on the backdrop
  document.body.append(dlg);
  dlg.showModal();
  return { dlg, close };
}

export function confirmDialog({ title, message, confirmLabel = "Confirm", danger = false, icon: ic }) {
  return new Promise((resolve) => {
    let ok = false;
    modal({
      title,
      body: typeof message === "string" ? h("p", null, message) : message,
      actions: [{ label: confirmLabel, kind: danger ? "danger solid" : "primary", icon: ic, onClick: () => { ok = true; } }],
      onClose: () => resolve(ok),
    });
  });
}

export function promptDialog({ title, label, value = "", placeholder = "", confirmLabel = "Save", hint, multiline }) {
  return new Promise((resolve) => {
    let result = null;
    const input = multiline
      ? h("textarea", { class: "input", placeholder }, value)
      : h("input", { class: "input", value, placeholder });
    const body = h("div", { class: "field" }, label ? h("label", null, label) : null, input, hint ? h("div", { class: "hint" }, hint) : null);
    const m = modal({
      title, body,
      actions: [{ label: confirmLabel, kind: "primary", onClick: () => { result = input.value.trim(); } }],
      onClose: () => resolve(result),
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !multiline) { e.preventDefault(); m.dlg.querySelector(".modal-foot .primary").click(); }
    });
    setTimeout(() => { input.focus(); input.select?.(); }, 30);
  });
}

// A small dropdown menu next to an element. items: [{label, icon, onClick, danger}] or "-" for a separator.
export function openMenu(anchor, items) {
  document.querySelectorAll(".menu").forEach((m) => m.remove());
  const menu = h("div", { class: "menu", role: "menu" });
  for (const it of items) {
    if (it === "-") { menu.append(h("hr")); continue; }
    if (!it) continue;
    menu.append(h("button", {
      type: "button", class: it.danger ? "danger" : "", role: "menuitem",
      onclick: () => { menu.remove(); it.onClick?.(); },
    }, it.icon ? icon(it.icon) : null, it.label));
  }
  document.body.append(menu);
  const r = anchor.getBoundingClientRect();
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  menu.style.left = `${Math.max(8, Math.min(r.right - mw, innerWidth - mw - 8))}px`;
  menu.style.top = `${r.bottom + 6 + mh > innerHeight ? Math.max(8, r.top - mh - 6) : r.bottom + 6}px`;
  const off = (e) => {
    if (e.type === "keydown" && e.key !== "Escape") return;
    if ((e.type === "mousedown" || e.type === "wheel") && menu.contains(e.target)) return;
    menu.remove();
    removeEventListener("mousedown", off, true);
    removeEventListener("keydown", off, true);
    removeEventListener("wheel", off, true);
    removeEventListener("touchmove", off, true);
    removeEventListener("resize", off);
  };
  setTimeout(() => {
    addEventListener("mousedown", off, true);
    addEventListener("keydown", off, true);
    // the user scrolling closes it; the page re-rendering underneath does not
    addEventListener("wheel", off, true);
    addEventListener("touchmove", off, true);
    addEventListener("resize", off);
  });
}

export function downloadText(filename, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function switchInput(label, { checked = false, desc, onChange } = {}) {
  const input = h("input", { type: "checkbox", checked, onchange: (e) => onChange?.(e.target.checked) });
  const el = h("label", { class: "switch" }, input, h("span", null, label, desc ? h("span", { class: "desc" }, desc) : null));
  el.input = input;
  return el;
}

export function segmented(options, value, onChange) {
  const el = h("div", { class: "segmented", role: "tablist" });
  const set = (v) => {
    value = v;
    el.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
  };
  for (const o of options) {
    el.append(h("button", {
      type: "button", dataset: { v: o.value }, role: "tab",
      onclick: () => { if (o.value !== value) { set(o.value); onChange(o.value); } },
    }, o.icon ? icon(o.icon) : null, o.label));
  }
  set(value);
  el.set = set;
  return el;
}
