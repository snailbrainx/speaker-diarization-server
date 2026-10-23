// App shell: navigation, hash routing (#/page/id), theme and the server status pill.

import { api } from "./api.js";
import { live } from "./live-session.js";
import { h, clear, icon } from "./ui.js";
import * as livePage from "./pages/live.js";
import * as processPage from "./pages/process.js";
import * as conversationsPage from "./pages/conversations.js";
import * as speakersPage from "./pages/speakers.js";
import * as backupsPage from "./pages/backups.js";
import * as settingsPage from "./pages/settings.js";

const PAGES = [
  { path: "live", label: "Live", icon: "mic", page: livePage },
  { path: "process", label: "Transcribe a file", icon: "upload", page: processPage },
  { path: "conversations", label: "Conversations", icon: "chat", page: conversationsPage },
  { path: "speakers", label: "Speakers", icon: "users", page: speakersPage },
  { path: "backups", label: "Backups", icon: "archive", page: backupsPage },
  { path: "settings", label: "Settings", icon: "sliders", page: settingsPage },
];

const main = document.getElementById("main");
const nav = document.getElementById("nav");
let cleanup = null;

function drawNav(active) {
  clear(nav);
  for (const p of PAGES) {
    nav.append(h("a", { href: `#/${p.path}`, class: p.path === active ? "active" : "", title: p.label },
      icon(p.icon), h("span", null, p.label),
      p.path === "live" && live.active ? h("i", { class: "rec-dot", title: "Recording" }) : null));
  }
}

function route() {
  const [path = "live", id] = location.hash.replace(/^#\/?/, "").split("/");
  const entry = PAGES.find((p) => p.path === path) || PAGES[0];
  drawNav(entry.path);
  try { cleanup?.(); } catch { /* a page failing to clean up must not block the next one */ }
  cleanup = null;
  clear(main);
  document.querySelectorAll(".menu").forEach((m) => m.remove());
  cleanup = entry.page.render(main, { id: id ? decodeURIComponent(id) : null }) || null;
  document.title = `${entry.label} · Speaker Diarization`;
  window.scrollTo(0, 0);
}

addEventListener("hashchange", route);
live.addEventListener("state", () => drawNav((location.hash.replace(/^#\/?/, "").split("/")[0]) || "live"));
addEventListener("beforeunload", (e) => {
  if (live.active) { e.preventDefault(); e.returnValue = ""; }
});

// ---------- theme ----------
const themeBtn = document.getElementById("theme-toggle");
const isDark = () => {
  const t = document.documentElement.dataset.theme;
  return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
};
function drawTheme() {
  clear(themeBtn).append(icon(isDark() ? "sun" : "moon"));
  themeBtn.title = isDark() ? "Light theme" : "Dark theme";
}
themeBtn.addEventListener("click", () => {
  const t = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("theme", t); } catch { /* private mode */ }
  drawTheme();
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawTheme);
drawTheme();

// ---------- server status ----------
const pill = document.getElementById("server-pill");
async function pollStatus() {
  try {
    const s = await api.status();
    pill.className = "server-pill ok";
    pill.querySelector(".label").textContent = s.gpu_available ? `Online · GPU (${s.device})` : "Online · CPU";
    pill.title = s.message;
  } catch {
    pill.className = "server-pill down";
    pill.querySelector(".label").textContent = "Server unreachable";
  }
}
pollStatus();
setInterval(pollStatus, 20000);

if (!location.hash) history.replaceState(null, "", "#/live");
route();
