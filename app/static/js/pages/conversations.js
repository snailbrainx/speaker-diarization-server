// Every conversation (uploads and live recordings), newest first.

import { api } from "../api.js";
import { h, clear, icon, button, toast, toastError, confirmDialog, loading, emptyState, fmtDate, fmtDuration, fmtRelative } from "../ui.js";
import { conversationPage } from "../viewer.js";

const PAGE = 50;

const STATUS = {
  completed: ["ok", "completed"], recording: ["err", "● recording"], processing: ["warn", "processing"],
  finalizing: ["warn", "finalizing"], failed: ["err", "failed"], deleting: ["warn", "deleting"],
};

export function render(root, params) {
  if (params.id) return conversationPage(root, params.id);

  let items = [];
  let total = 0;
  let query = "";
  const body = h("div", { class: "card" }, loading());
  const countEl = h("span", { class: "small muted" });

  async function load(more = false) {
    try {
      const r = await api.conversations({ skip: more ? items.length : 0, limit: PAGE });
      items = more ? items.concat(r.conversations) : r.conversations;
      total = r.total;
      draw();
    } catch (e) {
      clear(body).append(emptyState("alert", "Cannot load conversations", e.message));
    }
  }

  function draw() {
    clear(body);
    const q = query.trim().toLowerCase();
    const shown = items.filter((c) => !q || (c.title || "").toLowerCase().includes(q) || String(c.id) === q);
    countEl.textContent = `${total} conversation${total === 1 ? "" : "s"}`;
    if (!items.length) {
      body.append(emptyState("chat", "No conversations yet", "Record something live or transcribe a file, and it shows up here.",
        h("div", { class: "row", style: { justifyContent: "center" } },
          h("a", { class: "btn primary", href: "#/live" }, icon("mic"), "Go live"),
          h("a", { class: "btn", href: "#/process" }, icon("upload"), "Transcribe a file"))));
      return;
    }
    const tbody = h("tbody");
    for (const c of shown) {
      const [kind, label] = STATUS[c.status] || ["", c.status];
      const del = button("", { kind: "ghost", size: "sm", icon: "trash", title: "Delete" });
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!(await confirmDialog({ title: "Delete this conversation?", danger: true, confirmLabel: "Delete", icon: "trash",
          message: `“${c.title || `Conversation ${c.id}`}”: its recording and transcript are removed. Speakers stay enrolled.` }))) return;
        try {
          await api.deleteConversation(c.id);
          toast("Conversation deleted", "ok");
          items = items.filter((x) => x.id !== c.id);
          total--;
          draw();
        } catch (err) { toastError(err); }
      });
      tbody.append(h("tr", { class: "click", onclick: () => { location.hash = `#/conversations/${c.id}`; } },
        h("td", { class: "title-cell" }, h("div", { class: "row", style: { flexWrap: "nowrap", gap: "10px" } },
          icon(c.title?.startsWith("Uploaded:") ? "file" : "mic"),
          h("span", { style: { overflow: "hidden", textOverflow: "ellipsis" } }, c.title || `Conversation ${c.id}`))),
        h("td", { title: fmtDate(c.start_time) }, fmtRelative(c.start_time)),
        h("td", { class: "num" }, fmtDuration(c.duration)),
        h("td", { class: "num" }, c.num_speakers ?? "–"),
        h("td", { class: "num" }, c.num_segments ?? "–"),
        h("td", null, h("span", { class: `badge ${kind}` }, label)),
        h("td", { style: { textAlign: "right" } }, del)));
    }
    body.append(h("div", { class: "table-wrap" }, h("table", { class: "table" },
      h("thead", null, h("tr", null, ["Title", "When", "Length", "Speakers", "Lines", "Status", ""].map((t) => h("th", null, t)))),
      tbody)));
    if (!shown.length) body.append(h("div", { class: "empty" }, h("p", null, "No title matches.")));
    if (items.length < total) {
      body.append(h("div", { class: "row", style: { justifyContent: "center", padding: "14px" } },
        button(`Load more (${total - items.length} left)`, { onClick: (e) => { e.currentTarget.disabled = true; load(true); } })));
    }
  }

  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Conversations"), h("p", { class: "sub" }, "Uploads and live recordings. Open one to play it back with its speakers, correct who said what, and fix emotions.")),
      h("div", { class: "actions" },
        h("div", { class: "search", style: { width: "240px" } }, icon("search"),
          h("input", { class: "input", placeholder: "Filter by title…", oninput: (e) => { query = e.target.value; draw(); } })),
        button("", { icon: "refresh", title: "Refresh", onClick: () => load() }))),
    h("div", { class: "row" }, countEl),
    body));
  load();
}
