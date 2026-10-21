// Backups ("voice profiles" in the API): named snapshots of the speakers, their voices and emotion profiles, who said
// which line, and the settings. A profile is saved over as you go; checkpoints are dated copies of one.

import { api } from "../api.js";
import { h, clear, icon, button, toast, toastError, confirmDialog, modal, loading, emptyState, fmtDate, count } from "../ui.js";

// When the file was written (created_at carries its time zone); else its name's timestamp, 20261021_142233 in the
// server's local time
function stamp(ts, createdAt) {
  if (createdAt) return fmtDate(createdAt);
  const m = /^(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)/.exec(ts || "");
  return m ? new Date(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6]).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "–";
}

export function render(root) {
  let profiles = [];
  const open = new Set();
  const list = h("div", { class: "stack" }, loading());

  async function load() {
    try {
      profiles = (await api.profiles()).profiles;
      draw();
    } catch (e) {
      clear(list).append(h("div", { class: "card" }, emptyState("alert", "Cannot load backups", e.message)));
    }
  }

  function draw() {
    clear(list);
    if (!profiles.length) {
      list.append(h("div", { class: "card" }, emptyState("archive", "No backups yet",
        "Save the current speakers and settings under a name, so you can come back to them or move them to another server.",
        button("Save current state", { kind: "primary", icon: "save", onClick: saveAs }))));
      return;
    }
    for (const p of profiles) list.append(profileCard(p));
  }

  function profileCard(p) {
    const cps = h("div", { class: "checkpoints" + (open.has(p.name) ? "" : " hidden") });
    const card = h("div", { class: "card profile" },
      h("div", { class: "top" },
        h("div", { style: { minWidth: 0 } },
          h("h2", null, p.name),
          p.description ? h("p", { class: "muted", style: { marginTop: "4px" } }, p.description) : null,
          h("div", { class: "kv", style: { marginTop: "8px" } },
            h("span", null, h("b", null, p.speakers_count), " speakers"),
            h("span", null, h("b", null, p.segments_count), p.segments_count === 1 ? " line" : " lines"),
            h("span", null, "saved ", h("b", null, stamp(p.timestamp, p.created_at))))),
        h("div", { class: "actions" },
          button("Restore", { kind: "primary", size: "sm", icon: "history", onClick: () => restore(p.filename, p.name) }),
          button("Save over", { size: "sm", icon: "save", title: "Replace this backup with the current state", onClick: () => saveOver(p) }),
          button("Checkpoints", { size: "sm", icon: "clock", onClick: () => toggleCheckpoints(p, cps) }),
          h("a", { class: "btn sm ghost icon", href: api.profileDownloadUrl(p.name), title: "Download", download: "" }, icon("download")),
          button("", { size: "sm", kind: "ghost", icon: "trash", title: "Delete", onClick: () => remove(p) }))),
      cps);
    if (open.has(p.name)) loadCheckpoints(p, cps);
    return card;
  }

  function toggleCheckpoints(p, box) {
    if (open.has(p.name)) { open.delete(p.name); box.classList.add("hidden"); return; }
    open.add(p.name);
    box.classList.remove("hidden");
    loadCheckpoints(p, box);
  }

  async function loadCheckpoints(p, box) {
    clear(box).append(loading("Loading checkpoints…"));
    let cps = [];
    try { cps = (await api.checkpoints(p.name)).checkpoints; } catch (e) { clear(box).append(h("p", { class: "muted" }, e.message)); return; }
    clear(box).append(h("div", { class: "row" }, h("b", null, "Checkpoints"), h("span", { class: "small muted" }, "dated snapshots of the current state, filed under this backup"),
      h("span", { class: "spacer" }),
      button("New checkpoint", { size: "sm", icon: "plus", onClick: async (e) => {
        e.currentTarget.disabled = true;
        try { await api.createCheckpoint(p.name); toast("Checkpoint saved", "ok"); } catch (err) { toastError(err); }
        loadCheckpoints(p, box);
      } })));
    if (!cps.length) box.append(h("p", { class: "small muted" }, "None yet."));
    for (const c of cps) {
      box.append(h("div", { class: "checkpoint" },
        icon("clock"), h("span", null, stamp(c.timestamp, c.created_at)),
        h("span", { class: "muted" }, `${count(c.speakers_count, "speaker")} · ${count(c.segments_count, "line")}`),
        h("span", { class: "spacer" }),
        button("Restore", { size: "sm", icon: "history", onClick: () => restore(c.filename, `${p.name} (${stamp(c.timestamp, c.created_at)})`) }),
        button("", { size: "sm", kind: "ghost", icon: "trash", title: "Delete checkpoint", onClick: async () => {
          if (!(await confirmDialog({ title: "Delete this checkpoint?", danger: true, confirmLabel: "Delete", message: stamp(c.timestamp, c.created_at) }))) return;
          try { await api.deleteCheckpoint(p.name, c.timestamp); toast("Checkpoint deleted", "ok"); } catch (e) { toastError(e); }
          loadCheckpoints(p, box);
        } })));
    }
  }

  async function restore(filename, label) {
    if (!(await confirmDialog({
      title: `Restore ${label}?`, confirmLabel: "Restore", danger: true, icon: "history",
      message: h("div", { class: "stack" },
        h("p", null, "The current speakers, voice profiles, emotion profiles and settings are replaced by the ones in this backup, and lines are relabelled to match."),
        h("p", { class: "muted" }, "Save the current state first (a checkpoint) if you may want it back. A backup made with an older voice model is converted as it is restored.")),
    }))) return;
    toast("Restoring…", "info", 2500);
    try {
      const r = await api.restore(filename);
      toast(`${r.message}: ${r.speakers_restored} speakers, ${r.segments_updated} lines relabelled`, "ok", 7000);
      await load();
    } catch (e) { toastError(e); }
  }

  async function saveOver(p) {
    if (!(await confirmDialog({ title: `Save over “${p.name}”?`, confirmLabel: "Save", icon: "save",
      message: "This backup is replaced with the current speakers, lines and settings." }))) return;
    try {
      await api.updateProfile(p.name, p.description);
      toast(`Saved over ${p.name}`, "ok");
      await load();
    } catch (e) { toastError(e); }
  }

  async function remove(p) {
    if (!(await confirmDialog({ title: `Delete “${p.name}”?`, danger: true, confirmLabel: "Delete", icon: "trash",
      message: "The backup file is removed. Nothing in the current state changes." }))) return;
    try {
      await api.deleteProfile(p.name);
      toast("Backup deleted", "ok");
      await load();
    } catch (e) { toastError(e); }
  }

  function nameDialog(title, confirmLabel, run) {
    const name = h("input", { class: "input", placeholder: "e.g. home, office, before-tuning", maxlength: 80 });
    const desc = h("textarea", { class: "input", placeholder: "Optional" });
    modal({
      title,
      body: h("div", { class: "stack" }, h("div", { class: "field" }, h("label", null, "Name"), name),
        h("div", { class: "field" }, h("label", null, "Description"), desc)),
      actions: [{
        label: confirmLabel, kind: "primary", icon: "save",
        onClick: async () => {
          const n = name.value.trim();
          if (!n) { toast("Give it a name", "err"); return false; }
          const r = await run(n, desc.value.trim());
          toast(r.message || "Saved", "ok");
          await load();
        },
      }],
    });
    setTimeout(() => name.focus(), 30);
  }

  const saveAs = () => nameDialog("Save the current state", "Save", (n, d) => api.saveAsProfile(n, d));
  const createEmpty = () => nameDialog("New empty backup", "Create", (n, d) => api.createProfile(n, d));

  const importInput = h("input", { type: "file", accept: ".json,application/json", class: "hidden", onchange: async (e) => {
    const f = e.target.files[0];
    e.target.value = "";
    if (!f) return;
    try {
      const r = await api.importProfile(f);
      toast(r.message, "ok");
      await load();
    } catch (err) { toastError(err); }
  } });

  root.append(h("div", { class: "page" },
    h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Backups"),
        h("p", { class: "sub" }, "Named snapshots of who is who: speakers with their voices and emotion profiles, which speaker said each line, and the settings. Restore one to go back, or download it to move to another server.")),
      h("div", { class: "actions" },
        button("Import", { icon: "upload", onClick: () => importInput.click() }),
        h("a", { class: "btn", href: api.allProfilesDownloadUrl(), download: "" }, icon("download"), "Download all"),
        button("Empty", { icon: "plus", title: "A backup with no speakers and default settings", onClick: createEmpty }),
        button("Save current state", { kind: "primary", icon: "save", onClick: saveAs }),
        importInput)),
    list));
  load();
}

