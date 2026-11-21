// Board: one row per film/series, five stage cells, status line. Refreshes every 5 s,
// keeps opened detail panels and the collapsed state of the "done" group.
(() => {
  const root = document.getElementById("board");
  if (!root) return;
  const panels = new Map();            // row key -> detached panel element (kept across refreshes)
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function cell(c) {
    const pct = c.progress == null ? (c.status === "done" ? 100 : 0) : Math.round(c.progress * 100);
    return `<span class="cell c-${c.status}" title="${pct}%"><i style="width:${pct}%"></i></span>`;
  }

  function row(r) {
    const open = panels.has(r.key);
    const acts = r.actions.map((a) => a.open
      ? `<button class="small secondary" data-open="${r.id}">${esc(a.label)}</button>`
      : `<button class="small" data-post="${esc(a.post)}">${esc(a.label)}</button>`).join(" ");
    return `<div class="brow g-${r.group}${open ? " open" : ""}" id="${r.key}" data-key="${r.key}" data-id="${r.id ?? ""}">
      <div class="brow-grid"${r.id ? ` data-toggle="${r.id}"` : ""}>
        <div class="btitle"><b>${esc(r.title)}</b>
          <small class="muted">${esc([r.sub, r.category, r.folder ? "папка" : ""].filter(Boolean).join(" · "))}</small></div>
        ${r.cells.map(cell).join("")}
      </div>
      <div class="bline"><span class="bicon">${r.icon}</span><span class="btext">${esc(r.line)}</span>${acts ? `<span class="bacts">${acts}</span>` : ""}</div>
      <div class="bpanel-slot"></div>
    </div>`;
  }

  function render(b) {
    // detach open panels so re-rendering doesn't destroy them (and what was typed in them)
    root.querySelectorAll(".brow.open .bpanel").forEach((p) => panels.set(p.closest(".brow").dataset.key, p));
    const head = `<div class="brow-grid bhead"><div class="btitle muted">${b.dry_run ? "dry-run" : ""}</div>${b.stages.map((s, i) =>
      `<div class="bstage">${esc(s)}${b.counts[i] ? ` <span class="pill">${b.counts[i]}</span>` : ""}</div>`).join("")}</div>`;
    const groups = b.groups.filter((g) => g.rows.length).map((g) => {
      const collapsed = g.collapsed && store.get("board-open-" + g.key) !== "1";
      return `<section class="bgroup" data-group="${g.key}">
        <h2 class="bgroup-h" data-collapse="${g.collapsed ? g.key : ""}">${esc(g.label)} <span class="muted">(${g.rows.length})</span>
          ${g.collapsed ? `<span class="muted small">${collapsed ? "▸ показать" : "▾ скрыть"}</span>` : ""}</h2>
        ${collapsed ? "" : g.rows.map(row).join("")}
      </section>`;
    }).join("");
    root.innerHTML = head + (groups || `<p class="muted">Пока пусто. Задачи появятся, когда в выбранных категориях qBittorrent начнутся загрузки.</p>`);
    panels.forEach((p, key) => {
      const host = document.getElementById(key);
      if (host) { host.querySelector(".bpanel-slot").replaceWith(p); host.classList.add("open"); }
      else panels.delete(key);
    });
  }

  async function refresh() {
    try {
      const r = await fetch("/api/board", { cache: "no-store" });
      if (r.ok) render(await r.json());
    } catch (e) { /* server restarting: try again next tick */ }
  }

  async function togglePanel(id) {
    const host = document.getElementById("item-" + id);
    if (!host) return;
    const key = host.dataset.key;
    if (panels.has(key)) {
      panels.delete(key);
      host.classList.remove("open");
      const p = host.querySelector(".bpanel");
      if (p) p.replaceWith(Object.assign(document.createElement("div"), { className: "bpanel-slot" }));
      return;
    }
    const p = Object.assign(document.createElement("div"), { className: "bpanel", innerHTML: "<p class='muted'>загружаю…</p>" });
    panels.set(key, p);
    host.querySelector(".bpanel-slot")?.replaceWith(p);
    host.classList.add("open");
    try {
      const r = await fetch(`/items/${id}/panel`, { cache: "no-store" });
      p.innerHTML = await r.text();
    } catch (e) { p.innerHTML = "<p class='bad'>Не удалось загрузить</p>"; }
  }

  root.addEventListener("click", async (ev) => {
    const post = ev.target.closest("[data-post]");
    if (post) {
      ev.stopPropagation();
      post.disabled = true;
      await fetch(post.dataset.post, { method: "POST" });
      return refresh();
    }
    const open = ev.target.closest("[data-open]");
    if (open) { ev.stopPropagation(); return togglePanel(open.dataset.open); }
    const col = ev.target.closest("[data-collapse]");
    if (col && col.dataset.collapse) {
      const k = "board-open-" + col.dataset.collapse;
      store.set(k, store.get(k) === "1" ? "0" : "1");
      return refresh();
    }
    if (ev.target.closest(".bpanel")) return;            // clicks inside the details
    const t = ev.target.closest("[data-toggle]");
    if (t) togglePanel(t.dataset.toggle);
  });

  render(JSON.parse(document.getElementById("board-data").textContent));
  const m = location.hash.match(/^#item-(\d+)$/);
  if (m) {
    const host = document.getElementById("item-" + m[1]);
    if (!host) { store.set("board-open-done", "1"); refresh().then(() => togglePanel(m[1])); }
    else { togglePanel(m[1]); host.scrollIntoView({ block: "start" }); }
  }
  setInterval(() => { if (!document.hidden) refresh(); }, 5000);
})();
