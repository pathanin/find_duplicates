// Vanilla JS, no build step -- see PLAN.md's "Frontend" decision. Mirrors
// find_duplicates.py's DuplicateReviewApp: same status vocabulary, same
// keyboard shortcuts (with layout-independent aliases via
// KeyboardEvent.code -- see attachKeyboardHandler), same confirm/skip
// semantics via the same server-side primitives the TUI uses.

const state = {
  status: "idle",
  error: null,
  generation: 0,
  params: null,
  groups: [],       // summaries: {index, status, file_count, current_pick, suggested_idx, is_close_call}
  activeIndex: -1,
  detail: null,     // full detail of the active group: {index, status, current_pick, suggested_idx, is_close_call, paths, metrics}
  eventSource: null,
};

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
  });
  if (!res.ok) {
    let detail;
    try { detail = (await res.json()).detail; } catch { detail = res.statusText; }
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.status === 204 ? null : res.json();
}

function showToast(message, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = isError ? "error" : "";
  el.hidden = false;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => { el.hidden = true; }, 4000);
}

// ---------------------------------------------------------------------------
// State loading + rendering
// ---------------------------------------------------------------------------

async function refreshState() {
  const data = await api("/api/state");
  state.status = data.status;
  state.error = data.error;
  state.generation = data.generation;
  state.params = data.params;
  state.groups = data.groups;
  renderSidebar();
  updateScanStatusText();
  return data;
}

async function loadGroup(i) {
  if (i < 0 || i >= state.groups.length) return;
  const data = await api(`/api/group/${i}`);
  state.activeIndex = i;
  state.detail = data;
  renderSidebar();
  renderDetail();
}

function renderSidebar() {
  const ul = document.getElementById("group-list");
  ul.innerHTML = "";
  const MARKERS = { pending: "◻", confirmed: "✔", skipped: "—" };
  state.groups.forEach((g, i) => {
    const li = document.createElement("li");
    li.className = "group-item" + (i === state.activeIndex ? " active" : "");
    const close = g.is_close_call ? " ⚠" : "";
    const pick = g.status === "confirmed" ? ` → [${g.current_pick + 1}]` : "";
    li.textContent = `${MARKERS[g.status]} Group ${i + 1} (${g.file_count} files)${close}${pick}`;
    li.addEventListener("click", () => loadGroup(i));
    ul.appendChild(li);
  });
}

function renderDetail() {
  const d = state.detail;
  const row = document.getElementById("images-row");
  row.innerHTML = "";
  if (!d) {
    document.getElementById("status-line").textContent = state.groups.length
      ? "Select a group from the sidebar."
      : (state.status === "ready" ? "No potential duplicate groups found." : "");
    document.querySelector("#metrics-table thead").innerHTML = "";
    document.querySelector("#metrics-table tbody").innerHTML = "";
    return;
  }

  d.paths.forEach((path, j) => {
    const box = document.createElement("div");
    box.className = "preview-box" + (j === d.current_pick ? " picked" : "");
    const label = document.createElement("div");
    label.className = "preview-label";
    let tag = "";
    if (j === d.current_pick) tag = "✔ KEEP  ";
    else if (j === d.suggested_idx) tag = "★ suggested  ";
    label.textContent = `${tag}[${j + 1}] ${path}`;
    const img = document.createElement("img");
    img.src = `/api/thumb/${d.index}/${j}?g=${state.generation}`;
    img.alt = path;
    img.title = "Click to pick this file";
    img.addEventListener("click", () => pick(j));
    box.appendChild(label);
    box.appendChild(img);
    box.addEventListener("click", (e) => { if (e.target !== img) pick(j); });
    row.appendChild(box);
  });

  const thead = document.querySelector("#metrics-table thead");
  const tbody = document.querySelector("#metrics-table tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";
  const headRow = document.createElement("tr");
  headRow.appendChild(document.createElement("th"));
  d.paths.forEach((_, j) => {
    const th = document.createElement("th");
    th.textContent = `[${j + 1}]` + (j === d.suggested_idx ? " ★" : "");
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  d.metrics.forEach(({ label, values }) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = label;
    tr.appendChild(th);
    values.forEach((v) => {
      const td = document.createElement("td");
      td.textContent = v;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  document.getElementById("status-line").textContent = statusText(d);
}

function statusText(d) {
  const confirmed = state.groups.filter((g) => g.status === "confirmed").length;
  const skipped = state.groups.filter((g) => g.status === "skipped").length;
  const pending = state.groups.length - confirmed - skipped;
  const mode = state.params && state.params.dry_run ? "  [DRY RUN]" : "";
  const header = `Groups: ${state.groups.length}  confirmed=${confirmed}  skipped=${skipped}  pending=${pending}${mode}`;

  const nRemoved = d.paths.length - 1;
  const plural = nRemoved !== 1 ? "s" : "";
  let action = `keep [${d.current_pick + 1}] ${d.paths[d.current_pick]}`;
  if (nRemoved > 0) action += `, move ${nRemoved} other file${plural}`;
  if (d.current_pick !== d.suggested_idx) {
    action += `  ·  ★ suggested [${d.suggested_idx + 1}] ${d.paths[d.suggested_idx]}`;
  }

  let line;
  if (d.status === "confirmed") {
    line = `confirmed → [${d.current_pick + 1}]. ${action !== `keep [${d.current_pick + 1}] ${d.paths[d.current_pick]}` ? "Press Confirm again to apply the new pick." : ""}`;
  } else if (d.status === "skipped") {
    line = `skipped (was: ${action})`;
  } else {
    line = action;
  }
  return `${header}\n${line}`;
}

function updateScanStatusText() {
  const el = document.getElementById("scan-status");
  if (state.status === "scanning") {
    el.textContent = "scanning…";
  } else if (state.status === "error") {
    el.textContent = `error: ${state.error}`;
  } else if (state.status === "ready") {
    el.textContent = `${state.groups.length} group(s) found`;
  } else {
    el.textContent = "";
  }
  document.getElementById("scan-btn").disabled = state.status === "scanning";
}

// ---------------------------------------------------------------------------
// Actions: pick / confirm / skip -- mirror DuplicateReviewApp.action_pick /
// action_confirm / action_skip, via the same server-side primitives.
// ---------------------------------------------------------------------------

function applyGroupPatch(i, data) {
  state.groups[i] = {
    index: i,
    status: data.status,
    file_count: data.paths.length,
    current_pick: data.current_pick,
    suggested_idx: data.suggested_idx,
    is_close_call: data.is_close_call,
  };
  if (i === state.activeIndex) state.detail = data;
  renderSidebar();
  renderDetail();
}

async function pick(j) {
  const i = state.activeIndex;
  if (i < 0) return;
  try {
    const data = await api(`/api/group/${i}/pick`, { method: "POST", body: JSON.stringify({ idx: j }) });
    applyGroupPatch(i, data);
  } catch (e) {
    showToast(`Pick failed: ${e.message}`, true);
  }
}

async function pickRelative(delta) {
  const d = state.detail;
  if (!d) return;
  const n = d.paths.length;
  await pick(((d.current_pick + delta) % n + n) % n);
}

async function confirmGroup() {
  const i = state.activeIndex;
  if (i < 0) return;
  try {
    const data = await api(`/api/group/${i}/confirm`, { method: "POST" });
    applyGroupPatch(i, data);
    advance();
  } catch (e) {
    showToast(`Confirm failed: ${e.message}`, true);
  }
}

async function skipGroup() {
  const i = state.activeIndex;
  if (i < 0) return;
  try {
    const data = await api(`/api/group/${i}/skip`, { method: "POST" });
    applyGroupPatch(i, data);
    if (data.status === "skipped") advance();
  } catch (e) {
    showToast(`Skip failed: ${e.message}`, true);
  }
}

function advance() {
  const n = state.groups.length;
  if (n === 0) return;
  for (let off = 1; off <= n; off++) {
    const j = (state.activeIndex + off) % n;
    if (state.groups[j].status === "pending") {
      loadGroup(j);
      return;
    }
  }
  showToast("All groups reviewed.");
}

function openFullRes() {
  const d = state.detail;
  if (!d) return;
  window.open(`/api/full/${d.index}/${d.current_pick}?g=${state.generation}`, "_blank");
}

// ---------------------------------------------------------------------------
// Scan control panel + SSE progress
// ---------------------------------------------------------------------------

function populateFormFromParams() {
  if (!state.params) return;
  document.getElementById("f-directory").value = state.params.directory;
  document.getElementById("f-threshold").value = state.params.threshold;
  document.getElementById("f-recursive").checked = state.params.recursive;
  document.getElementById("f-dest").value = state.params.dest || "";
  document.getElementById("f-dry-run").checked = state.params.dry_run;
}

function connectProgress() {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource("/api/progress");
  state.eventSource = es;
  document.getElementById("progress-bar-wrap").hidden = false;

  es.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    const fill = document.getElementById("progress-fill");
    const label = document.getElementById("progress-label");
    const pct = data.total > 0 ? Math.round((100 * data.done) / data.total) : 0;
    fill.style.width = `${pct}%`;
    label.textContent = data.label ? `${data.label}: ${data.done}/${data.total}` : "";

    if (data.status !== "scanning") {
      // Must close explicitly: a browser EventSource treats a closed
      // stream as an error and auto-reconnects a few seconds later, which
      // would just re-open this same endpoint forever once a scan is done.
      es.close();
      state.eventSource = null;
      document.getElementById("progress-bar-wrap").hidden = true;
      refreshState().then(() => {
        state.detail = null;
        state.activeIndex = -1;
        if (state.groups.length > 0) {
          const firstPending = state.groups.findIndex((g) => g.status === "pending");
          loadGroup(firstPending >= 0 ? firstPending : 0);
        } else {
          renderDetail();
        }
      });
    }
  };
}

document.getElementById("scan-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = {
    directory: document.getElementById("f-directory").value,
    threshold: parseInt(document.getElementById("f-threshold").value, 10),
    recursive: document.getElementById("f-recursive").checked,
    dest: document.getElementById("f-dest").value || null,
    dry_run: document.getElementById("f-dry-run").checked,
  };
  try {
    await api("/api/scan", { method: "POST", body: JSON.stringify(body) });
    state.status = "scanning";
    updateScanStatusText();
    connectProgress();
  } catch (err) {
    showToast(`Scan failed to start: ${err.message}`, true);
  }
});

// ---------------------------------------------------------------------------
// Help modal, rendered from /api/metrics-info so it can't drift from what's
// actually scored -- same principle as find_duplicates.py's _help_body.
// ---------------------------------------------------------------------------

async function showHelp() {
  const info = await api("/api/metrics-info");
  const entries = Object.entries(info.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  let html = `
    <h2>Quality score</h2>
    <p>A weighted composite of the metrics below, normalized 0-1 within this group only
    (min-max against the other files here -- not comparable across different photos).
    It's a hand-tuned heuristic, not a lab measurement: treat it as a strong hint,
    not a verdict, especially on a close call (⚠).</p>
    <p>Dimensions and file size are shown for reference only and do NOT factor into the score.</p>
    <h3>Weighted metrics, sorted by influence</h3>
    <ul>`;
  for (const [name, weight] of entries) {
    const direction = weight > 0 ? "higher better" : "lower better";
    html += `<li><strong>${Math.abs(weight).toFixed(2)}</strong> ${name} (${direction})<br>
      <span class="dim">${info.descriptions[name]}</span></li>`;
  }
  html += `</ul>
    <h3>Keyboard shortcuts</h3>
    <ul>
      <li>&larr; / &rarr; -- move pick</li>
      <li>1-9 -- pick a specific file</li>
      <li>Enter / c -- confirm keep</li>
      <li>Delete / s -- skip group</li>
      <li>o -- open full-res in a new tab</li>
      <li>? / F1 -- this help</li>
    </ul>`;
  document.getElementById("help-body").innerHTML = html;
  document.getElementById("help-modal").hidden = false;
}

function closeHelp() {
  document.getElementById("help-modal").hidden = true;
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts, mirroring DuplicateReviewApp.BINDINGS. Uses
// KeyboardEvent.code (physical key position) rather than .key for letter
// bindings and their control-key aliases -- an alternate keyboard layout
// remaps .key to a different character before the browser even sees it,
// the exact concern the TUI's control-key aliases exist to solve.
// ---------------------------------------------------------------------------

function attachKeyboardHandler() {
  document.addEventListener("keydown", (e) => {
    const help = document.getElementById("help-modal");
    if (!help.hidden) {
      if (e.code === "Escape" || e.key === "?" || e.code === "KeyQ") { closeHelp(); e.preventDefault(); }
      return;
    }
    const active = document.activeElement;
    const typing = active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA");
    if (typing) return;

    if (e.code === "ArrowLeft") { pickRelative(-1); e.preventDefault(); }
    else if (e.code === "ArrowRight") { pickRelative(1); e.preventDefault(); }
    else if (e.code === "Enter" || e.code === "KeyC") { confirmGroup(); e.preventDefault(); }
    else if (e.code === "Delete" || e.code === "Backspace" || e.code === "KeyS") { skipGroup(); e.preventDefault(); }
    else if (e.code === "KeyO") { openFullRes(); e.preventDefault(); }
    else if (e.code === "F1" || e.key === "?") { showHelp(); e.preventDefault(); }
    else if (e.code.startsWith("Digit")) {
      const n = parseInt(e.code.slice(5), 10);
      if (n >= 1 && n <= 9 && state.detail && n <= state.detail.paths.length) { pick(n - 1); e.preventDefault(); }
    }
  });
}

function attachButtonHandlers() {
  document.getElementById("btn-confirm").addEventListener("click", confirmGroup);
  document.getElementById("btn-skip").addEventListener("click", skipGroup);
  document.getElementById("btn-open").addEventListener("click", openFullRes);
  document.getElementById("btn-help").addEventListener("click", showHelp);
  document.getElementById("help-close").addEventListener("click", closeHelp);
  document.getElementById("help-modal").addEventListener("click", (e) => {
    if (e.target.id === "help-modal") closeHelp();
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  attachKeyboardHandler();
  attachButtonHandlers();
  await refreshState();
  populateFormFromParams();
  if (state.status === "scanning") {
    connectProgress();
  } else if (state.groups.length > 0) {
    const firstPending = state.groups.findIndex((g) => g.status === "pending");
    await loadGroup(firstPending >= 0 ? firstPending : 0);
  } else {
    renderDetail();
  }
}

init();
