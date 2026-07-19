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
    let message;
    try {
      const detail = (await res.json()).detail;
      // FastAPI/Pydantic validation failures (422) return `detail` as an
      // array of {msg, loc, ...} objects, not a string -- new Error(array)
      // stringifies to "[object Object]", which told the user nothing.
      // Every other error path here (HTTPException(...)) returns a plain
      // string, so only the array case needs special handling.
      message = Array.isArray(detail)
        ? detail.map((d) => (d && d.msg) || JSON.stringify(d)).join("; ")
        : detail;
    } catch { message = res.statusText; }
    throw new Error(message || `HTTP ${res.status}`);
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
  renderBanners();
  return data;
}

// Monotonic per-call token: if group B is clicked before group A's fetch
// resolves, A's response must not clobber B's once B has already landed (or
// flicker the loading dim off while B is still in flight). Mirrors the
// generation-counter pattern duplicates_web.py's Session already uses for
// the same class of staleness problem (see the ?g= cache-buster on thumb/
// full-res URLs).
let loadGroupToken = 0;

async function loadGroup(i) {
  if (i < 0 || i >= state.groups.length) return;
  const token = ++loadGroupToken;
  const detailEl = document.getElementById("detail");
  detailEl.classList.add("is-loading");
  try {
    const data = await api(`/api/group/${i}`);
    if (token !== loadGroupToken) return; // superseded by a newer loadGroup call
    state.activeIndex = i;
    state.detail = data;
    renderSidebar();
    renderDetail();
  } catch (e) {
    if (token === loadGroupToken) showToast(`Failed to load group: ${e.message}`, true);
  } finally {
    if (token === loadGroupToken) detailEl.classList.remove("is-loading");
  }
}

// Winning index/indices for one row. Values are the exact strings the
// server already formatted (METRIC_ROWS' lambdas) -- always a plain decimal
// or "n/a" in this table, so parseFloat is enough and "n/a" naturally drops
// out as NaN rather than needing special-casing. Requires at least 2 real
// values: BRISQUE/NIQE can be "n/a" for some files and a real number for
// others (per-image analysis failure, not just the package being missing --
// see compare_image_quality.py), and a single lone value has nothing to have
// "won" against.
function bestIndices(direction, values) {
  if (!direction) return [];
  const nums = values.map((v) => parseFloat(v));
  const finite = nums.filter((n) => !Number.isNaN(n));
  if (finite.length < 2) return [];
  const best = direction > 0 ? Math.max(...finite) : Math.min(...finite);
  return nums.flatMap((n, i) => (n === best ? [i] : []));
}

// Shared by the image-row labels and the metrics-table header -- both need
// to say "this is the file you're keeping" / "this is the algorithm's top
// pick" for column j, so one function decides it instead of two copies that
// can drift apart.
function columnMarker(j, d) {
  const picked = j === d.current_pick;
  const suggested = j === d.suggested_idx;
  let title = "";
  if (picked && suggested) title = "Currently kept (also the suggested keeper -- top scored)";
  else if (picked) title = "Currently kept";
  else if (suggested) title = "Suggested keeper (top scored)";
  return { picked, suggested, title };
}

const STATUS_MARKERS = { pending: "◻", confirmed: "✔", skipped: "—" };
const STATUS_LABELS = { pending: "Pending", confirmed: "Confirmed", skipped: "Skipped" };

function renderSidebar() {
  const ul = document.getElementById("group-list");
  ul.innerHTML = "";
  state.groups.forEach((g, i) => {
    const li = document.createElement("li");
    // Status glyph alone doesn't let a long queue be scanned at a glance --
    // every row reads with identical weight until you read each character.
    // Dimming done rows (confirmed/skipped) lets pending rows -- the ones
    // still needing a decision -- stand out peripherally, without dropping
    // the glyph (color is never the only signal).
    li.className = "group-item"
      + (i === state.activeIndex ? " active" : "")
      + (g.status !== "pending" ? " status-done" : "");

    // Status is a bounded, fixed-vocabulary field (pending/confirmed/skipped)
    // -- a colored chip reads at a glance across a long list the way a bare
    // glyph character doesn't; the glyph is kept inside it so shape (not
    // just color) still carries the signal for colorblind-safety.
    const chip = document.createElement("span");
    chip.className = `status-chip status-${g.status}`;
    chip.textContent = STATUS_MARKERS[g.status];
    chip.title = STATUS_LABELS[g.status];

    const label = document.createElement("span");
    label.className = "group-item-text";
    const pick = g.status === "confirmed" ? ` → [${g.current_pick + 1}]` : "";
    label.textContent = `Group ${i + 1} (${g.file_count} files)${pick}`;

    li.appendChild(chip);
    li.appendChild(label);

    // Kept a low-key colored glyph rather than a filled badge deliberately:
    // close calls are common (most groups in a typical scan), not rare, so
    // giving this the same visual weight as a true "needs attention" badge
    // would blow past the ~10% red-chip guideline and stop reading as
    // urgent at all.
    if (g.is_close_call) {
      const warn = document.createElement("span");
      warn.className = "close-call-flag";
      // Spelled out, not a bare glyph: a caution-triangle alone reads as
      // "something's wrong", but a close call isn't an error -- it just
      // means the top two picks scored nearly the same and deserve a look.
      warn.textContent = "⚠ close";
      warn.title = "Close call -- the top two picks scored nearly the same";
      li.appendChild(warn);
    }

    li.title = g.is_close_call
      ? `${label.textContent} -- close call: the top two picks scored nearly the same`
      : label.textContent;
    li.addEventListener("click", () => loadGroup(i));
    ul.appendChild(li);
  });
  updateDocumentTitle();
}

function reviewCounts() {
  const confirmed = state.groups.filter((g) => g.status === "confirmed").length;
  const skipped = state.groups.filter((g) => g.status === "skipped").length;
  const pending = state.groups.length - confirmed - skipped;
  return { confirmed, skipped, pending, total: state.groups.length };
}

function updateDocumentTitle() {
  if (state.groups.length === 0) {
    document.title = "Duplicate image review";
    return;
  }
  const { pending, total } = reviewCounts();
  document.title = pending === 0 ? `✓ Done — Duplicate review` : `(${pending}/${total} left) Duplicate review`;
}

// Persistent (not a toast) home for state that must stay visible without
// the user needing to look at any specific group: scan errors, dry-run
// mode, and overall review progress -- including, once every group has
// been confirmed or skipped, an explicit "you're done, it's safe to close
// this tab" message. Nothing here is a toast because the whole point is
// that it doesn't disappear after a few seconds.
function renderBanners() {
  const errorBanner = document.getElementById("error-banner");
  if (state.status === "error" && state.error) {
    errorBanner.textContent = `⚠ Scan failed: ${state.error}`;
    errorBanner.hidden = false;
  } else {
    errorBanner.hidden = true;
  }

  document.getElementById("dry-run-banner").hidden = !(state.params && state.params.dry_run);

  const progressBanner = document.getElementById("review-progress");
  if (state.groups.length === 0) {
    progressBanner.hidden = true;
    return;
  }
  const { confirmed, skipped, pending, total } = reviewCounts();
  progressBanner.hidden = false;
  progressBanner.classList.toggle("all-done", pending === 0);
  progressBanner.textContent = pending === 0
    ? `✓ All ${total} group(s) reviewed (${confirmed} confirmed, ${skipped} skipped) -- nothing left to do. ` +
      `It's safe to close this tab now, or scan a different directory above.`
    : `${confirmed} confirmed · ${skipped} skipped · ${pending} of ${total} left -- ` +
      `each Confirm/Skip applies immediately, so it's safe to stop and close this tab at any point.`;
}

// Fixed width of the leading label column, shared by #images-row's spacer
// and #metrics-table's first column (see GRID_COLS below) -- must be wide
// enough for the longest metric label ("Eff. res. px equiv (higher
// better)") without wrapping in the common case; the CSS wrap fallback
// (see #metrics-table thead th:first-child) covers anything longer.
const LABEL_COL_WIDTH = "220px";

function gridColumns(n) {
  return `${LABEL_COL_WIDTH} repeat(${n}, minmax(180px, 1fr))`;
}

function renderDetail() {
  const d = state.detail;
  const row = document.getElementById("images-row");
  const table = document.getElementById("metrics-table");
  row.innerHTML = "";
  updateActionButtons();
  if (!d) {
    table.style.gridTemplateColumns = "";
    document.querySelector("#metrics-table thead").innerHTML = "";
    document.querySelector("#metrics-table tbody").innerHTML = "";
    document.getElementById("status-line").textContent = "";
    // A blank images-row reads as "broken", not "nothing to show yet" --
    // spell out which of the two it is. Left blank while state.status is
    // "scanning" since the header's progress bar already owns that message.
    const message = state.groups.length
      ? "Select a group from the sidebar to review it."
      : (state.status === "ready" ? "No potential duplicate groups found in this directory." : "");
    if (message) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = message;
      row.appendChild(empty);
    }
    return;
  }

  // Both grids get the identical column template -- this (not a
  // TUI-style post-layout measurement pass) is what keeps image box [j]
  // lined up with metrics column [j]: see the CSS comment above
  // #images-row for the full explanation.
  const cols = gridColumns(d.paths.length);
  row.style.gridTemplateColumns = cols;
  table.style.gridTemplateColumns = cols;

  const spacer = document.createElement("div");
  spacer.className = "images-spacer";
  row.appendChild(spacer);

  d.paths.forEach((path, j) => {
    const marker = columnMarker(j, d);
    const box = document.createElement("div");
    box.className = "preview-box"
      + (marker.picked ? " picked" : "")
      + (marker.suggested && !marker.picked ? " suggested" : "");
    box.title = path;
    const label = document.createElement("div");
    label.className = "preview-label";
    const tag = marker.picked ? "✔ KEEP  " : (marker.suggested ? "★ suggested  " : "");
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
  const cornerTh = document.createElement("th");
  cornerTh.textContent = "Metric";
  headRow.appendChild(cornerTh);
  d.paths.forEach((_, j) => {
    const th = document.createElement("th");
    const marker = columnMarker(j, d);
    th.textContent = (marker.picked ? "✔ " : "") + `[${j + 1}]` + (marker.suggested ? " ★" : "");
    if (marker.picked) th.classList.add("metric-col-pick");
    if (marker.title) th.title = marker.title;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);

  let prevWasReference = null;
  d.metrics.forEach(({ label, values, direction, kind }, rowIdx) => {
    const isScoreRow = kind === "score";
    const winners = isScoreRow ? [] : bestIndices(direction, values);
    const isReference = kind === "reference";

    const tr = document.createElement("tr");
    if (rowIdx % 2 === 1) tr.classList.add("metric-row-alt");
    if (isReference) tr.classList.add("metric-row-reference");
    else if (prevWasReference) tr.classList.add("metric-row-first-scored");
    if (isScoreRow) tr.classList.add("metric-row-score");
    prevWasReference = isReference;

    const th = document.createElement("th");
    th.textContent = label;
    tr.appendChild(th);

    values.forEach((v, j) => {
      const td = document.createElement("td");
      if (j === d.current_pick) td.classList.add("metric-col-pick");
      if (isScoreRow) {
        const frac = Math.max(0, Math.min(1, parseFloat(v) || 0));
        td.classList.add("metric-score-cell");
        const track = document.createElement("div");
        track.className = "score-bar-track";
        const fill = document.createElement("div");
        fill.className = "score-bar-fill";
        fill.style.width = `${frac * 100}%`;
        track.appendChild(fill);
        const value = document.createElement("span");
        value.className = "score-value";
        value.textContent = v;
        td.appendChild(track);
        td.appendChild(value);
      } else {
        td.textContent = v;
        if (winners.includes(j)) {
          td.classList.add("metric-best");
          td.title = "Best value in this row";
        } else if (v === "n/a") {
          td.title = "Optional dependency not installed -- see Help for details";
        }
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  document.getElementById("status-line").textContent = statusText(d);
}

function updateActionButtons() {
  const hasGroup = !!state.detail;
  document.getElementById("btn-confirm").disabled = !hasGroup;
  document.getElementById("btn-skip").disabled = !hasGroup;
  document.getElementById("btn-open").disabled = !hasGroup;
}

// Per-group action text only -- overall progress (confirmed/skipped/
// pending counts, dry-run mode, "all done") lives in the persistent
// #review-progress/#dry-run-banner banners (renderBanners) instead, so it
// stays visible even when no group is selected rather than being buried
// in this per-group line.
function statusText(d) {
  const nRemoved = d.paths.length - 1;
  const plural = nRemoved !== 1 ? "s" : "";
  let action = `keep [${d.current_pick + 1}] ${d.paths[d.current_pick]}`;
  if (nRemoved > 0) action += `, move ${nRemoved} other file${plural}`;
  const pickIsSuggested = d.current_pick === d.suggested_idx;
  if (!pickIsSuggested) {
    action += `  ·  ★ suggested [${d.suggested_idx + 1}] ${d.paths[d.suggested_idx]}`;
  }

  if (d.status === "confirmed") {
    return pickIsSuggested
      ? `confirmed → [${d.current_pick + 1}]`
      : `confirmed → [${d.current_pick + 1}]. Pick changed since confirming -- press Confirm again to apply it.`;
  }
  if (d.status === "skipped") {
    return `skipped (was: ${action})`;
  }
  return action;
}

function updateScanStatusText() {
  // The error case is covered by the more prominent, persistent
  // #error-banner (renderBanners) instead of duplicating the message here.
  const el = document.getElementById("scan-status");
  if (state.status === "scanning") {
    el.textContent = "scanning…";
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
  renderBanners();
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
  // Only threshold/recursive/dest/dry-run are tucked behind "Options" --
  // directory + Scan are the frequent, high-importance action and stay
  // always visible (see the scan-form-row split in index.html). But an
  // already-non-default setting (e.g. dry-run left on from a previous scan)
  // must never be silently hidden -- that's a trap, not decluttering -- so
  // the panel opens itself whenever reality disagrees with the defaults.
  setOptionsExpanded(!scanOptionsAreDefault());
}

function scanOptionsAreDefault() {
  const threshold = document.getElementById("f-threshold");
  return threshold.value === threshold.defaultValue
    && !document.getElementById("f-recursive").checked
    && !document.getElementById("f-dest").value
    && !document.getElementById("f-dry-run").checked;
}

function setOptionsExpanded(expanded) {
  const panel = document.getElementById("scan-options");
  const toggle = document.getElementById("options-toggle");
  panel.hidden = !expanded;
  toggle.setAttribute("aria-expanded", String(expanded));
  toggle.textContent = expanded ? "Options ▴" : "Options ▾";
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
    <h3>Sidebar status</h3>
    <p>◻ pending &middot; ✔ confirmed &middot; — skipped &middot; ⚠ close call (top two picks scored nearly the same)</p>
    <h3>Keyboard shortcuts</h3>
    <ul>
      <li>&larr; / &rarr; -- move pick</li>
      <li>1-9 -- pick a specific file</li>
      <li>Enter / c -- confirm keep</li>
      <li>Delete / s -- skip group</li>
      <li>o -- open full-res in a new tab</li>
      <li>? / F1 -- this help</li>
    </ul>
    <h3>Stopping and finishing</h3>
    <p>Confirm and Skip apply immediately -- there's no separate "save" step
    and nothing is left half-done. That means it's <strong>safe to close this
    tab at any point</strong>, reviewed or not; your progress is exactly what
    you see on screen. The server itself keeps running (so you can reopen
    this URL and pick up where you left off) until it's stopped from the
    terminal it was started in.</p>`;
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

// #images-row and #metrics-table are two separate scroll containers with
// identical column widths (see gridColumns) -- mirror scrollLeft between
// them 1:1 so a wide group (many files) stays column-aligned while
// scrolled, not just at rest. #metrics-table's own scrollbar is hidden via
// CSS since this makes it redundant.
function syncScroll(a, b) {
  let syncing = false;
  const mirror = (from, to) => {
    if (syncing) return;
    syncing = true;
    to.scrollLeft = from.scrollLeft;
    syncing = false;
  };
  a.addEventListener("scroll", () => mirror(a, b));
  b.addEventListener("scroll", () => mirror(b, a));
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
  document.getElementById("options-toggle").addEventListener("click", () => {
    setOptionsExpanded(document.getElementById("scan-options").hidden);
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
  attachKeyboardHandler();
  attachButtonHandlers();
  syncScroll(document.getElementById("images-row"), document.getElementById("metrics-table"));
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
