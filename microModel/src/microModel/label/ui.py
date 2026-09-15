"""The label app's embedded single-page UI (vanilla JS, no build step).

Layout (light theme, same approach as reduction_vis):

  top bar      queue + source + live stats + Undo / display controls
  left         Labels box (click = Collect queue, drag = reorder) +
               a Selected-label box with the auto-annotate actions +
               help/shortcuts in a collapsible box
  toolbar      target-label chips + Apply + / Apply − / Remove / selection
               tools + a context row (Collect sort & Shuffle / Manage scope
               & sort & Auto-only)
  grid         one page of thumbnails; click = select, double-click = zoom
  zoom         full-size view where labels are TOGGLED DIRECTLY (chips or
               keys 1-9 cycle undecided -> positive -> explicit negative),
               ←/→ walk the current queue, and suspicious cells show the
               contradicting evidence cell side by side

Workflow (Collect -> Auto -> Manage): pick a label, harvest positives in
Collect, the auto pass annotates confident undecided cells once the label
crosses its positive threshold, Manage verifies everything (uncertain
first / suspicious first) and takes explicit negatives for the ML
recommender. Everything writes through to label.db immediately; Ctrl+Z
undoes the last action; label_export.csv is re-written after every write.
"""

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>microModel — Cell Labeling</title>
<style>
:root { --bg:#f7f7f8; --line:#ddd; --acc:#4363d8; --warn:#e58a00; --ok:#0a8f3c; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:"Segoe UI",system-ui,sans-serif; background:var(--bg); height:100vh; display:flex; flex-direction:column; font-size:13px; color:#222; }
#topbar { background:#fff; border-bottom:1px solid var(--line); padding:5px 10px; display:flex; flex-direction:column; gap:4px; }
.trow { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
#topbar .brand { font-weight:700; margin-right:4px; }
#topbar select, #topbar input { font-size:12px; padding:1px 3px; }
#topbar label { font-size:12px; white-space:nowrap; color:#555; }
#mode { max-width:200px; } #source { max-width:150px; }
#stats { color:#555; font-size:12px; white-space:nowrap; }
#stats b { color:#222; }
button { font-size:13px; padding:3px 10px; border:1px solid #bbb; background:#fff; border-radius:4px; cursor:pointer; }
button:hover { background:#eee; }
button.primary { background:var(--acc); color:#fff; border-color:var(--acc); }
button.neg { color:#b00; }
button:disabled { opacity:.45; cursor:default; }
.sep { width:1px; height:18px; background:var(--line); flex-shrink:0; }
#main { flex:1; display:flex; min-height:0; }
#sidebar { width:250px; background:#fff; border-right:1px solid var(--line); padding:10px; overflow-y:auto; flex-shrink:0; display:flex; flex-direction:column; gap:10px; }
.side-box { border:1px solid var(--line); border-radius:6px; padding:8px; }
.side-box h3 { font-size:11px; color:#666; margin-bottom:6px; text-transform:uppercase; letter-spacing:.4px; }
.lbl-row { display:flex; align-items:center; gap:6px; padding:4px 6px; border-radius:4px; cursor:pointer; margin-bottom:2px; }
.lbl-row:hover { background:#f0f0f2; }
.lbl-row.active { outline:2px solid var(--acc); }
.lbl-row.dragging { opacity:.4; }
.lbl-row.dragover { box-shadow:0 -2px 0 var(--acc); }
.lbl-row .dot { width:12px; height:12px; border-radius:3px; flex-shrink:0; }
.lbl-row .cnt { margin-left:auto; color:#888; font-size:11px; white-space:nowrap; }
.lbl-row .del { display:none; color:#c33; cursor:pointer; font-weight:700; padding:0 4px; border-radius:3px; }
.lbl-row:hover .del { display:inline-block; }
.lbl-row .del.armed { display:inline-block; background:#c33; color:#fff; font-size:11px; font-weight:600; padding:1px 6px; }
#sel-box .name { font-weight:600; display:flex; align-items:center; gap:6px; margin-bottom:4px; }
#sel-box .counts { color:#666; font-size:12px; margin-bottom:4px; }
.bar { height:8px; background:#eee; border-radius:4px; overflow:hidden; margin:4px 0; }
.bar i { display:block; height:100%; background:var(--warn); }
.bar i.full { background:var(--ok); }
#sel-box .autoline { font-size:11px; color:#777; margin-bottom:6px; }
#sel-box .btns { display:flex; gap:6px; flex-wrap:wrap; }
#help { font-size:11px; color:#555; line-height:1.7; }
#help summary { cursor:pointer; font-size:12px; font-weight:600; color:#444; }
#help b { color:#333; }
#content { flex:1; display:flex; flex-direction:column; min-width:0; }
#banner { padding:6px 12px; background:#fffbe8; border-bottom:1px solid #eee; display:none; align-items:center; gap:8px; }
#toolbar { display:flex; flex-direction:column; gap:4px; padding:6px 10px; background:#fff; border-bottom:1px solid var(--line); }
.trow2 { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
#targets { display:flex; gap:4px; flex-wrap:wrap; align-items:center; }
.tgt { display:inline-flex; align-items:center; gap:5px; border:1.5px solid #ccc; border-radius:12px; padding:2px 10px; font-size:12px; cursor:pointer; user-select:none; background:#fff; }
.tgt .dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.tgt.on { border-color:var(--c); background:var(--c); color:#fff; font-weight:600; }
.tgt.on .dot { background:#fff !important; }
#sel-count { font-size:12px; color:#666; }
#qtools { display:none; align-items:center; gap:6px; flex-wrap:wrap; font-size:12px; color:#555; }
#qtools select { font-size:12px; padding:1px 3px; }
#grid { flex:1; overflow:auto; padding:10px; display:flex; flex-wrap:wrap; gap:8px; align-content:flex-start; }
#grid .cellbox { background:#fff; border:2px solid var(--line); border-radius:4px; padding:4px; cursor:pointer; position:relative; flex-shrink:0; }
/* Selected: 3px accent ring + light fill + a check badge in the free
   bottom-left corner (top corners hold the score/suspicion badges,
   bottom-right the label dots) — unmistakable at thumbnail size. */
#grid .cellbox.sel { border-color:var(--acc); box-shadow:0 0 0 3px var(--acc); background:#eef2fd; }
#grid .cellbox.sel::after {
  content:'✓';
  position:absolute; bottom:4px; left:4px;
  width:18px; height:18px; line-height:18px; text-align:center;
  background:var(--acc); color:#fff; border-radius:50%;
  font-size:12px; font-weight:700;
}
#grid .cellbox .automark { position:absolute; bottom:2px; left:26px; font-size:8px; font-weight:700; color:#4a90d9; }
#grid .cellbox img { object-fit:contain; background:#fff; display:block; }
#grid .cellbox .sugb { position:absolute; top:2px; left:4px; font-size:9px; background:rgba(6,140,60,.85); color:#fff; padding:0 4px; border-radius:3px; }
#grid .cellbox .cert { position:absolute; top:2px; right:4px; font-size:9px; background:rgba(224,120,0,.85); color:#fff; padding:0 4px; border-radius:3px; }
#grid .cellbox .susb { position:absolute; top:2px; right:4px; font-size:9px; background:rgba(200,30,30,.88); color:#fff; padding:0 4px; border-radius:3px; }
#grid .cellbox .dots { position:absolute; bottom:8px; right:6px; display:flex; gap:2px; }
#grid .cellbox .dot { width:9px; height:9px; border-radius:50%; }
#grid .cellbox .dot.neg { outline:1.5px solid #999; opacity:.65; }
#pager { display:flex; justify-content:center; align-items:center; gap:4px; padding:8px 0 4px; flex-wrap:wrap; }
#pager:empty { display:none; }
#pager .pg { min-width:32px; }
#pager .pg.cur { background:var(--acc); color:#fff; border-color:var(--acc); }
#pager .pgap { color:#888; padding:0 2px; }
#pager input { width:52px; font-size:12px; padding:1px 3px; text-align:center; }
#zoom { position:fixed; inset:0; background:rgba(0,0,0,.75); display:none; align-items:center; justify-content:center; z-index:50; cursor:zoom-out; }
#busy { position:fixed; inset:0; background:rgba(255,255,255,.65); display:none; align-items:center; justify-content:center; z-index:99; font-size:15px; color:#333; cursor:wait; }
#label-mode { max-width:150px; }
#zoom-box { background:#fff; border-radius:8px; padding:12px; cursor:default; max-width:92vw; max-height:92vh; display:flex; gap:12px; }
#zoom-left { display:flex; flex-direction:column; min-width:0; }
#zoom-img { max-width:72vmin; max-height:70vmin; background:#000; image-rendering:pixelated; display:block; }
#zoom-ev-wrap { display:none; flex-direction:column; align-items:center; gap:2px; }
#zoom-ev { max-width:34vmin; max-height:34vmin; background:#000; image-rendering:pixelated; display:block; border:2px solid var(--warn); }
#zoom-ev-cap { font-size:11px; color:var(--warn); font-weight:600; text-align:center; }
#zoom-meta { font-size:12px; color:#333; margin-top:6px; word-break:break-all; }
#zoom-meta .zg { color:var(--ok); font-weight:600; }
#zoom-meta .zo { color:var(--warn); font-weight:600; }
#zoom-meta .zr { color:#c22; font-weight:600; }
#zoom-meta .dim { color:#888; }
#zoom-chips { display:flex; gap:5px; flex-wrap:wrap; margin-top:8px; }
.zchip { display:inline-flex; align-items:center; gap:4px; border:1.5px solid #ccc; border-radius:12px; padding:3px 10px; font-size:12px; cursor:pointer; user-select:none; background:#fff; }
.zchip .dot { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
.zchip.pos { border-color:var(--c); background:var(--c); color:#fff; font-weight:600; }
.zchip.pos .dot { background:#fff !important; }
.zchip.neg { border-style:dashed; border-color:var(--c); opacity:.85; }
#zoom-hint { font-size:11px; color:#888; margin-top:6px; }
#toast { position:fixed; bottom:18px; left:50%; transform:translateX(-50%); background:#333; color:#fff; padding:8px 16px; border-radius:6px; display:none; z-index:10; }
</style>
</head>
<body>
<div id="topbar">
  <div class="trow">
    <span class="brand">🔬 micromodel label</span>
    <select id="mode">
      <option value="label_top" title="Collect: undecided cells ranked for the label selected on the left — Score ↓ = confident positives first (cold start = farthest-point spread), Uncertain ↑ = nearest the decision boundary first (active learning). Reach the positive threshold — then the Auto pass fires. Shuffle reshuffles the queue when a page shows nothing like your target.">Collect (selected label)</option>
      <option value="label_all" title="Manage: ALL labeled cells (union of every label). Click a label on the left, then pick the scope — union / with the label / without the label — to verify each class fast without losing labels.">Manage (labeled cells)</option>
      <option value="unlabeled" title="Every cell without any annotation yet (stable order).">Unlabeled</option>
      <option value="all" title="Every cell (filterable by source).">All</option>
    </select>
    <select id="source"><option value="">All sources</option></select>
    <label title="Two fully independent projects share this directory: Multi-label = a cell may hold any number of positives. Single-label = mutually exclusive classes (a new positive clears the cell's other positives, and only the best suggestion is shown). Switching swaps the whole store — labels and decisions of the other mode are untouched, exports are separate files.">
      Mode
      <select id="label-mode">
        <option value="multi">Multi-label</option>
        <option value="single">Single-label</option>
      </select>
    </label>
    <span id="stats"></span>
    <span style="flex:1"></span>
    <button id="btn-undo" title="Undo the last action (Ctrl+Z): a batch apply, a single write, an auto run — each is one revertible step. Keep pressing to undo further back.">Undo</button>
  </div>
  <div class="trow">
    <label title="Thumbnail display size (px). 96-512.">Size
      <input id="cell-px" type="number" min="96" max="512" step="8" style="width:56px">
    </label>
    <label title="Page size: images per page (the batch you review at once). 1-200.">K
      <input id="nb-k" type="number" min="1" max="200" style="width:56px">
    </label>
    <span class="sep"></span>
    <label title="Display contrast: symmetrically narrows the per-channel percentile window (0.1/99.9 at 0 → 10/90 at 100). Display only — features are unaffected.">Contrast
      <input id="disp-con" type="range" min="0" max="100" step="1" style="width:64px;vertical-align:middle">
    </label>
    <label title="Display gamma: 1 = off, > 1 brightens midtones, < 1 darkens. Display only — features are unaffected.">Gamma
      <input id="disp-gamma" type="range" min="20" max="300" step="5" style="width:64px;vertical-align:middle">
    </label>
    <button id="disp-reset" title="Reset contrast and gamma to the default render">Reset</button>
    <span class="sep"></span>
    <label title="Suggestion threshold (0-1): a label is suggested when the cell's nearest positive exemplar is at least this similar and beats the nearest explicit negative. Lower = more suggestions, higher = fewer but safer. Applies to queues loaded afterwards.">Suggest thr
      <input id="thr" type="number" step="0.05" min="0" max="1" style="width:58px">
    </label>
  </div>
</div>
<div id="main">
  <div id="sidebar">
    <div class="side-box">
      <h3>Labels — click = Collect queue</h3>
      <div id="labels"></div>
      <div style="display:flex;gap:4px;margin-top:6px">
        <input id="new-label" placeholder="New label name" style="flex:1;font-size:13px;padding:3px 6px">
        <button id="btn-add" class="primary">Add</button>
      </div>
      <button id="btn-model-labels" style="display:none;margin-top:6px;width:100%">Create labels from model classes</button>
    </div>
    <div class="side-box" id="sel-box" style="display:none">
      <div class="name"><span class="dot" id="sel-dot" style="width:12px;height:12px;border-radius:3px;display:inline-block"></span><span id="sel-name"></span></div>
      <div class="counts" id="sel-counts"></div>
      <div class="bar"><i id="sel-bar" style="width:0%"></i></div>
      <div class="autoline" id="sel-autoline"></div>
      <div class="btns">
        <button id="btn-auto-apply" title="Run the auto-annotate pass for this label now: every undecided cell scoring >= the auto threshold becomes an AUTO positive (blue A). Runs automatically once per label anyway when it crosses the positive threshold.">Auto-apply now</button>
        <button id="btn-auto-clear" title="Remove every AUTO annotation of this label (undo an auto run). Manual annotations are untouched; the next write may re-run the pass.">Remove auto</button>
      </div>
    </div>
    <div class="side-box">
      <details id="help">
        <summary>Workflow &amp; shortcuts</summary>
        <b>1. Collect</b> — click a label on the left; the queue ranks
        undecided cells for it. Select images, <b>Apply +</b> (A) until the
        label reaches the auto threshold. Score ↓ = confident first;
        Uncertain ↑ = informative first; Shuffle = new random order.
        <b>2. Auto</b> — at the threshold the auto pass marks every
        confident undecided cell for every eligible label (blue A), so
        nothing is missed.
        <b>3. Manage</b> — all labeled cells. Scope: with / without the
        selected label / union. Sort by Certainty ↑ (least certain first)
        or Suspicious ⚠ (likely mislabels, with the contradicting cell as
        evidence). Remove wrong ones; <b>Apply −</b> (Shift+A) records
        explicit negatives — they power the ML recommender.
        <b>Shortcuts</b><br>
        click = select · double-click = zoom · 1-9 = toggle target labels<br>
        A = Apply + · Shift+A = Apply − · R = Remove · Ctrl+Z = Undo
        (keep pressing to undo further)<br>
        In zoom: 1-9/Shift+1-9 label the zoomed cell, ←/→ walk the queue,
        click a chip to cycle pos → neg → off · Esc closes<br>
        ←/→ flip page · pager: type page + Enter · drag label rows to
        reorder · hover ✕ to delete (confirm twice)<br>
        Contrast/Gamma/Size change the display only · every write lands in
        the current mode's DB (label_multiple.db / label_single.db) and its
        export CSV immediately<br>
        Mode (top bar): Multi-label = any number of positives per cell;
        Single-label = exclusive classes (a new positive clears the cell's
        other positives, only the best suggestion is shown). The two are
        fully independent projects sharing one directory — switching swaps
        the whole store, nothing mixes
        <div id="legend">
          <span style="color:var(--ok)">■</span> green: suggestion score (top-left)
          · <span style="color:var(--warn)">■</span> orange: certainty
          · <span style="color:#c22">■</span> red: suspicion (mislabel risk)
        </div>
      </details>
    </div>
  </div>
  <div id="content">
    <div id="banner"></div>
    <div id="toolbar">
      <div class="trow2">
        <span id="targets"></span>
        <span style="flex:1"></span>
        <span id="sel-count"></span>
        <button id="btn-apply-pos" class="primary" title="Set every CHECKED target label as POSITIVE on every selected image (A)">Apply +</button>
        <button id="btn-apply-neg" class="neg" title="Set every CHECKED target label as an EXPLICIT NEGATIVE on every selected image (Shift+A). Explicit negatives power the ML recommender — label a few per confusable label.">Apply −</button>
        <button id="btn-remove" title="CLEAR the checked target labels' decisions on the selected images (R). In the Manage view this drops them from the label.">Remove</button>
        <button id="btn-sel-all">Select all</button>
        <button id="btn-sel-none">Clear</button>
      </div>
      <div id="qtools"></div>
    </div>
    <div id="grid"></div>
    <div id="pager"></div>
  </div>
</div>
<div id="zoom"><div id="zoom-box">
  <div id="zoom-left">
    <img id="zoom-img">
    <div id="zoom-meta"></div>
    <div id="zoom-chips"></div>
    <div id="zoom-hint">click a chip: off → positive → negative → off · 1-9 = positive · Shift+1-9 = negative · ←/→ next/previous · Esc close</div>
  </div>
  <div id="zoom-ev-wrap"><img id="zoom-ev"><div id="zoom-ev-cap"></div></div>
</div></div>
<div id="busy"><div>Switching label mode…</div></div>
<div id="toast"></div>
<script>
"use strict";
const $ = s => document.querySelector(s);
const S = { labels:[], labelById:{}, mode:'unlabeled', labelFilter:null, source:'',
            scope:'union', manageSort:'asc', uncSort:false,
            thr:0.75, nbK:100, page:0,
            cellPx:(()=>{ const v = parseInt(localStorage.getItem('label_cellsize'));
              return (v >= 96 && v <= 512) ? v : 100; })(),
            queue:[], queueTotal:0,
            sel:new Set(), targets:new Set(),
            disp:(()=>{ const d = { lo:0.1, hi:99.9, gamma:1.0 };
              try { Object.assign(d, JSON.parse(localStorage.getItem('label_display')||'{}')); } catch(e) {}
              return d; })(),
            dragId:null, zoomIdx:-1,
            hasModel:false, classNames:[], total:0, labeled:0, undecided:0,
            labelMode:'multi' };
const esc = s => (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
// auto_label.min_positives from the server (set by refreshStats).
let AUTO_MIN = 20;
function toast(m) { const t=$('#toast'); t.textContent=m; t.style.display='block'; clearTimeout(t._h); t._h=setTimeout(()=>t.style.display='none',2600); }
async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json().catch(()=>({}));
  if (!r.ok) { toast(j.error || ('HTTP '+r.status)); throw new Error(j.error || r.status); }
  return j;
}
// v = display pipeline version: bump when rendering changes so browsers
// drop their cached renders of the old pipeline (mirrors the server's
// imaging.RENDER_VERSION).
const IMG_V = 3;
// lo/hi = percentile display window, gamma = display gamma. All three ride
// on the URL, so grid thumbnails and the zoom overlay honor them.
const imgURL = (fp, px) => '/api/image?filepath=' + encodeURIComponent(fp) +
  '&max_px=' + px + '&v=' + IMG_V +
  '&lo=' + S.disp.lo + '&hi=' + S.disp.hi + '&gamma=' + S.disp.gamma;

// ---- state / stats ---------------------------------------------------------
async function refreshStats() {
  const j = await api('/api/state');
  S.labels = j.labels; S.hasModel = j.has_model; S.classNames = j.class_names;
  // The config value is only the default — a user-chosen K (stored locally)
  // wins until cleared.
  S.thr = j.threshold; S.nbK = parseInt(localStorage.getItem('label_nbk')) || j.page_size;
  S.total = j.total; S.labeled = j.labeled; S.undecided = j.undecided;
  AUTO_MIN = j.auto_min_positives || 20;
  S.labelById = {}; S.labels.forEach((l,i)=>{ l.idx = i+1; S.labelById[l.label_id] = l; });
  $('#thr').value = S.thr;
  $('#nb-k').value = S.nbK;
  $('#stats').innerHTML = `<b>${S.labeled}</b> / ${S.total} labeled` +
    ` · <b>${S.undecided}</b> undecided`;
  $('#btn-model-labels').style.display = S.classNames.length ? 'block' : 'none';
  const src = $('#source'), keep = src.value;
  src.innerHTML = '<option value="">All sources</option>' +
    j.sources.map(s=>`<option value="${esc(s.path)}">${esc(s.name)}</option>`).join('');
  src.value = keep;
  if (!S.hasModel) ['label_top'].forEach(v=>{
    const o=$('#mode').querySelector(`option[value=${v}]`); if(o){o.disabled=true;}
  });
  S.labelMode = j.label_mode || 'multi';
  $('#label-mode').value = S.labelMode;
  renderLabels(); renderSelBox(); renderTargets();
}
function renderLabels() {
  $('#labels').innerHTML = S.labels.map(l =>
    `<div class="lbl-row ${S.labelFilter===l.label_id?'active':''}" draggable="true" data-lid="${l.label_id}">` +
    `<span class="dot" style="background:${l.color}"></span>` +
    `<span style="color:#aaa;font-size:11px">${l.idx}</span><span>${esc(l.name)}</span>` +
    `<span class="cnt">+${l.n_pos}${l.n_neg ? ' −'+l.n_neg : ''}</span>` +
    `<span class="del" title="Delete label (confirm twice)">✕</span></div>`).join('')
    || '<div style="color:#999">No labels yet — add one below.</div>';
  document.querySelectorAll('.lbl-row').forEach(el => {
    const lid = +el.dataset.lid;
    el.onclick = () => {
      // Manage keeps the clicked label as its REFERENCE (the with/without
      // scopes need one) — never toggles off; other queues toggle as
      // before.
      if (S.mode === 'label_all') S.labelFilter = lid;
      else S.labelFilter = (S.labelFilter === lid) ? null : lid;
      // The clicked label becomes the ONLY checked target — switching
      // labels switches the Apply target with it (more chips can still be
      // checked by hand for a multi-label batch).
      if (S.labelFilter != null) S.targets = new Set([lid]);
      S.shuffle = 0;   // a new label restarts from the queue's own ranking
      renderLabels(); renderSelBox(); renderTargets();
      // Label-consuming queues just reload; any OTHER mode switches to the
      // Manage view (union scope) of all labeled cells.
      if (['label_top','label_all'].includes(S.mode)) {
        loadQueue(0);
      } else if (S.labelFilter) {
        S.mode = 'label_all'; $('#mode').value = 'label_all';
        updateModeTip(); loadQueue(0);
      }
    };
    // ---- drag to reorder ------------------------------------------------
    el.ondragstart = (ev) => { S.dragId = lid; el.classList.add('dragging');
      ev.dataTransfer.effectAllowed = 'move'; };
    el.ondragend = () => { S.dragId = null; el.classList.remove('dragging'); };
    el.ondragover = (ev) => { ev.preventDefault(); if (S.dragId && S.dragId !== lid) el.classList.add('dragover'); };
    el.ondragleave = () => el.classList.remove('dragover');
    el.ondrop = async (ev) => {
      ev.preventDefault(); el.classList.remove('dragover');
      if (!S.dragId || S.dragId === lid) return;
      const from = S.labels.findIndex(l => l.label_id === S.dragId);
      const to = S.labels.findIndex(l => l.label_id === lid);
      if (from < 0 || to < 0) return;
      const [moved] = S.labels.splice(from, 1);
      S.labels.splice(to, 0, moved);
      S.labels.forEach((l, i) => l.idx = i + 1);   // keyboard numbers follow
      renderLabels(); renderTargets();
      await api('/api/labels_reorder', { method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ label_ids: S.labels.map(l => l.label_id) }) });
    };
    // ---- delete with double confirm -------------------------------------
    const del = el.querySelector('.del');
    del.onclick = (ev) => {
      ev.stopPropagation();
      if (del.dataset.armed) { deleteLabel(lid); }
      else {
        del.dataset.armed = '1'; del.textContent = 'Confirm delete'; del.classList.add('armed');
        setTimeout(() => { if (del.isConnected) { del.dataset.armed = '';
          del.textContent = '✕'; del.classList.remove('armed'); } }, 3000);
      }
    };
  });
}
async function deleteLabel(lid) {
  const name = (S.labelById[lid] || {}).name || lid;
  await api('/api/labels_delete', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ label_id: lid }) });
  if (S.labelFilter === lid) S.labelFilter = null;
  S.targets.delete(lid); renderTargets();
  S.queue.forEach(c => delete c.labels[lid]);
  toast(`Label "${name}" deleted (with all of its decisions)`);
  await refreshStats();
  if (['label_top','label_all'].includes(S.mode)) loadQueue();
  else render();
}
// The selected-label box: counts, auto progress and the auto actions —
// every label-scoped control lives in ONE place.
function renderSelBox() {
  const box = $('#sel-box');
  const l = S.labelFilter != null ? S.labelById[S.labelFilter] : null;
  if (!l) { box.style.display = 'none'; return; }
  box.style.display = 'block';
  $('#sel-dot').style.background = l.color;
  $('#sel-name').textContent = l.name;
  $('#sel-counts').textContent = `+${l.n_pos} positives · −${l.n_neg} negatives`;
  const pct = Math.max(0, Math.min(100, l.n_pos / Math.max(1, AUTO_MIN) * 100));
  const bar = $('#sel-bar');
  bar.style.width = pct + '%';
  bar.className = l.n_pos >= AUTO_MIN ? 'full' : '';
  $('#sel-autoline').textContent = l.n_pos >= AUTO_MIN
    ? (l.auto_fired ? `auto pass: done (${AUTO_MIN}+ positives) — Remove auto resets it`
                    : `auto pass: ready — fires on the next positive write (${AUTO_MIN}+ positives)`)
    : `auto pass at ${AUTO_MIN} positives — Collect ${(AUTO_MIN - l.n_pos)} more`;
}

// ---- queue -----------------------------------------------------------------
function queueParams(extra) {
  const p = new URLSearchParams({ mode: S.mode, threshold: S.thr, ...extra });
  if (S.source) p.set('source', S.source);
  if (S.mode === 'label_all') {
    // Manage: union needs no label; with/without do. A chosen label rides
    // along in every scope so certainty badges have a reference.
    if (S.scope !== 'union' && !S.labelFilter) return null;
    p.set('scope', S.scope);
    if (S.labelFilter) p.set('label_id', S.labelFilter);
    p.set('sort', S.manageSort);
    if ($('#auto-only') && $('#auto-only').checked) p.set('auto_only', '1');
  } else if (['label_top'].includes(S.mode)) {
    if (!S.labelFilter) return null;
    p.set('label_id', S.labelFilter);
    if (S.uncSort) p.set('sort', 'unc');
  }
  if (S.shuffle && S.mode === 'label_top') p.set('shuffle', S.shuffle);
  return p;
}
// Every mode pages by K — one uniform batch workflow (select images ×
// target labels → Apply).
const pageSize = () => Math.max(1, S.nbK);
function renderQTools() {
  const q = $('#qtools');
  if (S.mode === 'label_top') {
    q.style.display = 'flex';
    q.innerHTML =
      `<button id="qt-unc">${S.uncSort ? 'Uncertain ↑' : 'Score ↓'}</button>` +
      `<button id="qt-shuffle" title="Reshuffle the queue with a new random order — use when the first page shows nothing like your target class. Same seed keeps pages stable while flipping.">Shuffle</button>` +
      `<span class="dim">for the selected label's undecided cells</span>`;
    $('#qt-unc').onclick = () => { S.uncSort = !S.uncSort; renderQTools(); loadQueue(0); };
    $('#qt-shuffle').onclick = () => { S.shuffle += 1; loadQueue(0); };
  } else if (S.mode === 'label_all') {
    q.style.display = 'flex';
    q.innerHTML =
      `<span>Scope</span>` +
      `<select id="qt-scope" title="union = every labeled cell; with = cells carrying the selected label; without = labeled cells missing it.">
         <option value="union">All labeled (union)</option>
         <option value="with">With selected label</option>
         <option value="without">Without selected label</option>
       </select>` +
      `<span>Sort</span>` +
      `<select id="qt-sort" title="Certainty ↑ = least certain first (re-check auto labels); Certainty ↓ = most certain first; Suspicious = likely mislabels with evidence.">
         <option value="asc">Certainty ↑ (uncertain first)</option>
         <option value="desc">Certainty ↓ (certain first)</option>
         <option value="review">Suspicious ⚠ (likely mislabels)</option>
       </select>` +
      `<label title="Show only cells AUTO-annotated for the selected label (blue A mark)."><input id="auto-only" type="checkbox"> Auto only</label>`;
    $('#qt-scope').value = S.scope;
    $('#qt-sort').value = S.manageSort;
    $('#qt-scope').onchange = () => { S.scope = $('#qt-scope').value; loadQueue(0); };
    $('#qt-sort').onchange = () => { S.manageSort = $('#qt-sort').value; loadQueue(0); };
    const ao = $('#auto-only');
    if (ao) ao.onchange = () => loadQueue(0);
  } else {
    q.style.display = 'none'; q.innerHTML = '';
  }
}
async function loadQueue(page, keepSel) {
  if (page !== undefined) S.page = page;
  const ps = pageSize();
  const p = queueParams({ limit: ps, offset: S.page * ps });
  if (!p) { toast('Select a label on the left first'); return; }
  renderQTools();
  const j = await api('/api/queue?' + p);
  // The queue may have shrunk (removals) — land on the last valid page.
  if (!j.cells.length && S.page > 0 && j.total > 0) {
    S.page = Math.floor((j.total - 1) / ps);
    return loadQueue();
  }
  S.queue = j.cells; S.queueTotal = j.total;
  // The selection survives PAGE FLIPS (collect wrong cells across pages,
  // then Remove once); any other queue-context change (mode / label /
  // source / re-sort) clears it, so a batch write never hits cells the
  // user can no longer see.
  if (!keepSel) S.sel.clear();
  render(); renderPager();
  // The banner tracks EVERY load — a stale empty-queue message must
  // disappear once the queue has cells again.
  showBanner(S.queue.length ? null : queueEmptyText());
}
function renderPager() {
  const el = $('#pager');
  if (S.queueTotal === 0) {
    el.style.display = 'none'; el.innerHTML = ''; return;
  }
  const pages = Math.max(1, Math.ceil(S.queueTotal / pageSize()));
  if (S.page >= pages) S.page = pages - 1;
  el.style.display = 'flex';
  const mk = (label, page, disabled, cur) =>
    `<button class="pg${cur ? ' cur' : ''}" data-p="${page}"${disabled ? ' disabled' : ''}>${label}</button>`;
  let html = mk('First', 0, S.page === 0, false);
  let last = -1;
  for (let p = 0; p < pages; p++) {
    if (p === 0 || p === pages - 1 || Math.abs(p - S.page) <= 2) {
      if (last >= 0 && p - last > 1) html += '<span class="pgap">…</span>';
      html += mk(String(p + 1), p, false, p === S.page);
      last = p;
    }
  }
  html += mk('Last', pages - 1, S.page === pages - 1, false);
  // Type a page number + Enter (or blur) to jump straight to it; a jump is
  // navigation, so the selection survives it like any page flip.
  html += `<input id="pg-jump" type="number" min="1" max="${pages}"` +
          ` value="${S.page + 1}" title="Type a page number and press Enter to jump">` +
          `<span class="pgap">/ ${pages}</span>`;
  el.innerHTML = html;
  el.querySelectorAll('button.pg').forEach(b => b.onclick = () => {
    const pg = +b.dataset.p;
    if (pg !== S.page) loadQueue(pg, true);   // page flip keeps the selection
  });
  const jump = el.querySelector('#pg-jump');
  const doJump = () => {
    const v = parseInt(jump.value);
    if (v >= 1 && v <= pages && v - 1 !== S.page) loadQueue(v - 1, true);
    else jump.value = S.page + 1;
  };
  jump.onkeydown = e => { if (e.key === 'Enter') doJump(); };
  jump.onchange = doJump;
}
function queueEmptyText() {
  if (S.mode==='label_top') return 'No undecided cells left for this label.';
  if (S.mode==='label_all') {
    if (S.manageSort === 'review')
      return S.scope === 'with'
        ? 'No suspicious decisions — every decided cell is consistent with its label.'
        : 'Suspicious ranking needs the "With selected label" scope.';
    if (S.scope === 'union') return 'No cell carries any label yet — the union appears here as you annotate.';
    if (S.scope === 'without') return 'Every labeled cell already carries this label.';
    return 'No cell carries this label yet — positive decisions appear here as you make them.';
  }
  if (S.mode==='unlabeled') return 'No unlabeled cells left.';
  return 'Queue is empty';
}
function showBanner(html) {
  const b = $('#banner');
  if (!html) { b.style.display='none'; return; }
  b.style.display = 'flex';
  b.innerHTML = html;
}
// ---- rendering -------------------------------------------------------------
function render() { renderTargets(); renderGrid(); renderSelCount(); }
function renderSelCount() {
  $('#sel-count').textContent = S.sel.size ? `${S.sel.size} selected` : '';
}
function renderTargets() {
  // Target labels: everything Apply + / Apply − / Remove writes. Toggled
  // here or with keys 1-9 (sidebar numbering); clicking a sidebar label
  // makes it the ONLY checked one.
  $('#targets').innerHTML = S.labels.map(l =>
    `<span class="tgt ${S.targets.has(l.label_id)?'on':''}" style="--c:${l.color}" data-lid="${l.label_id}"` +
    ` title="Checked = Apply + / Apply − / Remove write this label${l.idx<=9?' (toggle with key '+l.idx+')':''}">` +
    `<span class="dot" style="background:${l.color}"></span>${esc(l.name)}</span>`).join('')
    || '<span style="color:#999;font-size:12px">no labels yet — add one on the left</span>';
  document.querySelectorAll('#targets .tgt').forEach(el => el.onclick = () => {
    const lid = +el.dataset.lid;
    S.targets.has(lid) ? S.targets.delete(lid) : S.targets.add(lid);
    el.classList.toggle('on');
  });
}
function renderGrid() {
  $('#grid').innerHTML = S.queue.map((c,i) => {
    const dots = S.labels.filter(l => c.labels[l.label_id] !== undefined)
      .map(l => `<span class="dot ${c.labels[l.label_id]===0?'neg':''}" style="background:${l.color}"></span>`).join('');
    const tip = esc(c.filename) + ' · ' + esc(c.source_name) +
      (c.preset ? ' · preset: ' + esc(c.preset) : '') +
      (c.auto ? ' ⚠ auto-annotated — verify in Manage view' : '') +
      (c.susp ? ' ⚠ suspicious — see evidence in zoom' : '');
    return `<div class="cellbox ${S.sel.has(c.filepath)?'sel':''}" data-i="${i}" title="${tip}">` +
      `<img loading="lazy" src="${imgURL(c.filepath,S.cellPx)}" style="width:${S.cellPx}px;height:${S.cellPx}px">` +
      ((c.suggest||[]).length ? `<span class="sugb">${c.suggest[0].score.toFixed(2)}${({model:' M',aml:' A'})[c.suggest[0].src]||' K'}</span>` : '') +
      (c.susp ? `<span class="susb">⚠${c.susp.susp.toFixed(2)}</span>` :
        (c.cert != null ? `<span class="cert">${c.cert.toFixed(2)}</span>` : '')) +
      (c.auto ? '<span class="automark">A</span>' : '') +
      `<div class="dots">${dots}</div></div>`;
  }).join('');
  document.querySelectorAll('#grid .cellbox').forEach(el => {
    const c = S.queue[+el.dataset.i];
    el.onclick = () => {
      S.sel.has(c.filepath) ? S.sel.delete(c.filepath) : S.sel.add(c.filepath);
      el.classList.toggle('sel');
      renderSelCount();
    };
    el.ondblclick = () => openZoom(+el.dataset.i);
  });
}
// ---- zoom: inspect AND label ------------------------------------------------
// Double-click a thumbnail for a close look. Unlike a plain viewer, the
// zoomed cell can be labeled right here: chips (or keys) cycle its labels,
// ←/→ walk the queue, and suspicious cells show the contradicting evidence
// cell side by side.
function openZoom(i) {
  S.zoomIdx = i;
  renderZoom();
  $('#zoom').style.display = 'flex';
}
function closeZoom() { $('#zoom').style.display = 'none'; S.zoomIdx = -1; }
$('#zoom').onclick = (e) => { if (e.target.id === 'zoom') closeZoom(); };
function renderZoom() {
  const c = S.queue[S.zoomIdx];
  if (!c) { closeZoom(); return; }
  $('#zoom-img').src = imgURL(c.filepath, 768);
  $('#zoom-img').onclick = (e) => { e.stopPropagation(); closeZoom(); };
  const bits = [`<b>${esc(c.filename)}</b>`, `<span class="dim">${esc(c.source_name)}</span>`];
  if (c.preset) bits.push(`<span class="dim">preset: ${esc(c.preset)}</span>`);
  if ((c.suggest||[]).length) bits.push(`<span class="zg">sug ${c.suggest[0].score.toFixed(2)}</span>`);
  if (c.cert != null) bits.push(`<span class="zo">cert ${c.cert.toFixed(2)}</span>`);
  if (c.auto) bits.push('<span class="zo">auto</span>');
  if (c.susp) bits.push(`<span class="zr">⚠ suspicious: own ${c.susp.own_sim.toFixed(2)} vs ${c.susp.ev_state===0?'neg':'pos'} ${c.susp.ev_sim.toFixed(2)}</span>`);
  const cur = S.labels.filter(l => c.labels[l.label_id] !== undefined)
    .map(l => `<span class="dot ${c.labels[l.label_id]===0?'neg':''}" style="background:${l.color};width:9px;height:9px;border-radius:50%;display:inline-block"></span>`).join(' ');
  if (cur) bits.push(`<span>${cur}</span>`);
  $('#zoom-meta').innerHTML = bits.join(' · ');
  $('#zoom-chips').innerHTML = S.labels.map(l => {
    const st = c.labels[l.label_id];
    const cls = st === 1 ? 'pos' : (st === 0 ? 'neg' : '');
    return `<span class="zchip ${cls}" style="--c:${l.color}" data-lid="${l.label_id}"` +
      ` title="off → positive → explicit negative → off${l.idx<=9?' (key '+l.idx+' / Shift+'+l.idx+')':''}">` +
      `<span class="dot" style="background:${l.color}"></span>${l.idx<=9?l.idx+'. ':''}${esc(l.name)}</span>`;
  }).join('') || '<span style="color:#999;font-size:12px">no labels yet</span>';
  document.querySelectorAll('#zoom-chips .zchip').forEach(el =>
    el.onclick = (e) => { e.stopPropagation(); zoomCycle(+el.dataset.lid); });
  // Evidence cell (suspicious queue): shown side by side for comparison.
  const evw = $('#zoom-ev-wrap');
  if (c.susp && c.susp.ev_file) {
    $('#zoom-ev').src = imgURL(c.susp.ev_file, 384);
    $('#zoom-ev-cap').textContent =
      `evidence: the contradicting ${c.susp.ev_state===0 ? 'NEGATIVE' : 'POSITIVE'} cell (sim ${c.susp.ev_sim.toFixed(2)})`;
    evw.style.display = 'flex';
  } else evw.style.display = 'none';
}
async function zoomSet(lid, state) {
  const c = S.queue[S.zoomIdx];
  if (!c) return;
  const j = await api('/api/annotate', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepath: c.filepath, label_id: lid, state }) });
  if (state === 'clear') delete c.labels[lid]; else c.labels[lid] = state;
  if (j.auto_applied) toast(`auto-annotated ${j.auto_applied} more — see Manage`);
  renderZoom(); renderGrid(); refreshStats();
}
function zoomCycle(lid) {
  const c = S.queue[S.zoomIdx];
  const cur = c.labels[lid];
  const next = cur === undefined ? 1 : (cur === 1 ? 0 : 'clear');
  zoomSet(lid, next);
}

// ---- batch apply -----------------------------------------------------------
// The ONE apply path: selected images × checked target labels, one server
// action (one transaction, one undoable op). state: 1 = positive, 0 =
// explicit negative, 'clear' removes the decision (in the Manage view that
// drops the cells from the label).
async function applyTargets(state) {
  if (!S.sel.size) { toast('Select some images first (click thumbnails)'); return; }
  if (!S.targets.size) { toast('Check at least one target label above the grid'); return; }
  const n = S.sel.size, lids = [...S.targets];
  const j = await api('/api/annotate_batch', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepaths:[...S.sel], label_id: lids[0], state,
                           extra_label_ids: lids.slice(1) }) });
  let autoN = j.auto_applied || 0;
  S.queue.forEach(c => {
    if (!S.sel.has(c.filepath)) return;
    for (const lid of lids) {
      if (state === 'clear') delete c.labels[lid]; else c.labels[lid] = state;
    }
  });
  toast(`${state === 1 ? 'Applied +' : state === 0 ? 'Applied −' : 'Removed'} ` +
        `${lids.length} label(s) on ${n} cells` +
        (autoN ? ` — auto-annotated ${autoN} more` : ''));
  S.sel.clear();
  await refreshStats();
  loadQueue(S.page);   // memberships changed — recompute the page
}
$('#btn-apply-pos').onclick = () => applyTargets(1);
$('#btn-apply-neg').onclick = () => applyTargets(0);
$('#btn-remove').onclick = () => applyTargets('clear');
$('#btn-sel-all').onclick = () => { S.queue.forEach(c => S.sel.add(c.filepath)); renderGrid(); renderSelCount(); };
$('#btn-sel-none').onclick = () => { S.sel.clear(); renderGrid(); renderSelCount(); };
$('#btn-undo').onclick = doUndo;
async function doUndo() {
  const j = await api('/api/undo', { method:'POST' });
  if (!j.undone) { toast(j.message || 'Nothing to undo'); return; }
  toast(`Undone: ${j.n} decision(s) restored — keep Ctrl+Z to undo further`);
  await refreshStats();
  loadQueue(S.page);
}
$('#btn-auto-apply').onclick = async () => {
  if (!S.labelFilter) { toast('Select a label on the left first'); return; }
  const j = await api('/api/auto_apply', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ label_id: S.labelFilter }) });
  const parts = Object.entries(j.applied || {}).map(([n, c]) => `${n} ×${c}`);
  toast(parts.length
    ? `Auto-applied: ${parts.join(', ')} — review them in Manage (Suspicious / Certainty ↑)`
    : 'Nothing to auto-apply (more positives needed, or no confident cells)');
  await refreshStats();
  loadQueue(0);
};
let _autoClearArmed = false;
$('#btn-auto-clear').onclick = async () => {
  if (!S.labelFilter) { toast('Select a label on the left first'); return; }
  const b = $('#btn-auto-clear');
  if (!b.dataset.armed) {
    // Two-step confirm: a stray click must not wipe a whole auto-run.
    b.dataset.armed = '1'; b.textContent = 'Confirm';
    setTimeout(() => { if (b.isConnected) { b.dataset.armed = ''; b.textContent = 'Remove auto'; } }, 3000);
    return;
  }
  b.dataset.armed = ''; b.textContent = 'Remove auto';
  const j = await api('/api/auto_clear', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ label_id: S.labelFilter }) });
  toast(`Removed ${j.removed} auto annotations of the label`);
  await refreshStats();
  loadQueue(0);
};
$('#nb-k').onchange = () => {
  const v = parseInt($('#nb-k').value);
  if (v >= 1 && v <= 200) {
    S.nbK = v;
    localStorage.setItem('label_nbk', String(v));
    loadQueue(S.page);   // K is the page size everywhere
  } else $('#nb-k').value = S.nbK;
};
$('#cell-px').value = S.cellPx;
$('#cell-px').onchange = () => {
  const v = parseInt($('#cell-px').value);
  if (v >= 96 && v <= 512) {
    S.cellPx = v;
    localStorage.setItem('label_cellsize', String(v));
    render();
  } else $('#cell-px').value = S.cellPx;
};

// ---- top-level actions -----------------------------------------------------
$('#btn-add').onclick = async () => {
  const name = $('#new-label').value.trim(); if (!name) return;
  await api('/api/labels', { method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify({ name }) });
  $('#new-label').value = ''; await refreshStats(); toast(`Label "${name}" added`);
};
$('#new-label').addEventListener('keydown', e => { if (e.key === 'Enter') $('#btn-add').onclick(); });
$('#btn-model-labels').onclick = async () => {
  const j = await api('/api/labels_from_model', { method:'POST' });
  await refreshStats(); toast(`Created ${j.created.length} labels`);
};
function updateModeTip() {
  const o = $('#mode').selectedOptions[0];
  $('#mode').title = o ? o.title : '';
}
$('#mode').onchange = () => {
  S.mode = $('#mode').value;
  S.shuffle = 0;   // a mode switch restarts every queue from its own order
  updateModeTip(); refreshStats().then(() => loadQueue(0));
};
function setBusy(on) {
  document.querySelector('#busy').style.display = on ? 'flex' : 'none';
}
$('#label-mode').onchange = async () => {
  const mode = $('#label-mode').value;
  if (mode === S.labelMode) return;
  setBusy(true);   // the other store re-registers cells + invalidates scores
  try {
    await api('/api/label_mode', { method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ mode }) });
    // The other mode runs on its own database: label ids do not cross, so
    // drop every piece of state tied to the previous store and reload.
    S.labelMode = mode;
    S.labelFilter = null;
    S.targets.clear(); S.sel.clear();
    toast(mode === 'single'
      ? 'Switched to the single-label project (independent label_single.db)'
      : 'Switched to the multi-label project (independent label_multiple.db)');
    renderTargets();
    await refreshStats();
    await loadQueue(0);
  } finally {
    setBusy(false);
  }
};
$('#source').onchange = () => { S.source = $('#source').value; loadQueue(); };
$('#thr').onchange = () => {
  S.thr = parseFloat($('#thr').value) || S.thr;
  loadQueue(S.page);   // suggestions/certainties apply immediately
};

// ---- display adjustment (render-only: percentile window + gamma) -----------
let _dispTimer = null;
function setDisp(patch) {
  Object.assign(S.disp, patch);
  localStorage.setItem('label_display', JSON.stringify(S.disp));
  clearTimeout(_dispTimer);
  _dispTimer = setTimeout(render, 150);   // debounce the image refetches
}
$('#disp-con').oninput = e => {
  const t = +e.target.value / 100;   // 0 -> 0.1/99.9 window, 100 -> 10/90
  setDisp({ lo: +(0.1 + 9.9 * t).toFixed(2), hi: +(99.9 - 9.9 * t).toFixed(2) });
};
$('#disp-gamma').oninput = e => setDisp({ gamma: +e.target.value / 100 });
$('#disp-reset').onclick = () => {
  S.disp = { lo: 0.1, hi: 99.9, gamma: 1.0 };
  localStorage.removeItem('label_display');
  $('#disp-con').value = 0; $('#disp-gamma').value = 100;
  render();
};
// Restore slider positions from the persisted display state.
$('#disp-con').value = Math.max(0, Math.min(100, Math.round((S.disp.lo - 0.1) / 9.9 * 100)));
$('#disp-gamma').value = Math.max(20, Math.min(300, Math.round(S.disp.gamma * 100)));

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeZoom(); return; }
  const inZoom = S.zoomIdx >= 0;
  // Ctrl+Z undo works everywhere (also inside zoom).
  if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault(); doUndo(); return;
  }
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key;
  if (k >= '1' && k <= '9') {
    const l = S.labels[+k-1]; if (!l) return;
    if (inZoom) {
      // Keys label the ZOOMED cell: key = positive, Shift+key = explicit
      // negative — the fastest verify loop (double-click, 1, →, 1, …).
      const st = e.shiftKey ? 0 : 1;
      const cur = S.queue[S.zoomIdx].labels[l.label_id];
      zoomSet(l.label_id, cur === st ? 'clear' : st);
    } else {
      S.targets.has(l.label_id) ? S.targets.delete(l.label_id) : S.targets.add(l.label_id);
      renderTargets();
    }
  }
  else if (k === 'a' || k === 'A') applyTargets(e.shiftKey ? 0 : 1);
  else if (k === 'r' || k === 'R') applyTargets('clear');
  else if (k === 'ArrowRight') {
    if (inZoom) { e.preventDefault(); zoomNav(1); }
    else if ((S.page + 1) * pageSize() < S.queueTotal) loadQueue(S.page + 1, true);
  }
  else if (k === 'ArrowLeft') {
    if (inZoom) { e.preventDefault(); zoomNav(-1); }
    else if (S.page > 0) loadQueue(S.page - 1, true);
  }
});
function zoomNav(d) {
  const j = S.zoomIdx + d;
  if (j >= 0 && j < S.queue.length) { S.zoomIdx = j; renderZoom(); }
  else toast(d > 0 ? 'End of page — flip the page (→) to continue' : 'Start of page');
}

// Numeric inputs apply while typing (debounced 'input' — no blur needed);
// 'change' keeps the immediate commit on blur/Enter. Mid-typing values
// that do not parse are silently skipped — only the field's own change
// handler resets the visible value.
let _numTimer = null;
const debouncedApply = (apply) => {
  clearTimeout(_numTimer);
  _numTimer = setTimeout(apply, 400);
};
$('#nb-k').addEventListener('input', () => debouncedApply(() => {
  const v = parseInt($('#nb-k').value);
  if (v >= 1 && v <= 200) { S.nbK = v; localStorage.setItem('label_nbk', String(v)); loadQueue(S.page); }
}));
$('#cell-px').addEventListener('input', () => debouncedApply(() => {
  const v = parseInt($('#cell-px').value);
  if (v >= 96 && v <= 512) { S.cellPx = v; localStorage.setItem('label_cellsize', String(v)); render(); }
}));
$('#thr').addEventListener('input', () => debouncedApply(() => {
  const v = parseFloat($('#thr').value);
  if (v >= 0 && v <= 1) { S.thr = v; loadQueue(S.page); }
}));

// The select's DOM default is its first option — sync it with the actual
// startup mode so the control never shows a state the app is not in.
$('#mode').value = S.mode;
updateModeTip();
refreshStats().then(() => loadQueue(0));
</script>
</body>
</html>
"""
