"""The label app's embedded single-page UI (vanilla JS, no build step).

Layout (light theme, same approach as reduction_vis):

  top bar      brand + image source + live stats + Size/K + Refresh model
               / Undo on row 1; display controls (Contrast/Gamma/Reset),
               the classify prediction filter, then the scope and sort
               radio groups + Shuffle on row 2
  left         tabbed: Labels (click = load the label's queue, drag =
               reorder) + a Selected-label box with the live counts |
               Clusters (cluster-assisted bulk labeling: one card per
               Leiden cluster over the shared space, medoid thumbnail,
               click = the cluster's queue) + the target chips + help /
               shortcuts in a collapsible box
  toolbar      target-label chips + Apply + / Apply − / Remove / selection
               tools; inside a cluster an extra bar writes the checked
               labels to the whole cluster ∩ scope in one undoable step
  grid         one page of thumbnails; click = select, shift+click =
               instant negative, double-click = zoom; each thumbnail
               carries its score/certainty badge and every annotated cell
               shows its label dots bottom-right
  zoom         full-size view where labels are TOGGLED DIRECTLY (chips or
               keys 1-9 cycle undecided -> positive -> explicit negative),
               ←/→ walk the queue, and suspicious cells show the
               contradicting evidence cell side by side

ONE unified queue (no queue-mode switch): pick an image source, click a
label, and the scope radio decides the members — undecided (to label,
ranked by the exemplar/model score), positive, missing, negatives, or all
labeled — while the sort radio picks the ranking (best first / uncertain
first / suspicious mislabel check). Batch-select and Apply + / Apply − /
Remove work in every scope; shift+click records an instant negative; the
Refresh-model button fits a per-label logistic scorer on the current
positives + negatives (manual — new writes mark it stale until you refresh
again). Every write goes through to the DB immediately; Ctrl+Z undoes the
last action; label_export.csv is re-written after every write.
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
#source { max-width:150px; }#stats { color:#555; font-size:12px; white-space:nowrap; }
#stats b { color:#222; }
#refit-status { color:#555; font-size:12px; white-space:nowrap; }
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
/* Scope / sort radio groups (top bar row 2, after the display Reset). */
.radios { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.radios label { display:inline-flex; align-items:center; gap:3px; font-size:12px; color:#555; white-space:nowrap; cursor:pointer; }
.radios label:hover { color:#222; }
.radios input { accent-color:var(--acc); margin:0; cursor:pointer; }
.radios input:disabled { cursor:default; opacity:.4; }
#pred-wrap select { max-width:140px; }
/* Sidebar tabs: Labels | Clusters. */
.tabs { display:flex; gap:4px; }
.tab { flex:1; font-size:12px; padding:4px 0; background:#f0f0f2; border:1px solid var(--line); border-radius:4px; cursor:pointer; }
.tab:hover { background:#e8e8f0; }
.tab.on { background:var(--acc); color:#fff; border-color:var(--acc); font-weight:600; }
/* Cluster cards (Clusters tab): medoid thumbnail + id + size + per-label
   undecided count. */
.clu-card { display:flex; align-items:center; gap:8px; padding:4px; border:1px solid var(--line); border-radius:4px; cursor:pointer; margin-bottom:4px; }
.clu-card:hover { background:#f0f0f2; }
.clu-card.active { outline:2px solid var(--acc); }
.clu-card img { width:52px; height:52px; object-fit:contain; background:#fff; border-radius:3px; flex-shrink:0; }
.clu-info { display:flex; flex-direction:column; font-size:12px; gap:1px; min-width:0; }
.clu-info .clu-size { color:#666; }
.clu-info .clu-und { color:var(--warn); font-weight:600; }
.clu-info .clu-view { color:#888; }
.clu-info .clu-view.zero { color:#c22; font-weight:600; }
/* Whole-cluster action bar above the grid (visible inside a cluster). */
#cluster-bar { display:none; align-items:center; gap:6px; padding:5px 10px; background:#eef2fd; border-bottom:1px solid var(--line); flex-wrap:wrap; }
#clu-title { font-size:12px; color:#333; }
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
    <label title="Thumbnail display size (px). 48-512. Display only — the model always sees its own square input.">Size
      <input id="cell-px" type="number" min="48" max="512" step="8" style="width:56px">
    </label>
    <label title="Page size: images per page (the batch you review at once). 1-200.">K
      <input id="nb-k" type="number" min="1" max="200" style="width:56px">
    </label>
    <button id="btn-refit" title="Refit a per-label logistic scorer from the current positives + explicit negatives of every label that has enough of both. A refit label's score becomes its model probability — negatives then shape a real decision boundary instead of only nudging a similarity. Manual by design: new writes make the models stale (status below) until you click again.">Refresh model</button>
    <span id="refit-status"></span>
    <button id="btn-undo" title="Undo the last action (Ctrl+Z): a batch apply or a single write — each is one revertible step. Keep pressing to undo further back.">Undo</button>
  </div>
  <div class="trow">
    <label title="Display contrast: symmetrically narrows the per-channel percentile window (0.1/99.9 at 0 → 10/90 at 100). Display only — features are unaffected.">Contrast
      <input id="disp-con" type="range" min="0" max="100" step="1" style="width:64px;vertical-align:middle">
    </label>
    <label title="Display gamma: 1 = off, > 1 brightens midtones, < 1 darkens. Display only — features are unaffected.">Gamma
      <input id="disp-gamma" type="range" min="20" max="300" step="5" style="width:64px;vertical-align:middle">
    </label>
    <button id="disp-reset" title="Reset contrast and gamma to the default render">Reset</button>
    <span class="sep"></span>
    <label id="pred-wrap" title="Classify bundle only: keep only cells whose argmax prediction is this class. 'Prob ↓' (sort) ranks by this class's probability — or by each cell's argmax confidence while All is picked." style="display:none">Predicted
      <select id="pred-sel"><option value="">All</option></select>
    </label>
    <span class="sep" id="pred-sep" style="display:none"></span>
    <span class="radios" id="scope-radios">
      <label title="Cells WITHOUT a decision for the selected label, ranked by the label's score (model P or exemplar score) — the main labeling queue. Needs a model."><input type="radio" name="scope" value="undecided">To label</label>
      <label title="Cells positive for the selected label — verify them (Certainty or the Suspicious mislabel check)."><input type="radio" name="scope" value="with">Positive</label>
      <label title="Labeled cells MISSING the selected label — completion candidates."><input type="radio" name="scope" value="without">Missing</label>
      <label title="The selected label's EXPLICIT negatives — review them: Certainty ↓ puts the most positive-like, likely mislabeled ones first; select and Remove to undo."><input type="radio" name="scope" value="neg">Negatives</label>
      <label title="Every cell carrying ANY label — the whole pool to re-check."><input type="radio" name="scope" value="union">All labeled</label>
    </span>
    <span class="sep"></span>
    <span class="radios" id="sort-radios">
      <label title="undecided: highest score first. decided scopes: most positive-like (certain) first."><input type="radio" name="sort" value="desc">Best ↓</label>
      <label title="undecided: smallest pos/neg margin first (active learning). decided scopes: least positive-like first."><input type="radio" name="sort" value="unc">Uncertain ↑</label>
      <label title="Positive scope only: the leave-one-out mislabel check, most suspicious first (zoom shows the contradicting cell)."><input type="radio" name="sort" value="review">Suspicious ⚠</label>
      <label id="sort-medoid-wrap" title="Cluster view: most similar to the cluster's medoid first — the most typical members lead, so the first page tells you what the cluster is." style="display:none"><input type="radio" name="sort" value="medoid">Medoid ↓</label>
      <label id="sort-prob-wrap" title="Classify bundle: highest P(class) first — the class picked in the Predicted dropdown, or each cell's argmax confidence while All is picked." style="display:none"><input type="radio" name="sort" value="prob">Prob ↓</label>
    </span>
    <button id="qt-shuffle" title="Reshuffle the To-label queue with a new random order — use when the first page shows nothing like your target class. Same seed keeps pages stable while flipping.">Shuffle</button>
  </div>
</div>
<div id="main">
  <div id="sidebar">
    <div class="tabs">
      <button id="tab-labels" class="tab on" title="Per-label queues">Labels</button>
      <button id="tab-clusters" class="tab" title="Cluster-assisted bulk labeling: Leiden clusters over the shared embedding space (config cluster.target). Review a cluster's medoid, then write target labels to the whole cluster in ONE undoable step." style="display:none">Clusters</button>
    </div>
    <div id="side-labels" style="display:flex;flex-direction:column;gap:10px">
    <div class="side-box">
      <h3>Labels — click = its queue</h3>
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
    </div>
    </div>
    <div id="side-clusters" style="display:none;flex-direction:column;gap:10px">
      <div class="side-box">
        <h3>Targets — check what Apply writes</h3>
        <div id="clu-targets" class="targets-strip"></div>
      </div>
      <div class="side-box">
        <h3>Clusters — click = its queue <span id="clu-meta" style="float:right;color:#888;text-transform:none;letter-spacing:0;font-weight:400"></span></h3>
        <div id="cluster-list"></div>
      </div>
    </div>
    <div class="side-box">
      <details id="help">
        <summary>Workflow &amp; shortcuts</summary>
        <b>One queue per label</b> — pick an image source on top, click a
        label on the left, and the top-bar radios decide what the queue
        shows. <b>Scope</b>: To label = cells without a decision for this
        label, ranked by the <b>exemplar score</b> (similarity to the
        positives minus the negative penalty) — the main labeling queue;
        Positive / Missing / Negatives / All labeled = the already-decided
        cells for verification. <b>Sort</b>: Best ↓ = highest score (or
        most positive-like) first; Uncertain ↑ = smallest pos/neg margin
        (or least certain) first; Suspicious ⚠ = the leave-one-out
        mislabel check on the Positive scope (zoom shows the contradicting
        cell). Shuffle = new random To-label order. Select the confident
        ones, <b>Apply +</b> (A); <b>Apply −</b> (Shift+A) records
        explicit negatives for lookalikes that are NOT the label — or
        shift+click a thumbnail for an instant negative. Negatives push
        every similar cell down the To-label ranking, and every annotated
        cell shows its label dots bottom-right. The queue refreshes after
        every write, so the order follows your latest decisions.
        <b>Refresh model</b> (top bar) = refit a per-label logistic scorer
        on the current positives + negatives of every label that has
        enough of both — a refit label's score/badge becomes its model
        probability, where negatives define a real decision boundary. New
        writes mark it stale; click again to catch up.
        <b>Clusters tab</b> (left, with cluster.target configured) =
        cluster-assisted bulk labeling: the whole dataset is pre-split into
        fine-grained Leiden clusters over the embedding space, largest
        first, each card showing its medoid (most typical cell). Click a
        card to narrow the queue to that cluster — Medoid ↓ sorts the most
        typical members first so page one tells you what the cluster is —
        then use <b>Cluster Apply + / − / Remove</b> to write the checked
        target labels to the whole cluster ∩ current scope in ONE undoable
        step (the default To-label scope fills only undecided members).
        ←/→ prev/next cluster; the orange number on each card is how many
        members still lack the selected label.
        <b>Predicted</b> (top bar, classify bundle only) = keep only cells
        whose argmax prediction is the picked class; <b>Prob ↓</b> ranks by
        that class's probability (each cell's argmax confidence while All
        is picked) — the fast loop for verifying/correcting model
        predictions, e.g. after "Create labels from model classes".
        <b>Shortcuts</b><br>
        click = select · shift+click = instant negative · double-click =
        zoom · 1-9 = toggle target labels<br>
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
          <span style="color:var(--ok)">■</span> green: recommendation score
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
        <span id="targets" class="targets-strip"></span>
        <span style="flex:1"></span>
        <span id="sel-count"></span>
        <button id="btn-apply-pos" class="primary" title="Set every CHECKED target label as POSITIVE on every selected image (A)">Apply +</button>
        <button id="btn-apply-neg" class="neg" title="Set every CHECKED target label as an EXPLICIT NEGATIVE on every selected image (Shift+A). Negatives push lookalikes down the ranking and are the training data for the Refresh-model scorer. Shift+click a single thumbnail for the same write on one cell.">Apply −</button>
        <button id="btn-remove" title="CLEAR the checked target labels' decisions on the selected images (R). In the Manage view this drops them from the label.">Remove</button>
        <button id="btn-sel-all">Select all</button>
        <button id="btn-sel-none">Clear</button>
      </div>
    </div>
    <div id="cluster-bar">
      <button id="clu-exit" title="Leave the cluster view — back to the whole dataset">✕ Exit</button>
      <span id="clu-title"></span>
      <span style="flex:1"></span>
      <button id="clu-prev" title="Previous cluster (largest-first order)">← Prev</button>
      <button id="clu-next" title="Next cluster">Next →</button>
      <span class="sep"></span>
      <button id="clu-pos" class="primary" title="Apply every CHECKED target label as POSITIVE to the whole cluster ∩ current scope — ONE undoable step. The default To-label scope fills only the undecided members; your explicit positives/negatives are kept.">Cluster Apply +</button>
      <button id="clu-neg" class="neg" title="Apply every CHECKED target label as an EXPLICIT NEGATIVE to the whole cluster ∩ current scope — one undoable step.">Cluster Apply −</button>
      <button id="clu-clear" title="CLEAR the checked target labels' decisions on the whole cluster ∩ current scope — one undoable step.">Cluster Remove</button>
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
const S = { labels:[], labelById:{}, labelFilter:null, source:'',
            scope:'undecided', sort:'desc', shuffle:0,
            cluster:null, clusters:[], clusterRes:null, clusterEnabled:false,
            predLabel:'', sideTab:'labels',
            nbK:100, page:0,
            cellPx:(()=>{ const v = parseInt(localStorage.getItem('label_cellsize'));
              return (v >= 48 && v <= 512) ? v : 100; })(),
            queue:[], queueTotal:0,
            sel:new Set(), targets:new Set(),
            disp:(()=>{ const d = { lo:0.1, hi:99.9, gamma:1.0 };
              try { Object.assign(d, JSON.parse(localStorage.getItem('label_display')||'{}')); } catch(e) {}
              return d; })(),
            dragId:null, zoomIdx:-1,
            hasModel:false, classNames:[], total:0, labeled:0, undecided:0,
            modelsFitted:0, modelsStale:false,
            labelMode:'multi' };
const esc = s => (s ?? '').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
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
  S.nbK = parseInt(localStorage.getItem('label_nbk')) || j.page_size;
  S.total = j.total; S.labeled = j.labeled; S.undecided = j.undecided;
  S.labelById = {}; S.labels.forEach((l,i)=>{ l.idx = i+1; S.labelById[l.label_id] = l; });
  $('#nb-k').value = S.nbK;
  $('#stats').innerHTML = `<b>${S.labeled}</b> / ${S.total} labeled` +
    ` · <b>${S.undecided}</b> undecided`;
  $('#btn-model-labels').style.display = S.classNames.length ? 'block' : 'none';
  const src = $('#source'), keep = src.value;
  src.innerHTML = '<option value="">All sources</option>' +
    j.sources.map(s=>`<option value="${esc(s.path)}">${esc(s.name)}</option>`).join('');
  src.value = keep;
  // Reflect the queue controls (and disable the model-dependent options
  // without a model: the score-defined To-label queue, the Suspicious
  // check). Without a model the undecided scope falls back to Positive.
  syncQueueControls();
  S.labelMode = j.label_mode || 'multi';
  $('#label-mode').value = S.labelMode;
  // Refresh-model status: how many labels currently score with a fitted
  // logistic model, and whether writes have happened since the last fit.
  S.modelsFitted = j.models_fitted || 0;
  S.modelsStale = !!j.models_stale;
  $('#btn-refit').disabled = !S.hasModel;
  $('#refit-status').textContent = !S.hasModel ? ''
    : (S.modelsFitted ? `${S.modelsFitted} refit` : 'kNN')
      + (S.modelsStale ? ' · stale' : '');
  // ---- clusters tab + classify prediction filter ------------------------
  S.clusterEnabled = !!(j.cluster && j.cluster.enabled);
  S.clusterRes = (j.cluster && j.cluster.res) != null ? j.cluster.res : null;
  $('#tab-clusters').style.display = S.clusterEnabled ? '' : 'none';
  if (!S.clusterEnabled && S.sideTab === 'clusters') setSideTab('labels');
  if (!S.classNames.includes(S.predLabel)) S.predLabel = '';
  const pw = $('#pred-wrap'), psep = $('#pred-sep');
  pw.style.display = S.classNames.length ? '' : 'none';
  psep.style.display = S.classNames.length ? '' : 'none';
  if (S.classNames.length) {
    const ps = $('#pred-sel');
    ps.innerHTML = '<option value="">All</option>' +
      S.classNames.map(c=>`<option value="${esc(c)}">${esc(c)}</option>`).join('');
    ps.value = S.predLabel;
  }
  if (S.clusterEnabled) await loadClusters();
  syncQueueControls();   // medoid/prob sort visibility depends on the above
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
      // The sidebar promises "click = load its queue": picking a label
      // selects it as the queue's label and the only checked target
      // (more chips can still be checked by hand for a multi-label
      // batch).
      S.labelFilter = lid;
      S.targets = new Set([lid]);
      S.shuffle = 0;   // a new label restarts from the queue's own ranking
      renderLabels(); renderSelBox(); renderTargets();
      if (S.clusterEnabled) loadClusters();   // per-card counts follow the label
      loadQueue(0);
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
  loadQueue();   // re-rank with the label gone (prompts if it WAS selected)
}
// The selected-label box: the clicked label's live decision counts.
function renderSelBox() {
  const box = $('#sel-box');
  const l = S.labelFilter != null ? S.labelById[S.labelFilter] : null;
  if (!l) { box.style.display = 'none'; return; }
  box.style.display = 'block';
  $('#sel-dot').style.background = l.color;
  $('#sel-name').textContent = l.name;
  $('#sel-counts').textContent = `+${l.n_pos} positives · −${l.n_neg} negatives`;
}

// ---- queue -----------------------------------------------------------------
// ONE queue per (source, label): the scope radio picks the members, the
// sort radio picks the ranking (see /api/queue).
function queueParams(extra) {
  if (!S.labelFilter) return null;
  const p = new URLSearchParams({ label_id: S.labelFilter,
                                  scope: S.scope, sort: S.sort, ...extra });
  if (S.source) p.set('source', S.source);
  if (S.cluster != null) p.set('cluster', S.cluster);
  if (S.predLabel) p.set('pred_label', S.predLabel);
  if (S.shuffle && S.scope === 'undecided') p.set('shuffle', S.shuffle);
  return p;
}
// Every mode pages by K — one uniform batch workflow (select images ×
// target labels → Apply).
const pageSize = () => Math.max(1, S.nbK);
async function loadQueue(page, keepSel) {
  if (page !== undefined) S.page = page;
  const ps = pageSize();
  const p = queueParams({ limit: ps, offset: S.page * ps });
  if (!p) { toast('Select a label on the left first'); return; }
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
  renderClusterBar();   // the bar's "N in view" tracks the queue total
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
  if (S.cluster != null) {
    const src = S.source
      ? ` — the Source filter ("${($('#source').selectedOptions[0] || {}).textContent || S.source}") may be hiding this cluster's cells`
      : '';
    return `Cluster #${S.cluster} has no cells in the current scope${src} — switch the scope radio or Source, or exit the cluster.`;
  }
  if (S.scope==='undecided') return 'No undecided cells left for this label.';
  if (S.scope==='with') return 'No cell carries this label yet — positive decisions appear here as you make them.';
  if (S.scope==='without') return 'Every labeled cell already carries this label.';
  if (S.scope==='neg') return 'No explicit negatives for this label yet — mark some with Apply − or shift+click.';
  return 'No cell carries any label yet — the union appears here as you annotate.';
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
  // makes it the ONLY checked one. Rendered twice — the toolbar above the
  // grid AND the Clusters tab's Targets box share one definition.
  const html = S.labels.map(l =>
    `<span class="tgt ${S.targets.has(l.label_id)?'on':''}" style="--c:${l.color}" data-lid="${l.label_id}"` +
    ` title="Checked = Apply + / Apply − / Remove write this label${l.idx<=9?' (toggle with key '+l.idx+')':''}">` +
    `<span class="dot" style="background:${l.color}"></span>${esc(l.name)}</span>`).join('')
    || '<span style="color:#999;font-size:12px">no labels yet — add one on the left</span>';
  document.querySelectorAll('.targets-strip').forEach(el => {
    el.innerHTML = html;
    el.querySelectorAll('.tgt').forEach(t => t.onclick = () => {
      const lid = +t.dataset.lid;
      S.targets.has(lid) ? S.targets.delete(lid) : S.targets.add(lid);
      t.classList.toggle('on');
    });
  });
}
function renderGrid() {
  $('#grid').innerHTML = S.queue.map((c,i) => {
    const dots = S.labels.filter(l => c.labels[l.label_id] !== undefined)
      .map(l => `<span class="dot ${c.labels[l.label_id]===0?'neg':''}" style="background:${l.color}"></span>`).join('');
    // The green badge is the recommendation score: in Collect it is the
    // exact score the queue sorted by (c.score); elsewhere the best
    // suggestion of any label.
    const score = (c.score != null) ? c.score
                : ((c.suggest||[]).length ? c.suggest[0].score : null);
    const tip = esc(c.filename) + ' · ' + esc(c.source_name) +
      (c.preset ? ' · preset: ' + esc(c.preset) : '') +
      (c.pred ? ' · pred: ' + esc(c.pred.class) + ' (' + c.pred.prob.toFixed(2) + ')' : '') +
      (score != null ? ' · score: ' + score.toFixed(2) : '') +
      (c.susp ? ' ⚠ suspicious — see evidence in zoom' : '');
    return `<div class="cellbox ${S.sel.has(c.filepath)?'sel':''}" data-i="${i}" title="${tip}">` +
      `<img loading="lazy" src="${imgURL(c.filepath,S.cellPx)}" style="width:${S.cellPx}px;height:${S.cellPx}px">` +
      (score != null ? `<span class="sugb">${score.toFixed(2)}</span>` : '') +
      (c.susp ? `<span class="susb">⚠${c.susp.susp.toFixed(2)}</span>` :
        (c.cert != null ? `<span class="cert">${c.cert.toFixed(2)}</span>` : '')) +
      `<div class="dots">${dots}</div></div>`;
  }).join('');
  document.querySelectorAll('#grid .cellbox').forEach(el => {
    const c = S.queue[+el.dataset.i];
    el.onclick = (e) => {
      // Shift+click = instant explicit negative for the checked targets on
      // this one cell — the per-cell fast path next to select + Apply −.
      if (e.shiftKey) { applyNegOne(c); return; }
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
  if (c.pred) bits.push(`<span class="zo">pred: ${esc(c.pred.class)} ${c.pred.prob.toFixed(2)}</span>`);
  const zscore = (c.score != null) ? c.score
               : ((c.suggest||[]).length ? c.suggest[0].score : null);
  if (zscore != null) bits.push(`<span class="zg">score ${zscore.toFixed(2)}</span>`);
  if (c.cert != null) bits.push(`<span class="zo">cert ${c.cert.toFixed(2)}</span>`);
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
  await api('/api/annotate', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepath: c.filepath, label_id: lid, state }) });
  if (state === 'clear') delete c.labels[lid]; else c.labels[lid] = state;
  // Every write re-ranks: reload the queue with the new exemplars and show
  // whatever cell now sits at this slot (the just-labeled cell usually
  // leaves the Collect queue — label and the next one slides in).
  await refreshStats();
  await loadQueue(S.page, true);
  renderZoom();
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
  await api('/api/annotate_batch', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepaths:[...S.sel], label_ids: lids, state }) });
  S.queue.forEach(c => {
    if (!S.sel.has(c.filepath)) return;
    for (const lid of lids) {
      if (state === 'clear') delete c.labels[lid]; else c.labels[lid] = state;
    }
  });
  toast(`${state === 1 ? 'Applied +' : state === 0 ? 'Applied −' : 'Removed'} ` +
        `${lids.length} label(s) on ${n} cells`);
  S.sel.clear();
  await refreshStats();
  loadQueue(S.page);   // memberships changed — recompute the page
}
$('#btn-apply-pos').onclick = () => applyTargets(1);
$('#btn-apply-neg').onclick = () => applyTargets(0);
$('#btn-remove').onclick = () => applyTargets('clear');
// Shift+click on one thumbnail: the same write as Apply − but for a single
// cell and without touching the selection. The write re-ranks, so the
// queue reloads and the (now decided) cell is replaced by its successor.
async function applyNegOne(c) {
  if (!S.targets.size) { toast('Check a target label first'); return; }
  const lids = [...S.targets];
  await api('/api/annotate_batch', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ filepaths:[c.filepath], label_ids: lids,
                           state: 0 }) });
  for (const lid of lids) c.labels[lid] = 0;
  toast(`Applied − ${lids.length} label(s)`);
  await refreshStats();
  loadQueue(S.page, true);
}
// ---- cluster-assisted bulk labeling ----------------------------------------
// The Clusters tab lists the Leiden clusters over the shared embedding space
// (largest first), each card showing its medoid — the cluster's most typical
// cell. Click a card: the queue narrows to that cluster (composed with the
// scope radios), the Medoid ↓ sort leads with the most typical members, and
// the bar above the grid writes the checked target labels to the WHOLE
// cluster ∩ scope in one undoable step — ~one decision per cluster instead
// of one per cell.
async function loadClusters() {
  // Counts follow the CURRENT queue context (label + scope + source) so a
  // card never promises cells the filters hide.
  const q = new URLSearchParams();
  if (S.labelFilter != null) { q.set('label_id', S.labelFilter); q.set('scope', S.scope); }
  if (S.source) q.set('source', S.source);
  const j = await api('/api/clusters' + (q.toString() ? '?' + q.toString() : ''));
  S.clusters = j.clusters; S.clusterRes = j.res;
  renderClusters();
}
function renderClusters() {
  $('#clu-meta').textContent =
    `${S.clusters.length} · res ${S.clusterRes != null ? S.clusterRes.toFixed(2) : '?'}`;
  $('#cluster-list').innerHTML = S.clusters.map(c =>
    `<div class="clu-card ${S.cluster===c.id?'active':''}" data-cid="${c.id}">` +
    (c.medoid_filepath ? `<img loading="lazy" src="${imgURL(c.medoid_filepath,64)}">` : '') +
    `<div class="clu-info"><b>#${c.id}</b>` +
    `<span class="clu-size">${c.size} cells</span>` +
    (c.undecided != null ? `<span class="clu-und">${c.undecided} to label</span>` : '') +
    (c.in_view != null && c.in_view !== c.size
      ? `<span class="clu-view${c.in_view === 0 ? ' zero' : ''}">${c.in_view} in view</span>` : '') +
    `</div></div>`).join('') || '<div style="color:#999">No clusters.</div>';
  document.querySelectorAll('.clu-card').forEach(el =>
    el.onclick = () => selectCluster(+el.dataset.cid));
}
function selectCluster(cid) {
  S.cluster = (S.cluster === cid) ? null : cid;
  // Entering a cluster leads with its most typical members (Medoid ↓);
  // leaving falls back to the plain Best order.
  S.sort = (S.cluster != null) ? 'medoid' : 'desc';
  S.shuffle = 0;
  renderClusters(); syncQueueControls();
  loadQueue(0);   // clears the selection (a context change) and re-renders the bar
}
function clusterNeighbor(dir) {
  // Prev/next walk the same largest-first order the cards render in.
  const i = S.clusters.findIndex(c => c.id === S.cluster);
  const j = i + dir;
  return (i >= 0 && j >= 0 && j < S.clusters.length) ? S.clusters[j].id : null;
}
function renderClusterBar() {
  const bar = $('#cluster-bar');
  if (S.cluster == null) { bar.style.display = 'none'; return; }
  bar.style.display = 'flex';
  const cur = S.clusters.find(c => c.id === S.cluster);
  $('#clu-title').innerHTML =
    `Cluster <b>#${S.cluster}</b> · <b>${S.queueTotal}</b> in view` +
    (S.scope === 'undecided' ? ' (undecided)' : '') +
    (cur && cur.size !== S.queueTotal ? ` · ${cur.size} in cluster` : '');
  $('#clu-prev').disabled = clusterNeighbor(-1) == null;
  $('#clu-next').disabled = clusterNeighbor(1) == null;
}
$('#clu-exit').onclick = () => selectCluster(null);
$('#clu-prev').onclick = () => { const p = clusterNeighbor(-1); if (p != null) selectCluster(p); };
$('#clu-next').onclick = () => { const n = clusterNeighbor(1); if (n != null) selectCluster(n); };
async function applyCluster(state) {
  if (S.cluster == null) return;
  if (S.labelFilter == null) { toast('Click a label first (it defines the queue scope)'); return; }
  if (!S.targets.size) { toast('Check at least one target label first'); return; }
  await api('/api/annotate_cluster', { method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ cluster_id: S.cluster, label_ids: [...S.targets],
                           state, label_id: S.labelFilter, scope: S.scope,
                           source: S.source || null }) });
  toast(`${state === 1 ? 'Applied +' : state === 0 ? 'Applied −' : 'Removed'} ` +
        `${S.targets.size} label(s) on cluster #${S.cluster} (${S.scope}) — Ctrl+Z reverts the whole cluster`);
  await refreshStats();
  loadQueue(S.page);
}
$('#clu-pos').onclick = () => applyCluster(1);
$('#clu-neg').onclick = () => applyCluster(0);
$('#clu-clear').onclick = () => applyCluster('clear');

// ---- sidebar tabs -----------------------------------------------------------
function setSideTab(t) {
  S.sideTab = t;
  $('#tab-labels').classList.toggle('on', t === 'labels');
  $('#tab-clusters').classList.toggle('on', t === 'clusters');
  $('#side-labels').style.display = t === 'labels' ? 'flex' : 'none';
  $('#side-clusters').style.display = t === 'clusters' ? 'flex' : 'none';
}
$('#tab-labels').onclick = () => setSideTab('labels');
$('#tab-clusters').onclick = () => setSideTab('clusters');

$('#btn-refit').onclick = async () => {
  const j = await api('/api/refresh_model', { method:'POST' });
  const bits = [];
  if (j.fitted.length) bits.push(`refit ${j.fitted.length} label(s)`);
  if (j.skipped.length)
    bits.push(`skipped ${j.skipped.length} (need ≥${j.min_pos} positives ` +
              `and ≥${j.min_neg} negatives)`);
  toast(bits.join(' · ') || 'Nothing to refit yet');
  await refreshStats();
  loadQueue(S.page);   // re-rank with the new scorers
};
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
  if (v >= 48 && v <= 512) {
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
    S.cluster = null;   // cluster cards reload with the new DB's counts
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
$('#source').onchange = () => {
  S.source = $('#source').value;
  if (S.clusterEnabled) loadClusters();   // per-card in-view counts follow
  loadQueue();
};
// Classify-bundle prediction filter: '' = all classes. A queue-context
// change, so the selection is dropped (loadQueue without keepSel).
$('#pred-sel').onchange = () => {
  S.predLabel = $('#pred-sel').value;
  loadQueue(0);
};

// ---- scope / sort radios + Shuffle (top bar row 2) -------------------------
// Reflect S.scope/S.sort in the radio groups; options that need the model
// (the score-defined To-label queue, the Suspicious check) are enabled
// only with one — without a model the undecided scope falls back to
// Positive.
function syncQueueControls() {
  if (!S.hasModel && S.scope === 'undecided') S.scope = 'with';
  document.querySelectorAll('#scope-radios input').forEach(r => {
    r.checked = r.value === S.scope;
    r.disabled = !S.hasModel && r.value === 'undecided';
  });
  // The cross-cutting sorts appear only where they mean anything:
  // Medoid ↓ inside a cluster, Prob ↓ with a classify bundle.
  $('#sort-medoid-wrap').style.display =
    (S.clusterEnabled && S.cluster != null) ? '' : 'none';
  $('#sort-prob-wrap').style.display = S.classNames.length ? '' : 'none';
  document.querySelectorAll('#sort-radios input').forEach(r => {
    r.checked = r.value === S.sort;
    r.disabled = (!S.hasModel && r.value === 'review')
              || (r.value === 'medoid' && !(S.clusterEnabled && S.cluster != null))
              || (r.value === 'prob' && !S.classNames.length);
  });
  $('#qt-shuffle').style.display = S.scope === 'undecided' ? '' : 'none';
}
document.querySelectorAll('#scope-radios input').forEach(r => r.onchange = () => {
  if (!r.checked) return;
  S.scope = r.value;
  S.shuffle = 0;   // a scope switch restarts the queue's own order
  syncQueueControls();
  if (S.clusterEnabled) loadClusters();   // per-card in-view counts follow
  loadQueue(0);
});
document.querySelectorAll('#sort-radios input').forEach(r => r.onchange = () => {
  if (!r.checked) return;
  S.sort = r.value;
  loadQueue(0);
});
$('#qt-shuffle').onclick = () => { S.shuffle += 1; loadQueue(0); };

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
  if (v >= 48 && v <= 512) { S.cellPx = v; localStorage.setItem('label_cellsize', String(v)); render(); }
}));

// Startup: refreshStats wires the queue controls and loads the queue
// (with no label selected yet it just prompts for one).
refreshStats().then(() => loadQueue(0));
</script>
</body>
</html>
"""
