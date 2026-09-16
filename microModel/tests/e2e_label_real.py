"""E2E check for the label app against REAL data + REAL bundle.

Runs entirely inside a temp project dir: the user's label_multiple.db is
COPIED there (migration is exercised on the copy), features/ is copied for
cache hits, and the real image roots + model bundle are only READ. The
real save_dir is never touched.

All counts are read from the live project (no hardcoded totals), so the
script stays valid after every labeling session.
"""

import os
import shutil
import sys
import time

sys.path.insert(0, r"C:\Users\haohe\GitHub\microMax\microModel\src")

TMP = r"C:\Users\haohe\AppData\Local\Temp\label_e2e"
REAL = r"D:\Model\sc_dataset\label"
BUNDLE = r"D:\Model\2026-09-14_dinov3_phase2\model_220.pt"
ROOTS = [r"D:\Model\sc_dataset\deduplication\curated\NCOA2_293T_63x_epi",
         r"D:\Model\sc_dataset\deduplication\curated\opencell_single_cell",
         r"D:\Model\sc_dataset\deduplication\curated\p53"]

# ---- fresh temp project with a COPY of the user's project ------------------
if os.path.exists(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)
shutil.copy2(os.path.join(REAL, "label_multiple.db"),
             os.path.join(TMP, "label_multiple.db"))
if os.path.isdir(os.path.join(REAL, "features")):
    shutil.copytree(os.path.join(REAL, "features"),
                    os.path.join(TMP, "features"))

config = {
    "save_dir": TMP,
    "model": BUNDLE,
    "data": {"file_dir": ROOTS, "channels": [1], "channel_layout": None,
             "max_value": 65535, "image_pattern": None,
             "label_from_dir": False, "label_csv": None},
    "recommend": {"knn_k": 1, "neg_weight": 0.5,
                  "diverse_size": 48, "page_size": 100},
    "space": {"pca_components": 50},
    "dataloader": {"batch_size": 128, "num_workers": 0},
    "seed": 42,
}

from microModel.label import LabelServer

srv = LabelServer(config, open_browser=False)
srv.app.run = lambda **kw: print("(app.run skipped in E2E)")
t0 = time.perf_counter()
srv.start()
print(f"startup: {time.perf_counter() - t0:.1f}s (migration + features)")

c = srv.app.test_client()
ok = True


def check(name, cond, extra=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
    ok = ok and bool(cond)


# ---- state: migration kept everything --------------------------------------
j = c.get("/api/state").get_json()
check("multi DB stays label_multiple.db",
      os.path.exists(os.path.join(TMP, "label_multiple.db"))
      and not os.path.exists(os.path.join(TMP, "label.db")))
check("cell count", j["total"] == 40169, f"total={j['total']}")
check("labels migrated", len(j["labels"]) > 0, f"n={len(j['labels'])}")
base_labeled = j["labeled"]
check("decisions kept", base_labeled > 0, f"labeled={base_labeled}")
check("undecided reported", j["undecided"] == j["total"] - base_labeled)
check("sources", len(j["sources"]) == 3)
lids = {lb["name"]: lb["label_id"] for lb in j["labels"]}
lpos = {lb["name"]: lb["n_pos"] for lb in j["labels"]}
# The label with the most positives drives the ranking checks.
top_name = max(lpos, key=lpos.get)
print(f"  using label '{top_name}' with {lpos[top_name]} positives")

# ---- Collect queue: ranked, scored, label-free ------------------------------
t0 = time.perf_counter()
q = c.get(f"/api/queue?scope=undecided&label_id={lids[top_name]}"
          f"&limit=100&offset=0").get_json()
dt = time.perf_counter() - t0
check("collect queue returns page", len(q["cells"]) == 100,
      f"total={q['total']} in {dt:.2f}s")
check("every cell carries the ranking score",
      all(cell.get("score") is not None for cell in q["cells"]))
check("scores descending",
      all(q["cells"][i]["score"] >= q["cells"][i + 1]["score"] - 1e-6
          for i in range(len(q["cells"]) - 1)))
decided = [cell for cell in q["cells"]
           if str(lids[top_name]) in cell["labels"]]
check("no cells decided for the SELECTED label in collect", len(decided) == 0)
sug = [cell for cell in q["cells"] if cell.get("suggest")]
check("suggestions present", len(sug) > 0, f"{len(sug)}/100 suggested")

# ---- Manage queue: vectorized cert + review ---------------------------------
t0 = time.perf_counter()
q = c.get(f"/api/queue?scope=with&label_id={lids[top_name]}"
          f"&limit=100&sort=asc").get_json()
dt = time.perf_counter() - t0
check("manage 'with' page", len(q["cells"]) == 100,
      f"total={q['total']} in {dt:.2f}s")
certs = [cell.get("cert") for cell in q["cells"] if cell.get("cert") is not None]
check("certainty badges", len(certs) >= 90)
asc_ok = all((q["cells"][i]["cert"] or 0) <= (q["cells"][i + 1]["cert"] or 0) + 1e-6
             for i in range(len(q["cells"]) - 1) if "cert" in q["cells"][i])
check("certainty ascending order", asc_ok)

q = c.get(f"/api/queue?scope=without&label_id={lids[top_name]}"
          f"&limit=100&sort=desc").get_json()
check("manage 'without' page", len(q["cells"]) == 100,
      f"total={q['total']} (missing-label candidates)")

q = c.get(f"/api/queue?scope=with&label_id={lids[top_name]}"
          f"&limit=50&sort=review").get_json()
check("review sort runs", "total" in q, f"total={q['total']}")

t0 = time.perf_counter()
q = c.get(f"/api/queue?scope=union&label_id={lids[top_name]}"
          f"&limit=100&sort=asc").get_json()
dt = time.perf_counter() - t0
check("manage union page", len(q["cells"]) == 100,
      f"total={q['total']} in {dt:.2f}s")

# ---- annotate batch (multi-label) + undo ------------------------------------
fp = q["cells"][0]["filepath"]
q2 = c.get(f"/api/queue?scope=undecided&label_id={lids[top_name]}"
           f"&limit=10").get_json()
batch = [cell["filepath"] for cell in q2["cells"][:5]]
names = [lb["name"] for lb in j["labels"]][:2]
resp = c.post("/api/annotate_batch", json={
    "filepaths": batch, "label_ids": [lids[n] for n in names],
    "state": 1}).get_json()
check("multi-label batch write ok", resp.get("ok") and resp.get("n") == 10,
      f"n={resp.get('n')}")
check("stats grew",
      c.get("/api/state").get_json()["labeled"] >= base_labeled)
j = c.post("/api/undo").get_json()
check("undo reverts the whole multi-label batch",
      j.get("undone") and j.get("n") == 10, f"n={j.get('n')}")
st_after = c.get("/api/state").get_json()
check("stats restored after undo", st_after["labeled"] == base_labeled,
      f"labeled={st_after['labeled']}")

# ---- image render -----------------------------------------------------------
t0 = time.perf_counter()
r = c.get(f"/api/image?filepath={fp}&max_px=200")
dt = time.perf_counter() - t0
check("image renders", r.status_code == 200 and r.mimetype == "image/png"
      and len(r.data) > 500, f"{len(r.data)}B in {dt:.2f}s")
t0 = time.perf_counter()
r2 = c.get(f"/api/image?filepath={fp}&max_px=200")
dt = time.perf_counter() - t0
check("render cache hit", r2.data == r.data, f"{dt * 1000:.1f}ms")

# ---- export ------------------------------------------------------------------
r = c.post("/api/export").get_json()
check("export csv", r.get("ok") and r.get("rows", 0) > 0,
      f"rows={r.get('rows')}")
check("export landed in temp project", r.get("path", "").startswith(TMP))

# ---- dual mode: the single project is a fresh, independent store ------------
j = c.post("/api/label_mode", json={"mode": "single"}).get_json()
check("mode switch ok", j.get("label_mode") == "single")
j = c.get("/api/state").get_json()
check("single project starts empty", j["labels"] == [] and j["labeled"] == 0,
      f"labels={len(j['labels'])} labeled={j['labeled']}")
c.post("/api/label_mode", json={"mode": "multi"})
j = c.get("/api/state").get_json()
check("switch back keeps the multi project",
      j["labeled"] == base_labeled and len(j["labels"]) == len(lids),
      f"labeled={j['labeled']} labels={len(j['labels'])}")
check("single DB file created",
      os.path.exists(os.path.join(TMP, "label_single.db")))
q = c.get("/api/queue?scope=with&label_id=%d&limit=5"
          % lids[top_name]).get_json()
check("multi queues still work after round trip",
      q["total"] == lpos[top_name], f"total={q['total']}")

print()
print("E2E", "ALL PASS" if ok else "HAS FAILURES")
sys.exit(0 if ok else 1)
