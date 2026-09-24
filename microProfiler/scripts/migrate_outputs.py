"""One-shot migration: OLD microProfiler inference outputs -> new style.

Old style (per dataset):
    profiler.db + one raw microModel infer DB (``infer.db`` or a custom
    ``output_db`` name) + one ``merge_<mask>.db`` per mask.
New style:
    ``infer_<model_name>.db`` per model (one wide TABLE per mask,
    inference + reduction + find_cluster uid-merged) + one dataset-wide
    ``merged.db`` (profiler.db fused with every infer DB, per mask).

For every dataset directory the script
  1. finds the raw infer DB(s) — any ``*.db`` containing an ``inference``
     table (except profiler.db / merged.db),
  2. resolves a model name per raw DB (explicit --model-name, else the
     session.yml inference config that wrote it, else the file name),
  3. absorbs each raw DB into ``infer_<model_name>.db``
     (microBase.db_merge.absorb_infer_db),
  4. deletes the raw DBs and the ``merge_<mask>.db`` files,
  5. rebuilds ``merged.db`` (microBase.db_merge.build_merged_db),
  6. rewrites session.yml's inference section to the new keys
     (``model_name`` instead of ``output_db``) so the GUI pre-fills.

Run it (conda env `micro`, packages installed):

    python microProfiler/scripts/migrate_outputs.py --dataset "C:/path/to/dataset"
    python microProfiler/scripts/migrate_outputs.py --parent "C:/plate-root" [--dry-run]

``--parent`` migrates every subdirectory matching the dataset pattern
(default ``.*-Measurement <n>``). ``--dry-run`` only prints the plan.
The sources profiler.db and the raw infer DBs are read-only until step 4;
profiler.db is never modified or deleted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from microBase import SessionFile, save_yaml
from microBase.db_contracts import (
    INFER_DB_NAME,
    INFER_DB_STEM_PREFIX,
    MERGED_DB_NAME,
    PROFILER_DB_NAME,
    infer_db_file_name,
)
from microBase.db_merge import (
    absorb_infer_db,
    build_merged_db,
    table_names_of,
)

# Dataset-name pattern --parent matches subdirectories with (same default
# as `microprofiler run --dataset-pattern`).
DATASET_PATTERN = r".*\d-Measurement \d"


def suggest_model_name(model_path: str) -> str:
    """Model-name suggestion from a bundle path (same rule as the GUI).

    The file stem is 'model' for almost every microModel bundle, so the
    parent folder name carries the identity; a non-generic stem wins.
    """
    p = Path(model_path)
    stem = p.stem
    if stem and stem != "model":
        return stem
    return p.parent.name or stem or "model"


def _is_raw_infer_db(path: Path) -> bool:
    """True when the SQLite file holds a raw microModel `inference` table."""
    try:
        return "inference" in set(table_names_of(str(path)))
    except Exception:
        return False


def find_raw_infer_dbs(dataset_dir: Path) -> list[Path]:
    """Every raw infer DB of the dataset (old style keeps them in the root)."""
    return sorted(
        p for p in dataset_dir.glob("*.db")
        if p.name not in (PROFILER_DB_NAME, MERGED_DB_NAME)
        and not p.name.startswith(INFER_DB_STEM_PREFIX)
        and _is_raw_infer_db(p)
    )


def session_infer_configs(dataset_dir: Path) -> list[dict]:
    """The dataset's saved inference block configs ([] when absent)."""
    sf = SessionFile(dataset_dir)
    data = sf.load() or {}
    section = data.get("inference") or {}
    return list(section.get("configs") or [])


def resolve_model_names(dataset_dir: Path, raw_dbs: list[Path],
                        explicit: str | None) -> dict[Path, tuple[str, str | None]]:
    """(model_name, default_mask) per raw DB.

    Priority: the explicit --model-name (single-DB datasets — with several
    raw DBs it cannot tell them apart), then the session.yml inference
    config whose ``output_db`` matches the file name, then the file name
    itself (``infer_...`` prefix stripped).
    """
    out: dict[Path, tuple[str, str | None]] = {}
    if explicit:
        if len(raw_dbs) > 1:
            raise SystemExit(
                "--model-name applies to a single raw infer DB; this dataset "
                f"has {len(raw_dbs)}: {', '.join(p.name for p in raw_dbs)}")
        mask = None
        for cfg in session_infer_configs(dataset_dir):
            if cfg.get("mask_name"):
                mask = str(cfg["mask_name"])
                break
        return {raw_dbs[0]: (explicit, mask)}

    # session.yml mapping: old output_db file name -> (model name, mask).
    by_db: dict[str, tuple[str, str | None]] = {}
    for cfg in session_infer_configs(dataset_dir):
        db_name = str(cfg.get("output_db") or INFER_DB_NAME)
        model = cfg.get("model")
        name = suggest_model_name(str(model)) if model else None
        if name:
            by_db[db_name] = (name, cfg.get("mask_name") or None)

    for raw in raw_dbs:
        if raw.name in by_db:
            out[raw] = by_db[raw.name]
        else:
            # Filename fallback: strip the .db suffix and the infer_ prefix.
            out[raw] = (raw.name[:-3] if raw.name.endswith(".db")
                        else raw.name, None)
    return out


def migrate_dataset(dataset_dir: Path, model_name: str | None,
                    dry_run: bool) -> bool:
    """Migrate one dataset directory. Returns True when work was done."""
    print(f"\n=== {dataset_dir}")
    if not dataset_dir.is_dir():
        print("    not a directory — skipped")
        return False

    raw_dbs = find_raw_infer_dbs(dataset_dir)
    merge_dbs = sorted(dataset_dir.glob("merge_*.db"))
    merged_db = dataset_dir / MERGED_DB_NAME
    has_profiler = (dataset_dir / PROFILER_DB_NAME).exists()

    if not raw_dbs and not merge_dbs and not has_profiler:
        print("    no profiler.db and no raw infer DB — skipped")
        return False

    plans: list[tuple[Path, Path, str | None]] = []
    if raw_dbs:
        resolved = resolve_model_names(dataset_dir, raw_dbs, model_name)
        for raw in raw_dbs:
            name, mask = resolved[raw]
            plans.append((raw, dataset_dir / infer_db_file_name(name), mask))
            print(f"    absorb {raw.name} -> {infer_db_file_name(name)}"
                  f" (default mask: {mask or 'from rows'})")
    for m in merge_dbs:
        print(f"    delete  {m.name} (superseded by merged.db)")

    if dry_run:
        print(f"    build   {MERGED_DB_NAME}"
              + (" (exists — rebuild)" if merged_db.exists() else ""))
        print("    (dry run — nothing written)")
        return True

    for raw, out_db, mask in plans:
        absorb_infer_db(raw, out_db, default_mask=mask)
    for raw, _, _ in plans:
        raw.unlink()
        print(f"    deleted {raw.name}")
    for m in merge_dbs:
        m.unlink()
        print(f"    deleted {m.name}")

    if has_profiler or plans:
        out = build_merged_db(dataset_dir)
        print(f"    built   {Path(out).name}" if out else
              "    merged.db not buildable (no object tables)")

    rewrite_session_inference(dataset_dir, model_name)
    return True


def rewrite_session_inference(dataset_dir: Path,
                              model_name: str | None = None) -> None:
    """Swap session.yml's inference keys to the new style (model_name).

    The explicit --model-name wins over the bundle-path derivation, so the
    session names the SAME per-model DB the absorb step wrote. Written
    wholesale (not SessionFile.save — its deep merge never removes the old
    ``output_db`` keys).
    """
    sf = SessionFile(dataset_dir)
    data = sf.load()
    section = data.get("inference")
    if not isinstance(section, dict):
        return
    configs = section.get("configs")
    if not isinstance(configs, list):
        return
    changed = False
    for cfg in configs:
        if not isinstance(cfg, dict):
            continue
        old_db = cfg.pop("output_db", None)
        if old_db:
            # Explicit --model-name first; else derive from the bundle path
            # (the old db NAME, often just "infer.db", says nothing).
            model = cfg.get("model")
            cfg["model_name"] = (model_name
                                 or (suggest_model_name(str(model)) if model
                                     else Path(str(old_db)).stem))
            changed = True
        elif model_name and cfg.get("model_name") != model_name:
            # Already-rewritten session: an explicit --model-name aligns it
            # with the per-model DB the absorb step actually wrote.
            cfg["model_name"] = model_name
            changed = True
    if changed:
        save_yaml(sf.path, data)
        print("    session.yml inference section rewritten (model_name)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate old microProfiler inference outputs "
                    "(infer.db + merge_<mask>.db) to the new style "
                    "(infer_<model>.db per-mask tables + merged.db).")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--dataset", type=Path,
                        help="One dataset directory to migrate")
    target.add_argument("--parent", type=Path,
                        help="Parent directory whose subdirectories are "
                             "migrated (matched against --dataset-pattern)")
    parser.add_argument("--dataset-pattern", type=str,
                        default=DATASET_PATTERN,
                        help="Subdirectory pattern for --parent "
                             "(default: %(default)r)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Explicit model name for a single raw infer DB "
                             "(default: from session.yml / file name)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan without writing anything")
    args = parser.parse_args(argv)

    datasets = []
    if args.dataset:
        datasets = [args.dataset]
    else:
        root = args.parent
        if not root.is_dir():
            print(f"Error: {root} is not a directory", file=sys.stderr)
            return 1
        compiled = re.compile(args.dataset_pattern)
        datasets = sorted(p for p in root.iterdir()
                          if p.is_dir() and compiled.match(p.name))
        if not datasets:
            print(f"No subdirectories of {root} match "
                  f"'{args.dataset_pattern}'", file=sys.stderr)
            return 1

    migrated = 0
    for ds in datasets:
        if migrate_dataset(ds, args.model_name, args.dry_run):
            migrated += 1
    print(f"\nDone: {migrated}/{len(datasets)} dataset(s) "
          f"{'planned' if args.dry_run else 'migrated'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
