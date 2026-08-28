from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from microBase import ImageDataset, SessionFile

from microProfiler.config import config_to_dict, load_config, PipelineConfig, resolve_inference_db
from microProfiler.io import Database
from microProfiler.log_utils import set_default_logging_level, setup_logging
from microProfiler.pipeline import MetadataValidationError, apply_filters, run_pipeline
from microProfiler.pipeline._micromodel_bridge import INFERENCE_TABLE, REDUCTION_TABLES

logger = logging.getLogger(__name__)


def resolve_datasets(dataset_dir: Path, pattern: str) -> list[Path]:
    """Resolve one or more dataset directories.

    If dataset_dir itself matches the pattern, return it as a single dataset.
    Otherwise, find matching subdirectories under dataset_dir.
    """
    compiled = re.compile(pattern)
    if dataset_dir.is_dir() and compiled.match(dataset_dir.name):
        return [dataset_dir]
    if not dataset_dir.is_dir():
        logger.warning(
            "Dataset directory does not exist (or is not a directory): %s",
            dataset_dir,
        )
        return []
    results = sorted([
        entry for entry in dataset_dir.iterdir()
        if entry.is_dir() and compiled.match(entry.name)
    ])
    return results


def _is_dataset_complete(cfg: PipelineConfig, dataset_dir: Path) -> bool:
    """Check if result.db exists and has all expected profiling tables.

    Table existence is required, and the `image` table must additionally
    contain one row per dataset image — a partial table (e.g. from an
    interrupted or failed run) must not be treated as complete.
    When inference is enabled, each block's output DB must also exist with
    its `inference` table (plus the reduction tables when reduction is on) —
    table existence only, no row-count guard (masks can change between runs).
    """
    db_path = dataset_dir / "result.db"
    if not db_path.exists():
        return False

    expected: set[str] = set()

    if cfg.image_profile and cfg.image_profile.run and cfg.image_profile.image_channels:
        # A section with empty image_channels is skipped entirely at runtime —
        # its table is not expected.
        expected.add("image")

    if cfg.object_profile and cfg.object_profile.run:
        for entry in cfg.object_profile.configs:
            if not entry.intensity_channels:
                # An entry with empty intensity_channels is skipped entirely
                # (shape features included) — its table is not expected.
                continue
            name = entry.output_table_name or entry.mask_name
            if name:
                expected.add(name)

    if not expected:
        return False

    count: int | None = None
    db = Database(db_path)
    try:
        existing = db.list_tables()
        missing = expected - existing
        if missing:
            return False
        if "image" in expected:
            count = db.row_count("image")
    except Exception:
        return False
    finally:
        db.close()

    if "image" in expected:
        try:
            ds = ImageDataset(
                root=dataset_dir,
                image_pattern=cfg.image_pattern,
                mask_pattern=cfg.mask_pattern,
                image_subdir_pattern=cfg.image_subdir_pattern,
            )
            # The pipeline applies cfg.filter before profiling, so the
            # image table only holds rows matching the filters — count
            # the same way or filtered datasets would never be skipped.
            apply_filters(ds, cfg.filter or [])
            n_rows = len(ds)
        except (Exception, SystemExit) as e:
            # microBase hard-exits (sys.exit) on bad config/dataset
            # (missing root, absent filter column, bad regex) — convert
            # to "not complete" so the batch logs and continues instead
            # of aborting the whole plate scan.
            logger.warning("Could not count images for %s: %s", dataset_dir, e)
            return False
        if count != n_rows:
            logger.warning(
                "Incomplete image table in %s: %d rows for %d images",
                dataset_dir, count, n_rows,
            )
            return False
    if cfg.inference and cfg.inference.run and cfg.inference.configs:
        for entry in cfg.inference.configs:
            if not entry.channels:
                # A block with no channels is skipped at runtime — its DB is
                # not expected.
                continue
            infer_db = dataset_dir / resolve_inference_db(entry)
            tables = {INFERENCE_TABLE}
            if entry.reduction and entry.reduction.enabled:
                tables |= REDUCTION_TABLES
            missing = tables - _existing_tables(infer_db)
            if missing:
                logger.info(
                    "Incomplete inference DB %s for %s (missing %s) — reprocessing",
                    infer_db, dataset_dir, sorted(missing),
                )
                return False

    return True


def _existing_tables(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    db = Database(db_path)
    try:
        return db.list_tables()
    except Exception:
        return set()
    finally:
        db.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microProfiler",
        description="Microscopy image preprocessing, segmentation, and profiling pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable DEBUG-level logging (verbose output)",
    )
    parser.add_argument(
        "--log-file", type=Path, default=None,
        help="Log output to file",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser(
        "run", help="Run the full analysis pipeline on one or more datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument("--config", "-c", type=Path, required=True, help="YAML config file")
    run_parser.add_argument(
        "--dataset-dir", type=Path, default=Path("."),
        help="Dataset directory (single dataset) or parent directory containing multiple datasets",
    )
    run_parser.add_argument(
        "--dataset-pattern", type=str, default=r".*\d-Measurement \d",
        help="Regex pattern to identify dataset directories under --dataset-dir (ignored if --dataset-dir is itself a dataset)",
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Discover datasets and report status without processing",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    set_default_logging_level(log_level)
    log = setup_logging(level=log_level, log_file=args.log_file)
    log.debug("Debug logging enabled")

    if args.command == "run":
        try:
            cfg = load_config(args.config)
        except ValueError as e:
            # Unknown config keys / invalid values — clean hard-exit, not a
            # raw traceback.
            print(f"Error: invalid config {args.config}: {e}", file=sys.stderr)
            sys.exit(1)
        dataset_dir = Path(args.dataset_dir)
        dataset_pattern = args.dataset_pattern

        # A config that requests inference must have its tooling present —
        # silently skipping would fake a completed run (§2 no-guessing).
        if cfg.inference and cfg.inference.run and cfg.inference.configs:
            import importlib.util
            if importlib.util.find_spec("microModel") is None:
                print(
                    "Error: inference is enabled in the config but the "
                    "'microModel' package is not installed. Install microModel "
                    "or disable the 'inference' section.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # Duplicate output_db names would silently overwrite each other's
            # inference rows — the GUI's validate_blocks rejects these, so the
            # CLI must too.
            seen_dbs: dict = {}
            for entry in cfg.inference.configs:
                if not entry.channels:
                    continue
                db_name = resolve_inference_db(entry)
                if db_name in seen_dbs:
                    print(
                        f"Error: duplicate inference output_db '{db_name}' used "
                        f"by blocks '{seen_dbs[db_name]}' and '{entry.model}'. "
                        f"Rename one of them.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                seen_dbs[db_name] = entry.model

        datasets = resolve_datasets(dataset_dir, dataset_pattern)
        if not datasets:
            logger.warning(
                "No datasets found at %s matching pattern '%s'. "
                "Use --dataset-dir to point to a dataset directory, "
                "or to a parent directory whose subdirectories match --dataset-pattern.",
                dataset_dir, dataset_pattern,
            )
            return 1

        logger.info("Found %d dataset(s) at %s", len(datasets), dataset_dir)

        if args.dry_run:
            _dry_run(cfg, datasets, log)
            return 0

        processed = 0
        skipped = 0

        for dataset_dir in datasets:
            if _is_dataset_complete(cfg, dataset_dir):
                logger.info("Skipping %s — already processed (delete result.db to re-run)", dataset_dir)
                skipped += 1
                continue

            logger.info("Processing dataset: %s", dataset_dir)
            try:
                ds, applied_steps = run_pipeline(
                    cfg, root_dir=dataset_dir, log_file=args.log_file,
                )
                cfg_dict = config_to_dict(cfg)
                sf = SessionFile(dataset_dir)
                sf.save(cfg_dict)
                sf.set_applied_steps(applied_steps)
                logger.info("Dataset complete: %s", dataset_dir)
                processed += 1
                print()
            except MetadataValidationError as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
            except SystemExit as e:
                # microBase hard-exits (sys.exit) on bad dataset state (missing
                # files, invalid filter column, corrupt TIFF). Treat it as a
                # per-dataset failure so one bad dataset never aborts a plate
                # scan — the same policy _is_dataset_complete applies above.
                logger.error("Dataset failed: %s — %s", dataset_dir, e)
                logger.info("Continuing to next dataset...")
                print()
            except Exception as e:
                logger.error("Dataset failed: %s — %s", dataset_dir, e)
                logger.info("Continuing to next dataset...")
                print()

        logger.info(
            "Batch done: %d processed, %d skipped out of %d total",
            processed, skipped, len(datasets),
        )
        return 0

    return 1


def _dry_run(cfg: PipelineConfig, datasets: list[Path], log: logging.Logger) -> None:
    """Print dataset discovery + skip status without processing."""
    for i, dataset_dir in enumerate(datasets):
        complete = _is_dataset_complete(cfg, dataset_dir)
        status = "SKIP (already processed)" if complete else "WOULD PROCESS"

        if i < 5:
            try:
                ds = ImageDataset(
                    root=dataset_dir,
                    image_pattern=cfg.image_pattern,
                    mask_pattern=cfg.mask_pattern,
                    image_subdir_pattern=cfg.image_subdir_pattern,
                )
                info = (
                    f"rows={len(ds)}, "
                    f"channels={ds.intensity_colnames}, "
                    f"masks={ds.mask_colnames}"
                )
            except Exception as e:
                info = f"load failed: {e}"
        else:
            info = "(load skipped — first 5 only)"

        log.info("[DRY RUN] %s → %s | %s", dataset_dir, status, info)

    log.info("Dry run complete: %d dataset(s) found", len(datasets))

if __name__ == "__main__":
    sys.exit(main())
