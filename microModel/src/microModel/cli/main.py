"""CLI entry point: micromodel pretrain / train / infer / deduplication / label / augment-vis / reduction / reduction-vis / attention-vis."""

import argparse
import sys
import time

from .. import __version__
from ..utils import setup_logging, load_yaml, logger
from ..pretrain import run_pretrain
from ..attention_vis import run_attention_vis
from ..train import run_train
from ..infer import run_inference
from ..deduplication import run_deduplication
from ..label import run_label
from ..reduction import run_reduction
from ..augment_vis import run_augment_vis
from ..reduction_vis import main as main_interactive

from microBase import MicroMaxError


def cmd_pretrain(args):
    logger.info("Loading pretrain config from %s", args.config)
    cfg = load_yaml(args.config)
    run_pretrain(cfg, config_path=args.config)


def cmd_attention_vis(args):
    logger.info("Loading pretrain config from %s for attention visualization", args.config)
    cfg = load_yaml(args.config)
    run_attention_vis(cfg, config_path=args.config)


def cmd_train(args):
    logger.info("Loading train config from %s", args.config)
    cfg = load_yaml(args.config)
    run_train(cfg, config_path=args.config)


def cmd_infer(args):
    logger.info("Loading inference config from %s", args.config)
    cfg = load_yaml(args.config)
    run_inference(cfg, config_path=args.config)


def cmd_deduplication(args):
    logger.info("Loading deduplication config from %s", args.config)
    cfg = load_yaml(args.config)
    run_deduplication(cfg, config_path=args.config)


def cmd_label(args):
    logger.info("Loading label config from %s", args.config)
    cfg = load_yaml(args.config)
    run_label(cfg, config_path=args.config, port=args.port,
                 no_browser=args.no_browser)


def cmd_augment_vis(args):
    logger.info("Loading config from %s for augmentation preview", args.config)
    cfg = load_yaml(args.config)
    run_augment_vis(cfg)


def cmd_reduction(args):
    logger.info("Loading inference config from %s for reduction view", args.config)
    cfg = load_yaml(args.config)
    run_reduction(cfg)


def cmd_reduction_vis(args):
    logger.info("Loading inference config from %s for interactive reduction view", args.config)
    cfg = load_yaml(args.config)
    main_interactive(cfg, port=args.port)


def main():
    parser = argparse.ArgumentParser(
        prog="micromodel",
        description="Self-supervised learning, finetuning, and feature extraction for multi-channel fluorescent images")
    parser.add_argument("--version", action="version", version=f"microModel {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_pretrain = sub.add_parser("pretrain", help="SSL pretrain (DINOv3)")
    p_pretrain.add_argument("--config", required=True, help="Path to pretrain YAML config")
    p_pretrain.set_defaults(func=cmd_pretrain)

    p_train = sub.add_parser("train", help="Classification training from SSL backbone or scratch")
    p_train.add_argument("--config", required=True, help="Path to train YAML config")
    p_train.set_defaults(func=cmd_train)

    p_infer = sub.add_parser("infer", help="Inference + feature extraction -> infer.db (mode from bundle: SSL = features, train = classify + features)")
    p_infer.add_argument("--config", required=True, help="Path to inference YAML config")
    p_infer.set_defaults(func=cmd_infer)

    p_cur = sub.add_parser("deduplication",
                           help="Diversity-preserving selection over cropped single-cell folders: teacher-branch features + radius-coverage greedy (prune redundancy / pick diverse new data) -> selection_state.pkl + manifest + curated/ hardlinks")
    p_cur.add_argument("--config", required=True, help="Path to deduplication YAML config")
    p_cur.set_defaults(func=cmd_deduplication)

    p_ann = sub.add_parser("label",
                           help="Interactive multi-label annotation web server (Flask) with exemplar-similarity recommendation scores: SSL/train bundle -> kNN (+ classify head) ranking in Collect, explicit negatives push lookalikes down, Collect/Manage queues, per-action undo -> label_multiple.db / label_single.db + per-mode label_export(_single).csv")
    p_ann.add_argument("--config", required=True, help="Path to label YAML config")
    p_ann.add_argument("--port", type=int, default=5000, help="Server port (default 5000)")
    p_ann.add_argument("--no-browser", action="store_true",
                       help="Do not open the browser automatically")
    p_ann.set_defaults(func=cmd_label)

    p_aug = sub.add_parser("augment-vis", help="Visualize multi-view augmentation preview (no model needed)")
    p_aug.add_argument("--config", required=True, help="Path to pretrain or train YAML config")
    p_aug.set_defaults(func=cmd_augment_vis)

    p_red = sub.add_parser("reduction",
                           help="Fit DR reductions (pca/umap/pacmap/localmap) on feature vectors + optional Leiden clustering; writes reduction_<method>/find_cluster tables and multi-page PDFs")
    p_red.add_argument("--config", required=True, help="Path to inference YAML config")
    p_red.set_defaults(func=cmd_reduction)

    p_int = sub.add_parser("reduction-vis", help="Launch interactive PCA/UMAP viewer with live image inspection")
    p_int.add_argument("--config", required=True, help="Path to inference YAML config")
    p_int.add_argument("--port", type=int, default=5000, help="Server port (default 5000)")
    p_int.set_defaults(func=cmd_reduction_vis)

    p_att = sub.add_parser("attention-vis",
                           help="Offline attention/patch-similarity visualization from a trained SSL bundle (needs resume.ssl_model in the config)")
    p_att.add_argument("--config", required=True, help="Path to pretrain YAML config (resume.ssl_model -> trained bundle)")
    p_att.set_defaults(func=cmd_attention_vis)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()
    t0 = time.perf_counter()
    logger.info("microModel %s: %s (config: %s)", __version__, args.command, args.config)
    try:
        args.func(args)
    except MicroMaxError as e:
        # Deliberate library errors: concise message + exit code, no traceback.
        logger.error("%s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logger.info("Finished %s in %.1fs", args.command, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
