"""CLI entry point: micromodel pretrain / train / infer / vis-augment / vis-reduction / vis-reduction-interactive / vis-attention."""

import argparse
import sys
import time

from .. import __version__
from ..utils import setup_logging, load_yaml, logger
from ..pretrain import pretrain_ssl, vis_attention
from ..train import train
from ..infer import run_inference
from ..vis import show_augmentation, show_reduction
from ..vis_interactive import main as main_interactive


def cmd_pretrain(args):
    logger.info("Loading pretrain config from %s", args.config)
    cfg = load_yaml(args.config)
    pretrain_ssl(cfg, config_path=args.config)


def cmd_vis_attention(args):
    logger.info("Loading pretrain config from %s for attention visualization", args.config)
    cfg = load_yaml(args.config)
    vis_attention(cfg, config_path=args.config)


def cmd_train(args):
    logger.info("Loading train config from %s", args.config)
    cfg = load_yaml(args.config)
    train(cfg, config_path=args.config)


def cmd_infer(args):
    logger.info("Loading inference config from %s", args.config)
    cfg = load_yaml(args.config)
    run_inference(cfg, config_path=args.config)


def cmd_vis_augment(args):
    logger.info("Loading config from %s for augmentation preview", args.config)
    cfg = load_yaml(args.config)
    show_augmentation(cfg)


def cmd_vis_reduction(args):
    logger.info("Loading inference config from %s for reduction view", args.config)
    cfg = load_yaml(args.config)
    show_reduction(cfg)


def cmd_vis_reduction_interactive(args):
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

    p_aug = sub.add_parser("vis-augment", help="Visualize multi-view augmentation preview (no model needed)")
    p_aug.add_argument("--config", required=True, help="Path to pretrain or train YAML config")
    p_aug.set_defaults(func=cmd_vis_augment)

    p_red = sub.add_parser("vis-reduction",
                           help="Fit DR reductions (pca/umap/pacmap/localmap) on feature vectors + optional Leiden clustering; writes reduction_<method>/find_cluster tables and multi-page PDFs")
    p_red.add_argument("--config", required=True, help="Path to inference YAML config")
    p_red.set_defaults(func=cmd_vis_reduction)

    p_int = sub.add_parser("vis-reduction-interactive", help="Launch interactive PCA/UMAP viewer with live image inspection")
    p_int.add_argument("--config", required=True, help="Path to inference YAML config")
    p_int.add_argument("--port", type=int, default=5000, help="Server port (default 5000)")
    p_int.set_defaults(func=cmd_vis_reduction_interactive)

    p_att = sub.add_parser("vis-attention",
                           help="Offline attention/patch-similarity visualization from a trained SSL bundle (needs resume.ssl_model in the config)")
    p_att.add_argument("--config", required=True, help="Path to pretrain YAML config (resume.ssl_model -> trained bundle)")
    p_att.set_defaults(func=cmd_vis_attention)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    setup_logging()
    t0 = time.perf_counter()
    logger.info("microModel %s: %s (config: %s)", __version__, args.command, args.config)
    try:
        args.func(args)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logger.info("Finished %s in %.1fs", args.command, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
