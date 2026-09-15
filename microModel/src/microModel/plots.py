"""Training / diagnostic plots shared by pretrain, train and attention-vis.

No CLI of its own: plot_training_results serves the train loop,
plot_pretrain_loss / plot_head_track feed the pretrain loop's per-epoch
monitoring, and the patch-similarity / attention-map renderers back both the
in-training summary and the offline attention-vis command.
"""

import numpy as np
import torch
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

from .utils import logger


# ----------------------------------------------------------------------------
# Training plots (for train)
# ----------------------------------------------------------------------------

def plot_training_results(model, device, val_loader, num_classes, label_to_idx,
                          train_loss_history, val_acc_history, save_path=None,
                          multi_label=False):
    """Loss/accuracy history + a third panel that depends on the label mode:
    single-label draws the normalized confusion matrix; multi-label draws a
    per-class F1 bar chart at the 0.5 threshold. Returns a dict with the
    final metrics plus "cm" (single-label) or "per_class_f1" (multi-label).
    """
    if not train_loss_history:
        logger.warning("No training history to plot.")
        return None

    # Imported here (not at module top) to keep vis import light for GUI users.
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    # x-axis is the 1-based epoch number (package-wide convention).
    ep = range(1, len(train_loss_history) + 1)
    axes[0].plot(ep, train_loss_history, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    axes[1].plot(range(1, len(val_acc_history) + 1), val_acc_history,
                 marker="s", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True)

    model.eval()
    yt_all, yp_all = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits, _ = model(x)
            if multi_label:
                yt_all.extend(y.long().tolist())
                yp_all.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
            else:
                yt_all.extend(y.tolist())
                yp_all.extend(logits.argmax(1).cpu().tolist())

    # Class order for the confusion-matrix axes follows the label_to_idx
    # mapping (sorted by index), not dict insertion order.
    cls_names = sorted(label_to_idx, key=label_to_idx.get)
    if multi_label:
        # No confusion matrix exists for multi-label — bar-chart the F1 of
        # every category instead.
        per_class = f1_score(yt_all, yp_all, average=None,
                             zero_division=0) if yt_all else np.zeros(num_classes)
        per_class_f1 = {cls_names[i]: float(per_class[i])
                        for i in range(min(num_classes, len(per_class)))}
        axes[2].bar(range(num_classes), per_class[:num_classes], color="steelblue")
        axes[2].set_xticks(range(num_classes), cls_names, rotation=45, ha="right")
        axes[2].set_ylabel("F1")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Per-class F1 (threshold 0.5)")
        axes[2].grid(True, axis="y")
        cm = None
    else:
        cm = confusion_matrix(yt_all, yp_all, labels=list(range(num_classes)))
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
        im = axes[2].imshow(cm_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        axes[2].set_xlabel("Predicted")
        axes[2].set_ylabel("True")
        axes[2].set_title("Confusion Matrix")
        axes[2].set_xticks(range(num_classes), cls_names, rotation=45)
        axes[2].set_yticks(range(num_classes), cls_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_norm[i, j]
                axes[2].text(j, i, f"{val:.2f}",
                             ha="center", va="center",
                             color="white" if val > 0.5 else "black")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training plots saved to %s", save_path)

    plt.close(fig)

    final_acc = float(accuracy_score(yt_all, yp_all)) if yt_all else 0.0
    final_f1 = float(f1_score(yt_all, yp_all, average="macro", zero_division=0)) if yt_all else 0.0
    result = {"final_loss": train_loss_history[-1],
              "final_acc": val_acc_history[-1], "final_f1": final_f1,
              "final_val_acc": final_acc, "final_val_f1": final_f1}
    if cm is not None:
        result["cm"] = cm
    else:
        result["per_class_f1"] = per_class_f1
    return result


# ----------------------------------------------------------------------------
# Pretrain loss curve (for SSL pretrain)
# ----------------------------------------------------------------------------

def plot_pretrain_loss(loss_history, component_histories=None, save_path=None):
    """Plot the pretrain total loss plus one curve per loss component.

    loss_history: list of per-epoch total loss.
    component_histories: {component_name: [per-epoch values]} — whatever the
        method's train_step returned (dino/ibot/gram/recon/dist/adv/...).
        Components whose history length does not match loss_history are
        skipped. save_path: output PDF path. If None, the figure is not saved.
    """
    if not loss_history:
        logger.warning("No pretrain loss history to plot.")
        return None

    epochs = range(1, len(loss_history) + 1)
    components = {
        name: hist for name, hist in (component_histories or {}).items()
        if isinstance(hist, list) and len(hist) == len(loss_history)
    }

    if components:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, loss_history, label="total", linewidth=1.5)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Total Loss")
        axes[0].grid(True)
        axes[0].legend()
        for name, hist in components.items():
            axes[1].plot(epochs, hist, label=name, linewidth=1.2)
        axes[1].set_title("Loss Components")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True)
        axes[1].legend()
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, loss_history, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Pretrain Loss")
        ax.grid(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Pretrain loss curve saved to %s", save_path)
    plt.close(fig)
    return save_path


# ----------------------------------------------------------------------------
# DINOv3 training-quality diagnostics
# ----------------------------------------------------------------------------

#: Head-collapse threshold in logit space (DINOv3 collapse-guard value):
#: healthy heads sit at std ~0.02-0.06, collapsed heads at ~0.002.
HEAD_LOGITS_STD_COLLAPSE_THRESHOLD = 5e-3


def plot_head_track(head_std_history=None, head_entropy_history=None,
                    sts_history=None, gram_masked_history=None,
                    gram_unmasked_history=None, cdi_history=None,
                    save_path=None):
    """DINOv3 monitoring plots: head collapse signals + Gram split stats.

    head_std_history:   per-epoch student head-logit std (collapse if it
        drops below ~5e-3).
    head_entropy_history: normalized softmax entropy in [0, 1] (-> 1 collapse).
    sts_history:        mean teacher-student CLS cosine similarity.
    gram_masked/unmasked_history: per-epoch Gram losses on masked/unmasked
        patch tokens (gram anchoring enabled).
    cdi_history:        per-save content-dependence index (~1 = patch
        similarity maps homogenized / input-independent).
    """
    series = [
        (head_std_history, "head logits std", "Head logits std (collapse < 5e-3)"),
        (head_entropy_history, "head entropy", "Head entropy (norm., -> 1 = collapse)"),
        (sts_history, "teacher-student sim", "Teacher-student CLS similarity"),
    ]
    if cdi_history:
        series.append((cdi_history, "content dependence",
                       "Sim-map content dependence (1 = homogenized)"))
    present = [(h, lbl, ttl) for h, lbl, ttl in series if h]
    has_gram = (gram_masked_history or gram_unmasked_history)

    n_plots = len(present) + (1 if has_gram else 0)
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, (hist, label, title) in zip(axes, present):
        epochs = range(1, len(hist) + 1)
        ax.plot(epochs, hist, marker="o", label=label, linewidth=1.4)
        if title.startswith("Head logits std"):
            ax.axhline(HEAD_LOGITS_STD_COLLAPSE_THRESHOLD, color="red",
                       linestyle="--", linewidth=1.0,
                       label="collapse threshold")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend(fontsize=8)

    if has_gram and len(axes) > len(present):
        ax = axes[-1]
        if gram_masked_history:
            ax.plot(range(1, len(gram_masked_history) + 1), gram_masked_history,
                    marker="o", label="gram (masked)", linewidth=1.2)
        if gram_unmasked_history:
            ax.plot(range(1, len(gram_unmasked_history) + 1), gram_unmasked_history,
                    marker="s", label="gram (unmasked)", linewidth=1.2)
        ax.set_title("Gram loss split (masked / unmasked)")
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend(fontsize=8)

    fig.suptitle("DINOv3 training-quality diagnostics", fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Head-track diagnostics saved to %s", save_path)
    plt.close(fig)


def _to_display_rgb(x_cpu):
    """Normalized input tensor -> (H,W,3) uint8 display image.

    Takes the first channel of a (…,C,H,W)/(C,H,W) tensor and min-max
    rescales it, so the user sees the exact pixels fed to the model
    (brightness rescaled for display only).
    """
    a = np.asarray(x_cpu)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a[0]
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0
    g = ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.stack([g] * 3, axis=-1)


def plot_attention_combined(samples, save_path, inputs=None, anchor_idx=None,
                            title=None):
    """Combined diagnostic PDF: one row per image containing the input, the
    patch-similarity anchors (coolwarm) and the per-head + mean CLS attention
    maps (viridis) side by side.

    samples: list; each = {'anchors': [4 (g, g) cosine-sim maps],
    'attn': {'per_head': [6 (g, g) maps, one per attention head],
    'mean': (g, g)}}.
    inputs (optional): one (H,W,3) display per row -> prepended column.
    anchor_idx (optional): sorted anchor patch indices for the anchor labels.
    """
    if not samples:
        return
    n_rows = len(samples)
    n_anchors = len(samples[0]["anchors"])
    n_heads = len(samples[0]["attn"]["per_head"])
    base_cols = n_anchors + n_heads + 1  # anchors + heads + mean
    n_cols = base_cols + (1 if inputs is not None else 0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i, smp in enumerate(samples):
        col = 0
        if inputs is not None:
            axes[i][0].imshow(inputs[i])
            axes[i][0].set_xticks([])
            axes[i][0].set_yticks([])
            if i == 0:
                axes[i][0].set_title("input (normalized)", fontsize=8)
            col = 1
        im = None
        for j, mp in enumerate(smp["anchors"]):
            ax = axes[i][col + j]
            im = ax.imshow(mp, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                label = f"anchor {j + 1}"
                if anchor_idx:
                    g = mp.shape[0]
                    r, c = divmod(anchor_idx[j], g)
                    label += f"\n(idx {anchor_idx[j]}, r{r} c{c})"
                ax.set_title(label, fontsize=8)
        for hi, mp in enumerate(smp["attn"]["per_head"]):
            ax = axes[i][col + n_anchors + hi]
            im = ax.imshow(mp, cmap="viridis", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"head {hi + 1}", fontsize=8)
        mean = smp["attn"]["mean"]
        axm = axes[i][col + n_anchors + n_heads]
        im = axm.imshow(mean, cmap="viridis", vmin=0.0, vmax=1.0)
        axm.set_xticks([])
        axm.set_yticks([])
        if i == 0:
            axm.set_title("mean head", fontsize=8)
    fig.suptitle(title or f"Attention + patch similarity ({n_rows} samples)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    # The full-grid colorbar axes are not tight_layout-compatible (matplotlib
    # warns instead of raising on newer versions) — always use the manual
    # adjustment; savefig's bbox_inches="tight" crops the final margins.
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Combined attention maps saved to %s", save_path)
    plt.close(fig)
