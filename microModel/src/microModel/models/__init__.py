"""SSL model registry. Add new SSL methods by creating models/<method>.py
with a build_<method>(backbone_cfg, method_cfg, device) factory and registering
it in _SSL_REGISTRY below.
"""

import sys

from . import dinov3

_SSL_REGISTRY = {
    "dinov3": dinov3.build_dinov3,
}


def build_ssl_model(method, backbone_cfg, method_cfg, device):
    """Dispatch to the registered SSL method's build function.

    Returns the SSL model (nn.Module) on the given device.
    """
    if method not in _SSL_REGISTRY:
        print(
            f"Error: unknown SSL method '{method}'. "
            f"Available: {sorted(_SSL_REGISTRY)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return _SSL_REGISTRY[method](backbone_cfg, method_cfg, device)


def get_train_step(method):
    """Return the train_step function for the given SSL method."""
    if method == "dinov3":
        return dinov3.train_step
    print(
        f"Error: no train_step for SSL method '{method}'. "
        f"Available: dinov3",
        file=sys.stderr,
    )
    sys.exit(1)


def get_criterion(method, method_cfg, device):
    """Build and return the loss criterion for the given SSL method."""
    if method == "dinov3":
        # DINOv3 losses are ported into models/dinov3.py (Sinkhorn-Knopp
        # teacher centering + Gram anchoring); all temperature handling is
        # done inside dinov3.train_step.
        from .dinov3 import DINOLoss, iBOTPatchLoss, KoLeoLoss, GramLoss
        proto_dim = int(method_cfg.get("head_n_prototypes", 65536))
        student_temp = float(method_cfg.get("student_temp", 0.1))
        dino_criterion = DINOLoss(proto_dim, student_temp=student_temp).to(device)
        ibot_criterion = iBOTPatchLoss(proto_dim, student_temp=student_temp).to(device)
        koleo_criterion = KoLeoLoss()
        gram_cfg = method_cfg.get("gram", {}) or {}
        if gram_cfg.get("use_loss", False):
            gram_criterion = GramLoss(
                apply_norm=gram_cfg.get("normalized", True),
                remove_neg=gram_cfg.get("remove_neg", False),
                remove_only_teacher_neg=gram_cfg.get("remove_only_teacher_neg", False),
            ).to(device)
        else:
            gram_criterion = None
        return (dino_criterion, ibot_criterion, koleo_criterion, gram_criterion)
    print(
        f"Error: no criterion for SSL method '{method}'",
        file=sys.stderr,
    )
    sys.exit(1)
