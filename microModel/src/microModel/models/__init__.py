"""SSL model registry. Add new SSL methods by creating models/<method>.py
with a build_<method>(backbone_cfg, method_cfg, device) factory and registering
it in _SSL_REGISTRY below.
"""

import sys

from . import byol, dinov2

_SSL_REGISTRY = {
    "byol": byol.build_byol,
    "dinov2": dinov2.build_dinov2,
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
    if method == "byol":
        return byol.train_step
    elif method == "dinov2":
        return dinov2.train_step
    print(
        f"Error: no train_step for SSL method '{method}'. "
        f"Available: byol, dinov2",
        file=sys.stderr,
    )
    sys.exit(1)


def get_criterion(method, method_cfg, device):
    """Build and return the loss criterion for the given SSL method."""
    if method == "byol":
        from lightly.loss import NegativeCosineSimilarity
        criterion = NegativeCosineSimilarity()
        return criterion
    elif method == "dinov2":
        from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
        # output_dim must match DINOv2ProjectionHead's output_dim (65536),
        # NOT the backbone input dim — the losses compare the head outputs.
        # The teacher temperature is NOT configured here: dinov2.train_step
        # passes the cosine-scheduled teacher_temp explicitly every step,
        # which overrides the losses' internal schedules entirely.
        dino_criterion = DINOLoss(
            output_dim=65536,
            student_temp=0.1,
        ).to(device)
        ibot_criterion = IBOTPatchLoss(
            output_dim=65536,
            student_temp=0.1,
        ).to(device)
        koleo_criterion = KoLeoLoss()
        return (dino_criterion, ibot_criterion, koleo_criterion)
    print(
        f"Error: no criterion for SSL method '{method}'",
        file=sys.stderr,
    )
    sys.exit(1)
