"""Augmentation module: direct albumentations class instantiation from YAML.

YAML shape (class-as-key):
    augmentation:
      - HorizontalFlip: {p: 0.5}
      - VerticalFlip: {p: 0.5}
      - Rotate: {angle_range: [-180, 180], p: 0.5}
      - RandomBrightnessContrast: {brightness_range: [-0.1, 0.2], contrast_range: [-0.1, 0.2], p: 0.5}
      - GaussianBlur: {blur_range: [3, 5], p: 0.5}
      - GaussNoise: {std_range: [0.02, 0.08], p: 0.5}

Each key is an albumentations transform class name; value is its kwargs dict.
Transforms compose in list order.

Note: this suite runs on AlbumentationsX (import name `albumentations`), which
renamed several classic kwargs: Rotate.limit -> angle_range,
RandomScale.scale_limit -> scale_range,
RandomBrightnessContrast.{brightness,contrast}_limit -> {brightness,contrast}_range,
RandomGamma.gamma_limit -> gamma_range, GaussianBlur.blur_limit -> blur_range.
AlbumentationsX silently drops unknown kwargs, so build_pipeline treats an
"Argument(s) X are not valid" warning as a hard error.

For geometric transforms that co-transform masks (Rotate, Affine, etc.),
add mask-safe kwargs explicitly in the config if needed, e.g.:
  - Rotate: {angle_range: [-180, 180], border_mode: 0, mask_interpolation: 0, p: 0.5}
where border_mode=0 is cv2.BORDER_CONSTANT and mask_interpolation=0 is
cv2.INTER_NEAREST. Without these, albumentations defaults may smear masks.

build_pipeline(spec) returns an albumentations Compose (or None if spec empty).
apply(pipeline, image, mask) returns (image, mask) after augmentation.
"""

import sys
import warnings

import albumentations as A


def build_pipeline(spec):
    """Build an albumentations Compose from a YAML spec list.

    spec: list of single-key dicts, e.g. [{"Rotate": {"angle_range": 180, "p": 0.5}}].
          Key is an albumentations class name; value is its kwargs dict.
          None / empty list -> returns None (no augmentation).
    """
    if not spec:
        return None
    transforms = []
    for entry in spec:
        if not isinstance(entry, dict) or len(entry) != 1:
            print(
                f"Error: augmentation spec entries must be single-key dicts, "
                f"got {entry}",
                file=sys.stderr,
            )
            sys.exit(1)
        name = next(iter(entry))
        kwargs = entry[name] or {}
        cls = getattr(A, name, None)
        if cls is None:
            print(
                f"Error: albumentations has no transform '{name}'. "
                f"Check the class name against the albumentations docs.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not (isinstance(cls, type) and issubclass(cls, A.BasicTransform)):
            print(
                f"Error: '{name}' is not an albumentations transform class.",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                t = cls(**kwargs)
            for w in caught:
                if "not valid for transform" in str(w.message):
                    print(
                        f"Error: augmentation '{name}' got unknown kwargs "
                        f"{kwargs}: {w.message}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
        except Exception as e:
            print(
                f"Error: failed to build augmentation '{name}' with kwargs {kwargs}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        transforms.append(t)
    if not transforms:
        return None
    return A.Compose(transforms)


def apply(pipeline, image, mask=None):
    """Apply a pipeline to (image, mask).

    image: (H, W, C) uint8/uint16/float array.
    mask:  (H, W) integer array or None.
    Returns (image, mask) — augmented. If pipeline is None, returns inputs unchanged.
    """
    if pipeline is None:
        return image, mask
    if mask is not None:
        result = pipeline(image=image, mask=mask)
        return result["image"], result["mask"]
    result = pipeline(image=image)
    return result["image"], None
