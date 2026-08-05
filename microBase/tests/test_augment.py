"""Tests for microBase.augment: direct albumentations class instantiation."""

import numpy as np
import pytest

from microBase import augment


def test_build_pipeline_empty_returns_none():
    assert augment.build_pipeline(None) is None
    assert augment.build_pipeline([]) is None


def test_build_pipeline_basic():
    pipeline = augment.build_pipeline([
        {"HorizontalFlip": {"p": 0.5}},
        {"VerticalFlip": {"p": 0.5}},
    ])
    assert pipeline is not None
    assert len(pipeline.transforms) == 2


def test_build_pipeline_unknown_class_exits():
    with pytest.raises(SystemExit):
        augment.build_pipeline([{"NotARealTransform": {}}])


def test_build_pipeline_non_transform_class_exits():
    # Compose exists in albumentations but is not a BasicTransform
    with pytest.raises(SystemExit):
        augment.build_pipeline([{"Compose": {}}])


def test_build_pipeline_bad_spec_exits():
    with pytest.raises(SystemExit):
        augment.build_pipeline([{"a": {}, "b": {}}])  # two keys
    with pytest.raises(SystemExit):
        augment.build_pipeline(["not_a_dict"])


def test_build_pipeline_unknown_kwargs_exits():
    # AlbumentationsX renamed classic kwargs (Rotate.limit -> angle_range) and
    # silently drops unknown ones, so build_pipeline hard-exits on the warning.
    with pytest.raises(SystemExit):
        augment.build_pipeline([{"Rotate": {"limit": 30, "p": 1.0}}])
    with pytest.raises(SystemExit):
        augment.build_pipeline([{"GaussianBlur": {"blur_limit": [3, 5], "p": 1.0}}])


def test_apply_with_none_pipeline_returns_inputs():
    img = np.zeros((10, 10, 2), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.int32)
    out_img, out_mask = augment.apply(None, img, mask)
    assert out_img is img
    assert out_mask is mask


def test_apply_changes_image():
    """Applying HorizontalFlip with p=1.0 should mirror the image."""
    img = np.zeros((10, 10, 1), dtype=np.uint8)
    img[0, 0, 0] = 255  # top-left pixel is bright
    pipeline = augment.build_pipeline([{"HorizontalFlip": {"p": 1.0}}])
    out_img, _ = augment.apply(pipeline, img, None)
    assert out_img[0, 0, 0] == 0
    assert out_img[0, -1, 0] == 255


def test_apply_with_mask():
    """Pipeline with mask should return both image and mask transformed."""
    img = np.zeros((10, 10, 1), dtype=np.uint8)
    img[0, 0, 0] = 255
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[0, 0] = 1
    pipeline = augment.build_pipeline([{"HorizontalFlip": {"p": 1.0}}])
    out_img, out_mask = augment.apply(pipeline, img, mask)
    assert out_img[0, -1, 0] == 255
    assert out_mask[0, -1] == 1
