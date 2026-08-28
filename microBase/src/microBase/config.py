"""Config helpers: YAML load/save and per-dataset SessionFile.

SessionFile lives at <dataset>/session.yml (NOT under .microprofiler/).
It records dataset-level state shared across tools:
- regex patterns (image/mask/subdir)
- applied pipeline steps (microProfiler)
- channel colors and contrast (microVis)
- arbitrary key-value pairs

Tools deep-merge their own updates; none overwrites the whole file.
"""

import sys
from pathlib import Path

import yaml


def normalize_null_strings(obj):
    """Recursively convert case-insensitive 'null'/'none' string spellings to None.

    PyYAML (YAML 1.1) already maps unquoted null/Null/NULL/~ and empty values
    to Python None, but 'None', 'none', 'NONE', 'nOne', 'NUll', ... parse as
    literal strings and silently break downstream `is None` checks. This walks
    a loaded YAML structure (dicts/lists/scalars) and converts every string
    value whose lower-cased form is 'null' or 'none' to Python None.

    Dict keys are left untouched. Non-string, non-container values are
    returned as-is.
    """
    if isinstance(obj, str):
        return None if obj.lower() in ("null", "none") else obj
    if isinstance(obj, dict):
        return {k: normalize_null_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_null_strings(v) for v in obj]
    return obj


def load_yaml(path):
    """Load a YAML file. Print + exit on missing file or parse error;
    raise ValueError when the root is not a mapping (no silent {} fallback)."""
    path = Path(path)
    if not path.exists():
        print(f"Error: YAML file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: failed to parse YAML {path}: {e}", file=sys.stderr)
        sys.exit(1)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Error: YAML root must be a mapping, got {type(data).__name__} in {path}"
        )
    return normalize_null_strings(data)


def save_yaml(path, data):
    """Save data as YAML. Creates parent dirs. Print + exit on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    except Exception as e:
        print(f"Error: failed to write YAML {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _deep_merge(base, overlay):
    """Recursively merge overlay into base. Overlay wins on conflicts."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        return overlay
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


class SessionFile:
    """Per-dataset session state at <dataset>/session.yml.

    Used by microProfiler (patterns + applied steps), microVis (channel
    colors), and any tool that wants to persist per-dataset state.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.path = self.dataset_dir / "session.yml"

    def exists(self):
        return self.path.exists()

    def load(self):
        """Load the session dict. Returns {} if file doesn't exist yet."""
        if not self.path.exists():
            return {}
        return load_yaml(self.path)

    def save(self, updates):
        """Deep-merge updates into the existing session and write back.

        Does NOT overwrite fields not present in `updates`. Tools can
        update their own section without touching others.
        """
        existing = self.load()
        merged = _deep_merge(existing, updates)
        save_yaml(self.path, merged)

    # ---- Convenience accessors for common fields ----

    def get_patterns(self):
        """Return (image_pattern, mask_pattern, image_subdir_pattern)."""
        d = self.load()
        return (
            d.get("image_pattern"),
            d.get("mask_pattern"),
            d.get("image_subdir_pattern"),
        )

    def set_patterns(self, image_pattern=None, mask_pattern=None, image_subdir_pattern=None):
        updates = {}
        if image_pattern is not None:
            updates["image_pattern"] = image_pattern
        if mask_pattern is not None:
            updates["mask_pattern"] = mask_pattern
        if image_subdir_pattern is not None:
            updates["image_subdir_pattern"] = image_subdir_pattern
        if updates:
            self.save(updates)

    def get_applied_steps(self):
        # `or []`: the key may exist with a null value (load_yaml normalizes
        # 'null'/'none' strings to None) — treat null as "no steps".
        return self.load().get("applied_steps") or []

    def set_applied_steps(self, steps):
        self.save({"applied_steps": list(steps)})

    def get_channel_colors(self):
        """Return per-channel color/contrast config dict, or {} if absent."""
        return self.load().get("channel_colors") or {}

    def set_channel_colors(self, colors):
        """Update channel_colors section. colors: {ch_name: {color, vmin, vmax}}"""
        self.save({"channel_colors": colors})
