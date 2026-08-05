"""Default regex patterns for common microscopy file layouts.

These defaults follow the Operetta naming convention with optional
leading zeros (e.g. r1c1f01p01-ch1sk1). Users can override them by
typing custom patterns in the GUI or setting them in session.yml.
"""

DEFAULT_IMAGE_PATTERN = r"r(?P<row>\d+)c(?P<col>\d+)f0?(?P<field>\d+)p0?(?P<stack>\d+)-ch0?(?P<channel>\d+)(?:sk|t)0?(?P<timepoint>\d+).*\.tiff"

DEFAULT_MASK_PATTERN = r"r(?P<row>\d+)c(?P<col>\d+)f0?(?P<field>\d+)p0?(?P<stack>\d+)-ch0?(?P<channel>\d+)(?:sk|t)0?(?P<timepoint>\d+).*_cp_masks_(?P<mask_name>.+)\.png"

DEFAULT_IMAGE_SUBDIR_PATTERN = "images"
