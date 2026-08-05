"""Constants and configuration for microVis."""

# Plate format: name → (rows, cols)
PLATE_FORMATS = {
    "24-well": (4, 6),
    "96-well": (8, 12),
    "384-well": (16, 24),
}
DEFAULT_PLATE = "96-well"

# Channel display colors: name → (R, G, B) in [0, 1]
CHANNEL_COLORS = {
    "green": (0.0, 1.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "magenta": (1.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
    "white": (1.0, 1.0, 1.0),
}
DEFAULT_CHANNEL_COLORS = ["green", "red", "magenta", "blue", "cyan"]

# Colormap options for continuous data (matplotlib names)
CMAP_OPTIONS = [
    "viridis", "plasma", "inferno", "magma", "cividis",
    "RdBu", "RdYlBu", "RdYlGn", "Spectral", "coolwarm",
    "Blues", "Greens", "Reds", "Purples", "Oranges",
    "Greys", "turbo", "jet",
]
DEFAULT_CMAP = "viridis"

# Qualitative palettes for categorical data
QUALITATIVE_PALETTES = [
    "Set1", "Set2", "Set3", "Pastel1", "Pastel2",
    "Paired", "Accent", "Dark2",
]

# Aggregation methods for well grid
AGG_METHODS = ["mean", "sum", "std"]

# Contrast methods
CONTRAST_METHODS = ["none", "invert", "gamma", "histogram_equalization"]

# Image dtype → max pixel value
DTYPE_MAX = {
    "uint8": 255,
    "uint16": 65535,
    "uint32": 4294967295,
    "float32": 1.0,
    "float64": 1.0,
}


def get_cmap(name: str):
    """Case-insensitive matplotlib colormap lookup."""
    import matplotlib.pyplot as plt
    try:
        return plt.colormaps[name]
    except KeyError:
        return plt.colormaps[name.lower()]
