"""Nike-inspired color palette and matplotlib styling."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# Nike-inspired palette
NIKE_COLORS = {
    "black": "#111111",
    "orange": "#FA5400",
    "white": "#FFFFFF",
    "grey": "#7E7E7E",
    "light_grey": "#F5F5F5",
    "red": "#CC0000",
    "green": "#00A651",
    "blue": "#0077C8",
}

NIKE_PALETTE = [
    NIKE_COLORS["orange"],
    NIKE_COLORS["black"],
    NIKE_COLORS["blue"],
    NIKE_COLORS["red"],
    NIKE_COLORS["green"],
    NIKE_COLORS["grey"],
]


def apply_nike_style() -> None:
    """Apply Nike-themed styling to matplotlib and seaborn."""
    plt.rcParams.update(
        {
            "figure.facecolor": NIKE_COLORS["white"],
            "axes.facecolor": NIKE_COLORS["light_grey"],
            "axes.edgecolor": NIKE_COLORS["grey"],
            "axes.labelcolor": NIKE_COLORS["black"],
            "text.color": NIKE_COLORS["black"],
            "xtick.color": NIKE_COLORS["black"],
            "ytick.color": NIKE_COLORS["black"],
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": NIKE_COLORS["grey"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "figure.figsize": (12, 6),
        }
    )
    sns.set_palette(NIKE_PALETTE)
