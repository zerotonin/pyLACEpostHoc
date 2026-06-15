# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — constants                                       ║
# ║  « one source of truth for colours, paths, and figure rules »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Wong (2011) colourblind-safe palette, figure defaults, the      ║
# ║  save_figure() triple-output helper, arena geometry, and shared  ║
# ║  type aliases.  Import from here instead of hardcoding values.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Central configuration for pyLACEpostHoc colours, figures, and geometry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

# ┌────────────────────────────────────────────────────────────┐
# │ Type aliases  « shared vocabulary for array shapes »       │
# └────────────────────────────────────────────────────────────┘
Coords: TypeAlias = np.ndarray          # (N, 2) array of x, y points
Trajectory: TypeAlias = np.ndarray      # (T, 2) array of x, y over time
MidLine: TypeAlias = np.ndarray         # (T, K, 2) body mid-line over time
ArenaSize: TypeAlias = tuple[float, float]   # (x, y) extent in millimetres

# ┌────────────────────────────────────────────────────────────┐
# │ Wong (2011) palette  « colourblind-safe base colours »     │
# └────────────────────────────────────────────────────────────┘
WONG: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Semantic mapping for the counter-current / rheotaxis experiments.
# Extend per analysis rather than picking ad-hoc colours downstream.
SEMANTIC: dict[str, str] = {
    "with_current":    WONG["blue"],
    "against_current": WONG["vermilion"],
    "neutral":         WONG["black"],
    "wildtype":        WONG["sky_blue"],
    "mutant":          WONG["orange"],
}

# ┌────────────────────────────────────────────────────────────┐
# │ Figure defaults  « SVG-first, editable text, ≥200 dpi »    │
# └────────────────────────────────────────────────────────────┘
FIGURE_DPI: int = 300
FIGURE_SIZE: tuple[float, float] = (8.0, 6.0)
SVG_FONTTYPE: str = "none"   # keep text as <text>, not paths, for Inkscape

# ─────────────────────────────────────────────────────────────────
#  Arena geometry  « migrated piecemeal from the analysis modules »
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArenaGeometry:
    """Physical extent of a recording arena in millimetres."""

    name: str           #: experiment key, e.g. ``"counter_current"``
    extent: ArenaSize   #: (x, y) arena size in millimetres
    source: str         #: provenance of the value


# Values lifted from the legacy modules; confirm against the rigs as the
# owning sprints migrate each analysis off its hardcoded numbers.
ARENAS: dict[str, ArenaGeometry] = {
    "cruise": ArenaGeometry("cruise", (114.0, 248.0), "fishRecAnalysis legacy default"),
    "c_start": ArenaGeometry("c_start", (40.0, 80.0), "fishRecAnalysis legacy default"),
    "counter_current": ArenaGeometry(
        "counter_current", (45.0, 167.0), "fishRecAnalysis legacy default"
    ),
}

# ┌────────────────────────────────────────────────────────────┐
# │ Output conventions  « where figures and tables land »      │
# └────────────────────────────────────────────────────────────┘
FIGURE_DIRNAME: str = "figures"


def save_figure(
    fig,
    stem: str,
    output_dir: Path,
    data=None,
) -> dict[str, Path]:
    """Export a figure as SVG + PNG, plus an optional CSV data companion.

    Args:
        fig:        Matplotlib figure to save.
        stem:       Filename stem (no extension).
        output_dir: Target directory (created if needed).
        data:       Optional pandas DataFrame written alongside as CSV
                    so reviewers can check the numbers behind the plot.

    Returns:
        Mapping of format name to the path written.
    """
    import matplotlib

    matplotlib.rcParams["svg.fonttype"] = SVG_FONTTYPE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(svg_path)
    written["svg"] = svg_path

    png_path = output_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=FIGURE_DPI)
    written["png"] = png_path

    if data is not None:
        csv_path = output_dir / f"{stem}.csv"
        data.to_csv(csv_path, index=False)
        written["csv"] = csv_path

    return written
