"""Side-by-side rendering of anisotropic vs isotropic 5-grain simulations.

Reads ``five_grains_2D.h5`` (anisotropic) and ``isotropic_five_grain.h5``
(isotropic), renders each timestep side by side with matplotlib.imshow
using Crameri's perceptually uniform ``managua`` colormap, writes the
frames to ``render_frames/`` and assembles them into a GIF at 10 fps
with PIL.
"""

from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as cmc
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================

ANISO_FILE = "five_grains_2D.h5"
ISO_FILE = "isotropic_five_grain.h5"

# Variable to visualize (must exist in both files)
VAR_NAME = "grain_ID"

TIME_START = 1
TIME_END = 50

FRAME_DIR = Path("render_frames")
GIF_NAME = "aniso_vs_iso.gif"
FPS = 10

# grain_ID takes the discrete values 0..4: sample 5 evenly spaced colors
# from managua and use a BoundaryNorm so each grain gets exactly one color.
N_GRAINS = 5
CMAP = mpl.colors.ListedColormap(
    cmc.managua(np.linspace(0.0, 1.0, N_GRAINS)),
    name="managua_discrete",
)
NORM = mpl.colors.BoundaryNorm(
    boundaries=np.arange(N_GRAINS + 1) - 0.5,  # [-0.5, 0.5, ..., 4.5]
    ncolors=N_GRAINS,
)

# Physical domain extent (matches the input files)
SIDE = 80.0


# =============================================================================
# Publication style (adapted from the publication-quality-figures skill;
# usetex is disabled so the script runs without a LaTeX installation and
# renders 50 frames quickly -- flip it on for final paper figures)
# =============================================================================

def setup_publication_style() -> None:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 12,
        "savefig.dpi": 500,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


# =============================================================================
# HDF5 reading
# =============================================================================

def resolve_time_index(hf: h5py.File, var_name: str, time_idx: int) -> int:
    """Map an ordinal (possibly negative) index onto the stored timestep
    suffixes for ``var_name``, matching normal Python indexing."""
    prefix = f"{var_name}."
    suffixes = sorted(
        int(key[len(prefix):])
        for key in hf.keys()
        if key.startswith(prefix) and key[len(prefix):].isdigit()
    )
    if not suffixes:
        raise KeyError(f"No timesteps found for variable '{var_name}'")
    return suffixes[time_idx]


def read_h5var(
    hf: h5py.File,
    var_name: str,
    time_idx: int,
    *,
    resolve_negative: bool = True,
) -> np.ndarray:
    """Read a variable from an open HDF5 file and flip array axis 0.

    A negative ``time_idx`` is interpreted as an ordinal position in the
    sorted list of stored timestep suffixes, matching normal Python indexing.
    """
    resolved_idx = (
        resolve_time_index(hf, var_name, time_idx)
        if resolve_negative or time_idx < 0
        else time_idx
    )
    key = f"{var_name}.{resolved_idx}"
    if key not in hf:
        raise KeyError(f"Key '{key}' not found in HDF5 file")

    return np.flip(np.asarray(hf[key][()]), axis=0)


# =============================================================================
# Frame rendering
# =============================================================================

def render_frame(
    data_aniso: np.ndarray,
    data_iso: np.ndarray,
    t: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.0, 3.4),
        constrained_layout=True,
    )

    extent = (0.0, SIDE, 0.0, SIDE)

    for i, (ax, data, title) in enumerate(
        zip(axes, (data_aniso, data_iso), ("Anisotropic", "Isotropic"))
    ):
        im = ax.imshow(
            data,
            cmap=CMAP,
            norm=NORM,
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])
        ax.set_aspect("equal")

    # Single shared colorbar; one tick per grain, centered on each color
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, ticks=np.arange(N_GRAINS))
    cbar.set_label("grain ID")

    fig.suptitle(f"timestep {t}")

    # White (non-transparent) background: GIF frames composite poorly
    # with transparency, so we deviate from the transparent default here.
    fig.savefig(out_path, dpi=500, transparent=False, facecolor="white")
    plt.close(fig)


# =============================================================================
# Poster figure: initial condition | isotropic | anisotropic
# =============================================================================

# Note: grain_ID does not exist at timestep 0 (INITIAL, before the first
# solve), so "initial condition" here means the first available grain_ID
# snapshot, i.e. TIME_START (t=1) rather than t=0.
POSTER_FIGURE_NAME = "ic_iso_aniso_comparison"


def render_poster_figure(
    data_ic: np.ndarray,
    data_iso: np.ndarray,
    data_aniso: np.ndarray,
    out_dir: Path,
) -> None:
    """Static 3-panel figure: initial condition, isotropic result, anisotropic
    result. Saved as both PDF (vector, for print) and PNG (quick preview)."""
    fig, axes = plt.subplots(
        1, 3,
        figsize=(9.5, 3.4),
        constrained_layout=True,
    )

    extent = (0.0, SIDE, 0.0, SIDE)
    panels = (
        (data_ic, "Initial condition"),
        (data_iso, "Isotropic"),
        (data_aniso, "Anisotropic"),
    )

    for i, (ax, (data, title)) in enumerate(zip(axes, panels)):
        im = ax.imshow(
            data,
            cmap=CMAP,
            norm=NORM,
            origin="lower",
            extent=extent,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        else:
            ax.set_yticklabels([])
        ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, ticks=np.arange(N_GRAINS))
    cbar.set_label("grain ID")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"{POSTER_FIGURE_NAME}.{ext}",
            dpi=500,
            transparent=False,
            facecolor="white",
        )
    plt.close(fig)
    print(f"Wrote {out_dir / POSTER_FIGURE_NAME}.[pdf|png]")


# =============================================================================
# GIF assembly
# =============================================================================

def frames_to_gif(frame_paths: list[Path], gif_path: Path, fps: int) -> None:
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # ms per frame
        loop=0,
        optimize=False,
    )
    print(f"Wrote {gif_path} ({len(frames)} frames @ {fps} fps)")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_publication_style()

    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    steps = range(TIME_START, TIME_END + 1)

    with h5py.File(ANISO_FILE, "r") as hf_a, h5py.File(ISO_FILE, "r") as hf_b:

        # Initial condition: grain_ID at t=1 (t=0/INITIAL has no grain_ID).
        # Both files share the same IC, so either one works; use ANISO_FILE.
        data_ic = read_h5var(hf_a, VAR_NAME, TIME_START)

        frame_paths = []
        data_aniso = data_iso = None
        for t in steps:
            data_aniso = read_h5var(hf_a, VAR_NAME, t)
            data_iso = read_h5var(hf_b, VAR_NAME, t)

            out_path = FRAME_DIR / f"frame_{t:04d}.png"
            render_frame(data_aniso, data_iso, t, out_path)
            frame_paths.append(out_path)

            if t % 10 == 0 or t == TIME_START:
                print(f"  rendered timestep {t}")

        # data_aniso / data_iso now hold the TIME_END snapshot from the loop
        # above -- reuse them for the poster figure instead of re-reading.
        render_poster_figure(data_ic, data_iso, data_aniso, FRAME_DIR)

    frames_to_gif(frame_paths, FRAME_DIR / GIF_NAME, FPS)


if __name__ == "__main__":
    main()