#!/usr/bin/env python3
"""Render Ni and UO2 grain surfaces from Marlin HDF5 output.

The script extracts the ``gr0 = 0.5`` isosurface from each HDF5 file,
interpolates ``sigma_gr0_gr1`` onto that surface, and exports a side-by-side
interactive PyVista HTML scene.  Each material uses its own scalar range and
therefore its own Batlow colorbar.

Examples
--------
List the available timestep indices in both files::

    python render_spherical_grains.py --list-times

Render independent timestep indices and export an interactive HTML file::

    python render_spherical_grains.py \
        --ni-time-index 40 --uo2-time-index 75 \
        --output spherical_grain_comparison.html

Reuse a previously saved camera::

    python render_spherical_grains.py \
        --ni-time-index 40 --uo2-time-index 75 \
        --camera-json spherical_grain_camera.json

Create the final Matplotlib composite with independent colorbars::

    python render_spherical_grains.py \
        --ni-time-index 100 --uo2-time-index 30 \
        --camera-json spherical_grain_camera.json \
        --matplotlib-output spherical_grain_comparison.png

Render just the gr0 = 0.5 shape (no sigma coloring) at t=1 from both files::

    python render_spherical_grains.py \
        --camera-json spherical_grain_camera.json \
        --contour-output spherical_grain_contour_t1.png

On a machine with a graphical display, ``--gui`` opens a native PyVista
window.  Rotate to the desired view and press ``c`` to write the current
camera to ``--camera-output``.  The default headless HTML mode writes the
initial camera, but camera changes made later in a web browser cannot be sent
back to Python by a standalone exported HTML file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from cmcrameri import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


ETA_NAME = "gr0"
SIGMA_NAME = "sigma_gr0_gr1"
ETA_LEVEL = 0.5

# Fixed timestep suffix used for the plain gr0-contour figure. This is
# intentionally independent of --ni-time-index/--uo2-time-index, which drive
# the sigma-colored composite instead.
CONTOUR_TIME_INDEX = 1


@dataclass(frozen=True)
class RenderCase:
    name: str
    path: Path
    time_idx: int


@dataclass(frozen=True)
class SurfaceData:
    case: RenderCase
    resolved_time_idx: int
    surface: pv.PolyData
    scalar_range: tuple[float, float]


def available_time_indices(hf: h5py.File, var_name: str) -> list[int]:
    """Return sorted integer timestep suffixes stored for ``var_name``."""
    prefix = f"{var_name}."
    indices: list[int] = []

    for key in hf.keys():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        try:
            indices.append(int(suffix))
        except ValueError:
            continue

    return sorted(indices)


def resolve_time_index(hf: h5py.File, var_name: str, time_idx: int) -> int:
    """Resolve a possibly negative ordinal index to a stored timestep suffix."""
    if not isinstance(time_idx, (int, np.integer)):
        raise TypeError(
            f"time_idx must be an integer, got {type(time_idx).__name__}"
        )

    var_times = available_time_indices(hf, var_name)
    if not var_times:
        raise KeyError(f"No timesteps found for variable '{var_name}'")

    if time_idx < 0:
        if not -len(var_times) <= time_idx < len(var_times):
            raise IndexError(
                f"time_idx {time_idx} out of range for {len(var_times)} "
                f"stored '{var_name}' timesteps"
            )
        return var_times[time_idx]

    if time_idx not in var_times:
        raise IndexError(
            f"Timestep suffix {time_idx} is not available for '{var_name}'. "
            f"Available range: {var_times[0]} ... {var_times[-1]}"
        )

    return time_idx


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


def finite_scalar_range(values: np.ndarray, label: str) -> tuple[float, float]:
    """Return a finite, nonzero scalar range suitable for a color mapper."""
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"'{label}' contains no finite values on the surface")

    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if np.isclose(lower, upper):
        pad = max(abs(lower), 1.0) * 1.0e-12
        lower -= pad
        upper += pad
    return lower, upper


def extract_surface(
    case: RenderCase,
    spacing: Sequence[float],
    feature_angle: float,
) -> SurfaceData:
    """Extract one colored eta isosurface from an HDF5 result."""
    if not case.path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {case.path}")

    with h5py.File(case.path, "r") as hf:
        resolved_idx = resolve_time_index(hf, ETA_NAME, case.time_idx)
        eta = read_h5var(
            hf, ETA_NAME, resolved_idx, resolve_negative=False
        ).astype(np.float64, copy=False)
        sigma = read_h5var(
            hf, SIGMA_NAME, resolved_idx, resolve_negative=False
        ).astype(np.float64, copy=False)

    if eta.ndim != 3:
        raise ValueError(
            f"{case.path}: '{ETA_NAME}' must be 3D; received shape {eta.shape}"
        )
    if sigma.shape != eta.shape:
        raise ValueError(
            f"{case.path}: '{SIGMA_NAME}' shape {sigma.shape} does not match "
            f"'{ETA_NAME}' shape {eta.shape}"
        )
    if not np.nanmin(eta) <= ETA_LEVEL <= np.nanmax(eta):
        raise ValueError(
            f"{case.path}: eta={ETA_LEVEL} is outside the stored '{ETA_NAME}' "
            f"range [{np.nanmin(eta):.6g}, {np.nanmax(eta):.6g}] at timestep "
            f"{resolved_idx}"
        )

    # PyVista/VTK consumes point data with x varying fastest.  For an array
    # whose axes are interpreted as (x, y, z), Fortran flattening preserves
    # that mapping.  The axis-0 flip above intentionally matches read_h5var.
    grid = pv.ImageData(
        dimensions=eta.shape,
        spacing=tuple(float(value) for value in spacing),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data[ETA_NAME] = eta.ravel(order="F")
    grid.point_data[SIGMA_NAME] = sigma.ravel(order="F")

    surface = grid.contour(isosurfaces=[ETA_LEVEL], scalars=ETA_NAME)
    if surface.n_points == 0:
        raise RuntimeError(
            f"{case.path}: contouring '{ETA_NAME}' at {ETA_LEVEL} produced "
            "an empty surface"
        )
    if SIGMA_NAME not in surface.point_data:
        raise RuntimeError(
            f"PyVista did not interpolate '{SIGMA_NAME}' onto the contour"
        )

    # Preserve smooth lighting within each physical facet while preventing
    # point-normal averaging from visually rounding genuine facet edges.
    # Vertex splitting duplicates only points whose neighboring triangle
    # normals differ by more than feature_angle; point-data arrays such as
    # sigma_gr0_gr1 are carried onto the duplicated vertices.
    surface = surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=True,
        feature_angle=feature_angle,
        consistent_normals=True,
        auto_orient_normals=True,
    )

    # Remove the contour field from the active display data so that the mapper
    # cannot accidentally fall back to eta instead of grain-boundary energy.
    surface.set_active_scalars(SIGMA_NAME)
    scalar_range = finite_scalar_range(surface[SIGMA_NAME], SIGMA_NAME)

    return SurfaceData(
        case=case,
        resolved_time_idx=resolved_idx,
        surface=surface,
        scalar_range=scalar_range,
    )


def camera_to_dict(camera_position: pv.CameraPosition) -> dict[str, list[float]]:
    """Convert a PyVista camera position to JSON-serializable lists."""
    return {
        "position": [float(value) for value in camera_position.position],
        "focal_point": [float(value) for value in camera_position.focal_point],
        "view_up": [float(value) for value in camera_position.viewup],
    }


def camera_from_dict(data: dict[str, Sequence[float]]) -> pv.CameraPosition:
    """Construct a PyVista camera position from a saved JSON dictionary."""
    required = ("position", "focal_point", "view_up")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Camera JSON is missing: {', '.join(missing)}")

    vectors = [tuple(float(value) for value in data[key]) for key in required]
    if any(len(vector) != 3 for vector in vectors):
        raise ValueError("Each saved camera vector must contain exactly 3 values")

    return pv.CameraPosition(*vectors)


def save_camera(plotter: pv.Plotter, output_path: Path) -> None:
    """Save and print the active camera."""
    data = camera_to_dict(plotter.camera_position)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Camera saved to {output_path}")
    print(json.dumps(data, indent=2))


def load_camera(input_path: Path) -> pv.CameraPosition:
    """Load a camera from JSON."""
    with input_path.open("r", encoding="utf-8") as stream:
        return camera_from_dict(json.load(stream))


def add_crystal_axes(plotter: pv.Plotter) -> None:
    """Add crystallographic direction axes using the conventional RGB colors."""
    plotter.add_axes(
        xlabel="[100]",
        ylabel="[010]",
        zlabel="[001]",
        x_color="red",
        y_color="green",
        z_color="blue",
        line_width=3,
    )


def add_surface_subplot(
    plotter: pv.Plotter,
    row: int,
    column: int,
    data: SurfaceData,
) -> None:
    """Add one material surface and its independent colorbar."""
    plotter.subplot(row, column)
    plotter.set_background("white")

    actor = plotter.add_mesh(
        data.surface,
        scalars=SIGMA_NAME,
        cmap=cm.batlow,
        clim=data.scalar_range,
        smooth_shading=True,
        show_edges=False,
        show_scalar_bar=False,
        nan_color="lightgray",
    )
    plotter.add_scalar_bar(
        title="sigma (J/m^2)",
        mapper=actor.mapper,
        vertical=True,
        position_x=0.84,
        position_y=0.12,
        width=0.09,
        height=0.68,
        title_font_size=17,
        label_font_size=14,
        fmt="%.3f",
        color="black",
    )
    plotter.add_text(
        f"{data.case.name}",
        position="upper_left",
        font_size=30,
        color="black",
    )
    add_crystal_axes(plotter)


def build_plotter(
    surfaces: Sequence[SurfaceData],
    *,
    window_size: Sequence[int],
    off_screen: bool,
    camera_position: pv.CameraPosition | None,
) -> pv.Plotter:
    """Build the side-by-side PyVista scene."""
    if len(surfaces) != 2:
        raise ValueError("Exactly two surfaces are required")

    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=tuple(int(value) for value in window_size),
        off_screen=off_screen,
        border=False,
    )

    for column, data in enumerate(surfaces):
        add_surface_subplot(plotter, 0, column, data)

    # Link after actors and bounds exist so both panels start with exactly the
    # same crystallographic viewing direction.
    plotter.link_views()
    plotter.subplot(0, 0)
    if camera_position is None:
        plotter.view_isometric(render=False)
        plotter.reset_camera(render=False)
    else:
        plotter.camera_position = camera_position
    plotter.render()
    return plotter


def render_surface_image(
    data: SurfaceData,
    *,
    camera_position: pv.CameraPosition,
    window_size: Sequence[int],
    panel_zoom: float,
    anti_aliasing: str,
    scalars: str | None = SIGMA_NAME,
    color: str = "lightgray",
) -> np.ndarray:
    """Render one independently fitted surface with a fixed orientation.

    The saved camera supplies only the viewing direction and view-up vector.
    Its absolute position and focal point are replaced for each surface, after
    which ``reset_camera`` fits that surface to the panel.  Thus grains with
    different physical radii occupy comparable image areas without changing
    their shapes or viewing orientations.

    If ``scalars`` is ``None`` the surface is drawn as a plain shaded solid
    in ``color`` instead of being colored by a scalar field (used for the
    geometry-only gr0 contour figure).
    """
    plotter = pv.Plotter(
        window_size=tuple(int(value) for value in window_size),
        off_screen=True,
        border=False,
    )
    # plotter.set_background("white")
    if anti_aliasing != "none":
        plotter.enable_anti_aliasing(anti_aliasing)
    if scalars is None:
        plotter.add_mesh(
            data.surface,
            color=color,
            smooth_shading=True,
            show_edges=False,
        )
    else:
        plotter.add_mesh(
            data.surface,
            scalars=scalars,
            cmap=cm.batlow,
            clim=data.scalar_range,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=False,
            nan_color="lightgray",
        )
    add_crystal_axes(plotter)

    saved_position = np.asarray(camera_position.position, dtype=float)
    saved_focal_point = np.asarray(camera_position.focal_point, dtype=float)
    view_vector = saved_position - saved_focal_point
    saved_distance = float(np.linalg.norm(view_vector))
    if not np.isfinite(saved_distance) or saved_distance <= 0.0:
        raise ValueError("Saved camera position and focal point must be distinct")

    view_direction = view_vector / saved_distance
    surface_center = np.asarray(data.surface.center, dtype=float)
    fitted_position = surface_center + saved_distance * view_direction
    plotter.camera_position = pv.CameraPosition(
        tuple(fitted_position),
        tuple(surface_center),
        tuple(camera_position.viewup),
    )
    plotter.reset_camera(render=False)
    plotter.camera.zoom(panel_zoom)
    plotter.render()
    image = plotter.screenshot(return_img=True, transparent_background=True)
    plotter.close()
    return np.asarray(image)


def export_matplotlib_figure(
    surfaces: Sequence[SurfaceData],
    *,
    camera_position: pv.CameraPosition,
    output_path: Path,
    render_size: Sequence[int],
    figure_dpi: int,
    panel_zoom: float,
    anti_aliasing: str,
) -> None:
    """Compose fixed-camera PyVista renders with Matplotlib colorbars."""
    images = [
        render_surface_image(
            data,
            camera_position=camera_position,
            window_size=render_size,
            panel_zoom=panel_zoom,
            anti_aliasing=anti_aliasing,
        )
        for data in surfaces
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.1),
        constrained_layout=True,
    )

    for ax, data, image in zip(axes, surfaces, images, strict=True):
        ax.imshow(image)
        ax.set_axis_off()
        material_label = r"UO$_2$" if data.case.name == "UO2" else data.case.name
        ax.set_title(
            f"{material_label} (step {data.resolved_time_idx})",
            fontsize=18,
            pad=8,
        )

        norm = Normalize(vmin=data.scalar_range[0], vmax=data.scalar_range[1])
        mappable = ScalarMappable(norm=norm, cmap=cm.batlow)
        mappable.set_array([])
        colorbar = fig.colorbar(
            mappable,
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.018,
            aspect=28,
        )
        colorbar.set_label(
            r"$\sigma_{\mathrm{gr0,gr1}}$ (J m$^{-2}$)",
            fontsize=18,
            labelpad=13,
        )
        colorbar.ax.tick_params(labelsize=16, width=1.2, length=5)

        lower, upper = data.scalar_range
        print(
            f"{data.case.name} surface sigma range: "
            f"[{lower:.9g}, {upper:.9g}] J/m^2"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=figure_dpi,
        bbox_inches="tight",
        transparent=True
    )
    plt.close(fig)
    print(f"Matplotlib figure written to {output_path}")


def export_gr0_contour_figure(
    surfaces: Sequence[SurfaceData],
    *,
    camera_position: pv.CameraPosition,
    output_path: Path,
    render_size: Sequence[int],
    figure_dpi: int,
    panel_zoom: float,
    anti_aliasing: str,
    contour_color: str,
) -> None:
    """Compose fixed-camera PyVista renders of the bare gr0 = 0.5 shape.

    Identical rendering/fitting pipeline to ``export_matplotlib_figure``,
    except the surface is drawn as a plain shaded solid (no sigma scalar
    coloring, no colorbar) -- this is a geometry-only view of the isosurface.
    """
    images = [
        render_surface_image(
            data,
            camera_position=camera_position,
            window_size=render_size,
            panel_zoom=panel_zoom,
            anti_aliasing=anti_aliasing,
            scalars=None,
            color=contour_color,
        )
        for data in surfaces
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.1),
        constrained_layout=True,
    )

    for ax, data, image in zip(axes, surfaces, images, strict=True):
        ax.imshow(image)
        ax.set_axis_off()
        material_label = r"UO$_2$" if data.case.name == "UO2" else data.case.name
        ax.set_title(
            f"{material_label} (step {data.resolved_time_idx})",
            fontsize=18,
            pad=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=figure_dpi,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    print(f"gr0 contour figure written to {output_path}")


def print_available_times(case: RenderCase) -> None:
    """Print available gr0/sigma timestep suffixes for one file."""
    if not case.path.is_file():
        print(f"{case.name}: file not found: {case.path}")
        return

    with h5py.File(case.path, "r") as hf:
        eta_times = available_time_indices(hf, ETA_NAME)
        sigma_times = available_time_indices(hf, SIGMA_NAME)

    common = sorted(set(eta_times).intersection(sigma_times))
    print(f"{case.name}: {case.path}")
    print(f"  {ETA_NAME}: {eta_times}")
    print(f"  {SIGMA_NAME}: {sigma_times}")
    print(f"  common: {common}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ni-file", type=Path, default=Path("spherical_grain_Ni.h5")
    )
    parser.add_argument(
        "--uo2-file", type=Path, default=Path("spherical_grain.h5")
    )
    parser.add_argument(
        "--ni-time-index",
        type=int,
        default=-1,
        help="Ni timestep suffix; a negative value indexes available steps from the end",
    )
    parser.add_argument(
        "--uo2-time-index",
        type=int,
        default=-1,
        help="UO2 timestep suffix; a negative value indexes available steps from the end",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        default=(1.0, 1.0, 1.0),
        help="Uniform-grid point spacing interpreted along array axes 0, 1, and 2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spherical_grain_comparison.html"),
    )
    parser.add_argument(
        "--camera-output",
        type=Path,
        default=Path("spherical_grain_camera.json"),
    )
    parser.add_argument(
        "--camera-json",
        type=Path,
        help="Optional camera JSON to apply before rendering",
    )
    parser.add_argument(
        "--matplotlib-output",
        type=Path,
        help=(
            "Write a fixed-camera Matplotlib composite (for example, PNG or PDF) "
            "instead of the interactive HTML"
        ),
    )
    parser.add_argument(
        "--contour-output",
        type=Path,
        help=(
            f"Write a fixed-camera Matplotlib composite of the bare "
            f"{ETA_NAME} = {ETA_LEVEL} shape (no sigma coloring) at timestep "
            f"suffix {CONTOUR_TIME_INDEX} from both files, instead of the "
            "interactive HTML"
        ),
    )
    parser.add_argument(
        "--contour-color",
        type=str,
        default="lightgray",
        help="Solid shading color used for --contour-output (default: lightgray)",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1800, 1800),
        help="Pixel dimensions of each off-screen PyVista panel",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=300,
        help="DPI used for the Matplotlib output",
    )
    parser.add_argument(
        "--panel-zoom",
        type=float,
        default=1.08,
        help=(
            "Common post-fit zoom applied to each grain in the Matplotlib panels "
            "(default: 1.08)"
        ),
    )
    parser.add_argument(
        "--feature-angle",
        type=float,
        default=25.0,
        help=(
            "Normal-angle threshold in degrees for preserving sharp facet edges "
            "(default: 25)"
        ),
    )
    parser.add_argument(
        "--anti-aliasing",
        choices=("ssaa", "msaa", "fxaa", "none"),
        default="ssaa",
        help="Anti-aliasing used for off-screen Matplotlib panels (default: ssaa)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1800, 850),
    )
    parser.add_argument(
        "--list-times",
        action="store_true",
        help="List stored timestep suffixes and exit",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open a native interactive window; press c to save the camera",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = (
        RenderCase("Ni", args.ni_file, args.ni_time_index),
        RenderCase("UO2", args.uo2_file, args.uo2_time_index),
    )

    if args.list_times:
        for case in cases:
            print_available_times(case)
        return

    camera_position = load_camera(args.camera_json) if args.camera_json else None
    if not np.isfinite(args.feature_angle) or not 0.0 < args.feature_angle <= 180.0:
        raise ValueError("--feature-angle must be in the interval (0, 180]")

    if args.contour_output is not None:
        if camera_position is None:
            raise ValueError(
                "--contour-output requires --camera-json so the final view "
                "is explicit and reproducible"
            )
        if not np.isfinite(args.panel_zoom) or args.panel_zoom <= 0.0:
            raise ValueError("--panel-zoom must be a finite positive number")
        contour_cases = (
            RenderCase("Ni", args.ni_file, CONTOUR_TIME_INDEX),
            RenderCase("UO2", args.uo2_file, CONTOUR_TIME_INDEX),
        )
        contour_surfaces = [
            extract_surface(case, args.spacing, args.feature_angle)
            for case in contour_cases
        ]
        export_gr0_contour_figure(
            contour_surfaces,
            camera_position=camera_position,
            output_path=args.contour_output,
            render_size=args.render_size,
            figure_dpi=args.figure_dpi,
            panel_zoom=args.panel_zoom,
            anti_aliasing=args.anti_aliasing,
            contour_color=args.contour_color,
        )
        return

    surfaces = [
        extract_surface(case, args.spacing, args.feature_angle) for case in cases
    ]

    if args.matplotlib_output is not None:
        if camera_position is None:
            raise ValueError(
                "--matplotlib-output requires --camera-json so the final view "
                "is explicit and reproducible"
            )
        if not np.isfinite(args.panel_zoom) or args.panel_zoom <= 0.0:
            raise ValueError("--panel-zoom must be a finite positive number")
        export_matplotlib_figure(
            surfaces,
            camera_position=camera_position,
            output_path=args.matplotlib_output,
            render_size=args.render_size,
            figure_dpi=args.figure_dpi,
            panel_zoom=args.panel_zoom,
            anti_aliasing=args.anti_aliasing,
        )
        return

    plotter = build_plotter(
        surfaces,
        window_size=args.window_size,
        off_screen=not args.gui,
        camera_position=camera_position,
    )

    if args.gui:
        def capture_camera() -> None:
            save_camera(plotter, args.camera_output)

        plotter.add_key_event("c", capture_camera)
        print("Rotate the scene and press 'c' to save the camera.")
        plotter.show(auto_close=False)
        save_camera(plotter, args.camera_output)
        plotter.close()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() != ".html":
        args.output = args.output.with_suffix(".html")
    plotter.export_html(args.output)
    save_camera(plotter, args.camera_output)
    plotter.close()
    print(f"Interactive HTML written to {args.output}")


if __name__ == "__main__":
    main()