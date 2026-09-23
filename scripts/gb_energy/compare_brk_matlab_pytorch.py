#!/usr/bin/env python3
"""Compare MATLAB and vectorized PyTorch BRK energies on identical inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat, savemat

from GB5DOF import GB5DOF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load P, Q, and MATLAB BRK energies from a MAT file, evaluate the "
            "same boundaries with the vectorized PyTorch model, and create "
            "parity and residual plots."
        )
    )
    parser.add_argument(
        "reference",
        type=Path,
        nargs="?",
        default=Path("brk_matlab_reference.mat"),
        help="MATLAB reference file produced by generate_matlab_brk_reference.m",
    )
    parser.add_argument(
        "--material",
        default=None,
        help="Override the material stored in the MAT file, e.g. Cu or Ni",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Maximum number of boundaries evaluated in one PyTorch batch",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/bulatov/brk_matlab_pytorch_comparison"),
        help=(
            "Output prefix. Figures are written as PREFIX_parity.* and "
            "PREFIX_residual.* (default directory: figures/bulatov)"
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "pgf"),
        default=("pdf", "png", "pgf"),
        help="Figure formats to write (default: pdf png pgf)",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=3.35,
        help="Width of each separate figure in inches (default: 3.35)",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=2.75,
        help="Height of each separate figure in inches (default: 2.75)",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=8.0,
        help="Base font size in points (default: 8)",
    )
    parser.add_argument(
        "--pgf-texsystem",
        choices=("pdflatex", "xelatex", "lualatex"),
        default="pdflatex",
        help="TeX engine used to measure text in PGF output",
    )
    parser.add_argument(
        "--pgf-preamble",
        default="",
        help=(
            "Optional LaTeX preamble used while generating PGF, for example "
            "r'\\usepackage{newtxtext,newtxmath}'"
        ),
    )
    parser.add_argument(
        "--show-titles",
        action="store_true",
        help="Put a title above each axes (normally supplied by LaTeX captions)",
    )
    parser.add_argument(
        "--orientation-convention",
        choices=("as-saved", "transpose", "swap", "swap-transpose", "auto"),
        default="as-saved",
        help=(
            "Input convention used for the plotted comparison. All four fixed "
            "conventions are diagnosed regardless. 'auto' plots the convention "
            "with the smallest RMSE; use it only as a diagnostic."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of largest-error cases to export to a MAT file",
    )
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    return parser.parse_args()


def matlab_string(value: object) -> str:
    """Convert common scipy.io.loadmat string layouts to a Python string."""
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        return "".join(array.ravel().tolist()).strip()
    if array.dtype == object:
        return matlab_string(array.ravel()[0])
    return str(array.squeeze())


def to_batch(array: np.ndarray, name: str) -> np.ndarray:
    """Convert MATLAB 3x3xN or Python Nx3x3 storage to Nx3x3."""
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 2 and array.shape == (3, 3):
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[:2] == (3, 3):
        array = np.moveaxis(array, 2, 0)
    elif array.ndim != 3 or array.shape[1:] != (3, 3):
        raise ValueError(
            f"{name} must have shape (3, 3, N) or (N, 3, 3); got {array.shape}"
        )
    return np.ascontiguousarray(array)


def rotation_errors(rotations: np.ndarray) -> tuple[float, float]:
    identity = np.eye(3, dtype=np.float64)
    gram = np.swapaxes(rotations, 1, 2) @ rotations
    orthogonality = np.linalg.norm(gram - identity, axis=(1, 2))
    determinant = np.abs(np.linalg.det(rotations) - 1.0)
    return float(orthogonality.max()), float(determinant.max())


def evaluate_batched(
    model: GB5DOF,
    p_values: np.ndarray,
    q_values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(p_values), batch_size):
            stop = min(start + batch_size, len(p_values))
            p_batch = torch.as_tensor(
                p_values[start:stop], dtype=torch.float64, device=device
            )
            q_batch = torch.as_tensor(
                q_values[start:stop], dtype=torch.float64, device=device
            )
            energy = model(p_batch, q_batch)
            predictions.append(energy.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions)


def apply_orientation_convention(
    p_values: np.ndarray,
    q_values: np.ndarray,
    convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply candidate row/column and grain-order conventions."""
    if convention == "as-saved":
        p_test, q_test = p_values, q_values
    elif convention == "transpose":
        p_test = np.swapaxes(p_values, 1, 2)
        q_test = np.swapaxes(q_values, 1, 2)
    elif convention == "swap":
        p_test, q_test = q_values, p_values
    elif convention == "swap-transpose":
        p_test = np.swapaxes(q_values, 1, 2)
        q_test = np.swapaxes(p_values, 1, 2)
    else:
        raise ValueError(f"Unknown orientation convention: {convention}")
    return np.ascontiguousarray(p_test), np.ascontiguousarray(q_test)


def calculate_metrics(
    matlab_energy: np.ndarray,
    torch_energy: np.ndarray,
) -> dict[str, float]:
    error = torch_energy - matlab_energy
    denominator = float(np.sum((matlab_energy - matlab_energy.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
        "max_relative": float(
            np.max(
                np.abs(error)
                / np.maximum(np.abs(matlab_energy), np.finfo(np.float64).tiny)
            )
        ),
        "mean_signed": float(np.mean(error)),
        "r_squared": (
            1.0 - float(np.sum(error**2)) / denominator
            if denominator > 0.0
            else float("nan")
        ),
    }


def save_convention_metrics(
    filename: Path,
    results: dict[str, tuple[np.ndarray, dict[str, float]]],
) -> None:
    fields = ("mae", "rmse", "max_abs", "mean_signed", "r_squared")
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("convention", *fields))
        for convention, (_, metrics) in results.items():
            writer.writerow((convention, *(metrics[field] for field in fields)))


def save_worst_cases(
    filename: Path,
    p_original: np.ndarray,
    q_original: np.ndarray,
    p_evaluated: np.ndarray,
    q_evaluated: np.ndarray,
    matlab_energy: np.ndarray,
    torch_energy: np.ndarray,
    top_k: int,
    convention: str,
) -> None:
    count = min(top_k, len(matlab_energy))
    indices = np.argsort(np.abs(torch_energy - matlab_energy))[-count:][::-1]
    savemat(
        filename,
        {
            # Store matrices as 3x3xK for convenient MATLAB indexing.
            "P_original": np.moveaxis(p_original[indices], 0, 2),
            "Q_original": np.moveaxis(q_original[indices], 0, 2),
            "P_evaluated": np.moveaxis(p_evaluated[indices], 0, 2),
            "Q_evaluated": np.moveaxis(q_evaluated[indices], 0, 2),
            "energy_matlab": matlab_energy[indices, None],
            "energy_pytorch": torch_energy[indices, None],
            "signed_error": (torch_energy - matlab_energy)[indices, None],
            "sample_index": (indices + 1)[:, None],
            "orientation_convention": convention,
        },
    )


def save_csv(
    filename: Path,
    matlab_energy: np.ndarray,
    torch_energy: np.ndarray,
) -> None:
    error = torch_energy - matlab_energy
    relative_error = np.divide(
        error,
        matlab_energy,
        out=np.full_like(error, np.nan),
        where=np.abs(matlab_energy) > np.finfo(np.float64).tiny,
    )
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample",
                "energy_matlab_J_per_m2",
                "energy_pytorch_J_per_m2",
                "signed_error_J_per_m2",
                "relative_error",
            ]
        )
        for index, values in enumerate(
            zip(matlab_energy, torch_energy, error, relative_error), start=1
        ):
            writer.writerow((index, *values))


def publication_style(
    font_size: float,
    pgf_texsystem: str,
    pgf_preamble: str,
) -> dict[str, object]:
    """Matplotlib settings suitable for compact, single-column figures."""
    return {
        "font.family": "serif",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "axes.linewidth": 0.7,
        "axes.formatter.use_mathtext": True,
        "xtick.labelsize": font_size - 1.0,
        "ytick.labelsize": font_size - 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "lines.linewidth": 1.0,
        "savefig.dpi": 600,
        "savefig.transparent": False,
        # PGF text remains editable and inherits the document's font family.
        # Supply the template's font packages through --pgf-preamble when their
        # text metrics differ materially from standard LaTeX fonts.
        "pgf.rcfonts": False,
        "pgf.texsystem": pgf_texsystem,
        "pgf.preamble": pgf_preamble,
    }


def save_figure_formats(
    figure: mpl.figure.Figure,
    stem: Path,
    formats: tuple[str, ...] | list[str],
) -> list[Path]:
    """Save each format, embedding all plotted points directly in PGF."""
    written: list[Path] = []
    collections = [
        collection
        for axes in figure.axes
        for collection in axes.collections
    ]
    for file_format in formats:
        filename = stem.with_suffix(f".{file_format}")
        # Dense point clouds remain rasterized in PDF/PNG, but PGF stores every
        # marker as a vector object so the PGF is completely self-contained.
        for collection in collections:
            collection.set_rasterized(file_format != "pgf")
        try:
            figure.savefig(filename)
        except Exception as error:
            if file_format != "pgf":
                raise
            print(
                f"WARNING: Could not write {filename}: {error}\n"
                "         Install/configure the selected TeX engine or omit "
                "'pgf' from --formats."
            )
        else:
            written.append(filename)
            if file_format == "pgf":
                pgf_text = filename.read_text(encoding="utf-8")
                pgf_text = pgf_text.replace(
                    "\\begingroup%\n",
                    "\\begingroup%\n"
                    "\\providecommand{\\mathdefault}[1]{#1}%\n",
                    1,
                )
                filename.write_text(pgf_text, encoding="utf-8")
    return written


def make_figures(
    matlab_energy: np.ndarray,
    torch_energy: np.ndarray,
    output_prefix: Path,
    metrics: dict[str, float],
    formats: tuple[str, ...] | list[str],
    figure_width: float,
    figure_height: float,
    font_size: float,
    pgf_texsystem: str,
    pgf_preamble: str,
    show_titles: bool,
) -> list[Path]:
    """Create independent parity and residual figures for LaTeX placement."""
    error = torch_energy - matlab_energy
    lower = float(min(matlab_energy.min(), torch_energy.min()))
    upper = float(max(matlab_energy.max(), torch_energy.max()))
    padding = max(0.02 * (upper - lower), 1.0e-12)
    limits = (lower - padding, upper + padding)

    summary = (
        f"$N={len(matlab_energy):,}$\n"
        f"MAE = {metrics['mae']:.3e}\n"
        f"RMSE = {metrics['rmse']:.3e}\n"
        f"max $|e|$ = {metrics['max_abs']:.3e}\n"
        f"$R^2$ = {metrics['r_squared']:.10f}"
    )

    style = publication_style(font_size, pgf_texsystem, pgf_preamble)
    written: list[Path] = []
    with mpl.rc_context(style):
        parity_figure, parity_axes = plt.subplots(
            figsize=(figure_width, figure_height), constrained_layout=True
        )
        parity_axes.scatter(
            matlab_energy,
            torch_energy,
            s=7,
            alpha=0.45,
            edgecolors="none",
            color="#3569d4",
            rasterized=True,
        )
        parity_axes.plot(
            limits,
            limits,
            color="black",
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            zorder=3,
        )
        parity_axes.set_xlim(limits)
        parity_axes.set_ylim(limits)
        parity_axes.set_aspect("equal", adjustable="box")
        parity_axes.set_xlabel(r"MATLAB energy ($\mathrm{J\,m^{-2}}$)")
        parity_axes.set_ylabel(r"PyTorch energy ($\mathrm{J\,m^{-2}}$)")
        parity_axes.minorticks_on()
        if show_titles:
            parity_axes.set_title("Energy parity", pad=5)
        parity_axes.text(
            0.04,
            0.96,
            summary,
            transform=parity_axes.transAxes,
            va="top",
            ha="left",
            fontsize=font_size - 0.5,
            linespacing=1.15,
            bbox={
                "boxstyle": "square,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.7",
                "linewidth": 0.6,
                "alpha": 0.92,
            },
        )
        written.extend(
            save_figure_formats(
                parity_figure,
                output_prefix.with_name(output_prefix.name + "_parity"),
                formats,
            )
        )
        plt.close(parity_figure)

        residual_figure, residual_axes = plt.subplots(
            figsize=(figure_width, figure_height), constrained_layout=True
        )
        residual_axes.scatter(
            matlab_energy,
            error,
            s=7,
            alpha=0.35,
            edgecolors="none",
            color="#d44a3a",
            rasterized=True,
        )
        residual_axes.axhline(
            0.0,
            color="black",
            linewidth=0.9,
            linestyle=(0, (4, 2)),
            zorder=3,
        )
        residual_axes.set_xlim(limits)
        residual_limit = 1.05 * float(np.max(np.abs(error)))
        if residual_limit == 0.0:
            residual_limit = np.finfo(np.float64).eps
        residual_axes.set_ylim(-residual_limit, residual_limit)
        residual_axes.set_xlabel(r"MATLAB energy ($\mathrm{J\,m^{-2}}$)")
        residual_axes.set_ylabel(
            r"$E_{\mathrm{PyTorch}}-E_{\mathrm{MATLAB}}$ "
            r"($\mathrm{J\,m^{-2}}$)"
        )
        residual_axes.ticklabel_format(
            axis="y", style="sci", scilimits=(0, 0), useMathText=True
        )
        residual_axes.minorticks_on()
        if show_titles:
            residual_axes.set_title("Signed residual", pad=5)
        written.extend(
            save_figure_formats(
                residual_figure,
                output_prefix.with_name(output_prefix.name + "_residual"),
                formats,
            )
        )
        plt.close(residual_figure)

    return written


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.figure_width <= 0.0 or args.figure_height <= 0.0:
        raise ValueError("--figure-width and --figure-height must be positive")
    if args.font_size <= 1.0:
        raise ValueError("--font-size must be greater than 1 point")

    reference = loadmat(args.reference)
    required = {"P", "Q", "energy_matlab"}
    missing = required - reference.keys()
    if missing:
        raise KeyError(f"Missing variables in {args.reference}: {sorted(missing)}")

    p_values = to_batch(reference["P"], "P")
    q_values = to_batch(reference["Q"], "Q")
    matlab_energy = np.asarray(reference["energy_matlab"], dtype=np.float64).reshape(-1)

    if not (len(p_values) == len(q_values) == len(matlab_energy)):
        raise ValueError(
            "P, Q, and energy_matlab contain different numbers of samples: "
            f"{len(p_values)}, {len(q_values)}, {len(matlab_energy)}"
        )
    if not np.all(np.isfinite(matlab_energy)):
        raise ValueError("The MATLAB reference contains non-finite energies")

    material = args.material
    if material is None:
        material = matlab_string(reference.get("material", np.array(["Cu"])))

    p_orth, p_det = rotation_errors(p_values)
    q_orth, q_det = rotation_errors(q_values)

    device = torch.device(args.device)
    model = GB5DOF(system=material, device=device, dtype=torch.float64).eval()

    conventions = ("as-saved", "transpose", "swap", "swap-transpose")
    convention_results: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    for convention in conventions:
        p_test, q_test = apply_orientation_convention(
            p_values, q_values, convention
        )
        prediction = evaluate_batched(
            model, p_test, q_test, device=device, batch_size=args.batch_size
        )
        convention_results[convention] = (
            prediction,
            calculate_metrics(matlab_energy, prediction),
        )

    best_convention = min(
        conventions, key=lambda name: convention_results[name][1]["rmse"]
    )
    selected_convention = (
        best_convention
        if args.orientation_convention == "auto"
        else args.orientation_convention
    )
    torch_energy, metrics = convention_results[selected_convention]
    p_evaluated, q_evaluated = apply_orientation_convention(
        p_values, q_values, selected_convention
    )

    mae = metrics["mae"]
    rmse = metrics["rmse"]
    max_abs = metrics["max_abs"]
    max_relative = metrics["max_relative"]
    r_squared = metrics["r_squared"]
    agrees = bool(
        np.allclose(torch_energy, matlab_energy, atol=args.atol, rtol=args.rtol)
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_prefix.with_suffix(".csv"), matlab_energy, torch_energy)
    save_convention_metrics(
        args.output_prefix.with_name(args.output_prefix.name + "_conventions.csv"),
        convention_results,
    )
    save_worst_cases(
        args.output_prefix.with_name(args.output_prefix.name + "_worst_cases.mat"),
        p_values,
        q_values,
        p_evaluated,
        q_evaluated,
        matlab_energy,
        torch_energy,
        args.top_k,
        selected_convention,
    )
    figure_files = make_figures(
        matlab_energy,
        torch_energy,
        args.output_prefix,
        metrics,
        args.formats,
        args.figure_width,
        args.figure_height,
        args.font_size,
        args.pgf_texsystem,
        args.pgf_preamble,
        args.show_titles,
    )

    print(f"Reference file: {args.reference}")
    print(f"Samples: {len(matlab_energy)}")
    print(f"Material: {material}")
    print(f"Device: {device}")
    print(f"Maximum P orthogonality error: {p_orth:.3e}")
    print(f"Maximum Q orthogonality error: {q_orth:.3e}")
    print(f"Maximum P determinant error: {p_det:.3e}")
    print(f"Maximum Q determinant error: {q_det:.3e}")
    print("\nOrientation-convention diagnostic:")
    print("convention          MAE           RMSE          max |e|       mean signed")
    for convention in conventions:
        diagnostic = convention_results[convention][1]
        marker = " *" if convention == best_convention else ""
        print(
            f"{convention:16s}  {diagnostic['mae']:.6e}  "
            f"{diagnostic['rmse']:.6e}  {diagnostic['max_abs']:.6e}  "
            f"{diagnostic['mean_signed']:.6e}{marker}"
        )
    print("* smallest RMSE (diagnostic only)")
    print(f"Selected convention for outputs: {selected_convention}")
    print(f"Mean absolute energy error: {mae:.6e} J/m^2")
    print(f"RMSE: {rmse:.6e} J/m^2")
    print(f"Maximum absolute energy error: {max_abs:.6e} J/m^2")
    print(f"Maximum relative energy error: {max_relative:.6e}")
    print(f"R^2: {r_squared:.12g}")
    print(f"allclose(atol={args.atol:g}, rtol={args.rtol:g}): {agrees}")
    for filename in figure_files:
        print(f"Wrote {filename}")
    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(
        "Wrote "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_conventions.csv')}"
    )
    print(
        "Wrote "
        f"{args.output_prefix.with_name(args.output_prefix.name + '_worst_cases.mat')}"
    )


if __name__ == "__main__":
    main()
