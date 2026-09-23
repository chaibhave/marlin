#!/usr/bin/env python3
# *                    DO NOT MODIFY THIS HEADER
# *            Marlin, a Fourier spectral solver for MOOSE
# *
# *            Copyright 2024 Battelle Energy Alliance, LLC
# *                        ALL RIGHTS RESERVED
# *
# *        Licensed under LGPL 2.1, please see LICENSE for details
# *             https://www.gnu.org/licenses/lgpl-2.1.html

"""Measure convergence of Ni grain-boundary convex-hull energies.

For Sigma3, Sigma5, and one reproducible random misorientation, this script:

1. samples the reciprocal BRK energy surface with Fibonacci sphere points;
2. constructs raw three-dimensional convex hulls for N = 10, ..., 10^7;
3. evaluates every hull at the same 1,000 random inclination directions; and
4. reports RMS percentage error relative to the N = 10^7 hull.

The reference hull is built first for each boundary. Only its predictions at
the evaluation directions are retained; the hull itself is not stored.
"""

from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import ConvexHull

import GrainBoundaryEnergyHull
from GB5DOF import GB5DOF


SAMPLE_SIZES = (10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
BOUNDARY_ORDER = ("sigma3", "sigma5", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="BRK evaluation batch size (default: 100000)",
    )
    parser.add_argument(
        "--num-evaluation-directions",
        type=int,
        default=1_000,
        help="Number of random normals used for hull lookup (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260923,
        help="Seed for the random boundary and evaluation normals",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=SAMPLE_SIZES,
        help="Hull sample counts; the largest is the reference",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("figures/bulatov/ni_hull_sampling_convergence"),
        help="Prefix for PDF, PNG, CSV, and NPZ outputs",
    )
    return parser.parse_args()


def fibonacci_sphere_chunk(
    start: int,
    stop: int,
    total: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return Fibonacci sphere points ``[start:stop]`` from a total of N."""
    index = torch.arange(start, stop, device=device, dtype=dtype)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (index + 0.5) / total
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    azimuth = 2.0 * torch.pi * index / golden_ratio
    return torch.stack(
        (radius * torch.cos(azimuth), radius * torch.sin(azimuth), z),
        dim=1,
    )


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a reproducible Haar-uniform rotation from a unit quaternion."""
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_boundaries(
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    identity = np.eye(3, dtype=np.float64)

    # Sigma3 coherent-twin misorientation: 60 degrees about [111].
    sigma3 = np.array(
        [
            [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0],
            [-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
        ],
        dtype=np.float64,
    )

    # Sigma5 misorientation: 36.86989765 degrees about [001].
    sigma5 = np.array(
        [[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    rng = np.random.default_rng(seed)
    random_q = random_rotation(rng)

    def tensor(matrix: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(matrix, dtype=dtype, device=device)

    p = tensor(identity)
    return {
        "sigma3": (p, tensor(sigma3)),
        "sigma5": (p, tensor(sigma5)),
        "random": (p, tensor(random_q)),
    }


def random_unit_normals(count: int, seed: int) -> np.ndarray:
    """Generate random lookup normals independently of Fibonacci sampling."""
    rng = np.random.default_rng(seed)
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


@torch.inference_mode()
def sample_reciprocal_surface(
    model: GB5DOF,
    p: torch.Tensor,
    q: torch.Tensor,
    sample_count: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    """Evaluate ``n / gamma(n)`` at Fibonacci directions in bounded memory."""
    points = np.empty((sample_count, 3), dtype=np.float64)
    progress_interval = max(chunk_size, sample_count // 10)
    next_progress = progress_interval

    for start in range(0, sample_count, chunk_size):
        stop = min(start + chunk_size, sample_count)
        normals = fibonacci_sphere_chunk(
            start, stop, sample_count, device=device, dtype=dtype
        )
        lab_rotations = GrainBoundaryEnergyHull.lab_rotations_from_normals(normals)
        lab_rotations = lab_rotations.to(device=device, dtype=dtype)
        energy = model(lab_rotations @ p, lab_rotations @ q).reshape(-1)

        if not torch.all(torch.isfinite(energy)) or torch.any(energy <= 0.0):
            raise RuntimeError("GB5DOF returned a non-finite or non-positive energy")

        points[start:stop] = (
            normals / energy[:, None]
        ).detach().cpu().numpy()

        if stop >= next_progress or stop == sample_count:
            print(f"      sampled {stop:,} / {sample_count:,}", flush=True)
            next_progress += progress_interval

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return points


def build_hull_equations(points: np.ndarray) -> np.ndarray:
    """Build the raw reciprocal-space convex hull and retain only its planes."""
    hull = ConvexHull(points)
    equations = np.array(hull.equations, dtype=np.float64, copy=True)
    if np.any(equations[:, 3] >= 0.0):
        raise RuntimeError("The reciprocal-space hull does not contain the origin")
    return equations


def evaluate_hull(equations: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Evaluate the exact radial hull energy without a large dense allocation."""
    energy = np.full(len(normals), -np.inf, dtype=np.float64)
    normal_chunk_size = 100
    facet_chunk_size = 50_000

    for normal_start in range(0, len(normals), normal_chunk_size):
        normal_stop = min(normal_start + normal_chunk_size, len(normals))
        normal_chunk = normals[normal_start:normal_stop]
        chunk_energy = np.full(len(normal_chunk), -np.inf, dtype=np.float64)

        for facet_start in range(0, len(equations), facet_chunk_size):
            facet_stop = min(facet_start + facet_chunk_size, len(equations))
            facets = equations[facet_start:facet_stop]
            projections = normal_chunk @ facets[:, :3].T
            candidates = np.divide(
                projections,
                -facets[None, :, 3],
                out=np.full_like(projections, -np.inf),
                where=projections > 0.0,
            )
            chunk_energy = np.maximum(chunk_energy, np.max(candidates, axis=1))

        energy[normal_start:normal_stop] = chunk_energy

    if not np.all(np.isfinite(energy)) or np.any(energy <= 0.0):
        raise RuntimeError("Hull lookup produced an invalid energy")
    return energy


def rmspe(prediction: np.ndarray, reference: np.ndarray) -> float:
    relative_error = (prediction - reference) / reference
    return 100.0 * float(np.sqrt(np.mean(relative_error * relative_error)))


def save_results(filename: Path, records: list[dict[str, float | int | str]]) -> None:
    with filename.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("boundary", "sample_count", "rmspe_percent", "hull_facets"),
        )
        writer.writeheader()
        writer.writerows(records)


def save_reference_predictions(
    filename: Path,
    evaluation_normals: np.ndarray,
    predictions: dict[str, np.ndarray],
    boundaries: dict[str, tuple[torch.Tensor, torch.Tensor]],
    reference_sample_count: int,
    seed: int,
) -> None:
    contents: dict[str, np.ndarray | int] = {
        "evaluation_normals": evaluation_normals,
        "reference_sample_count": reference_sample_count,
        "seed": seed,
    }
    for name, (p, q) in boundaries.items():
        contents[f"P_{name}"] = p.detach().cpu().numpy()
        contents[f"Q_{name}"] = q.detach().cpu().numpy()
    contents.update(
        {f"energy_{name}": values for name, values in predictions.items()}
    )
    np.savez_compressed(filename, **contents)


def plot_convergence(
    records: list[dict[str, float | int | str]],
    output_prefix: Path,
    sample_sizes: tuple[int, ...],
) -> None:
    style = {
        "font.family": "serif",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
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
        "legend.fontsize": 7.0,
        "savefig.dpi": 600,
    }
    appearance = {
        "sigma3": (r"$\Sigma3$", "#0072B2", "o"),
        "sigma5": (r"$\Sigma5$", "#D55E00", "s"),
        "random": ("Random", "#009E73", "^"),
    }

    with mpl.rc_context(style):
        figure, axes = plt.subplots(figsize=(3.35, 2.75), constrained_layout=True)
        for boundary in BOUNDARY_ORDER:
            selected = sorted(
                (row for row in records if row["boundary"] == boundary),
                key=lambda row: int(row["sample_count"]),
            )
            label, color, marker = appearance[boundary]
            axes.plot(
                [int(row["sample_count"]) for row in selected],
                [float(row["rmspe_percent"]) for row in selected],
                color=color,
                marker=marker,
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linewidth=1.1,
                label=label,
            )

        axes.set_xscale("log")
        axes.set_xlabel(r"Fibonacci samples, $N$")
        axes.set_ylabel(r"RMSPE (\%)")
        axes.set_xlim(min(sample_sizes) / 1.25, max(sample_sizes) * 1.25)
        axes.set_ylim(bottom=0.0)
        axes.minorticks_on()
        axes.legend(frameon=False, handlelength=1.8)
        axes.grid(axis="y", color="0.9", linewidth=0.5)

        figure.savefig(output_prefix.with_suffix(".pdf"))
        figure.savefig(output_prefix.with_suffix(".png"))
        plt.close(figure)


def main() -> None:
    args = parse_args()
    sample_sizes = tuple(sorted(set(args.sample_sizes)))
    if sample_sizes[0] < 4:
        raise ValueError("ConvexHull requires at least four sample directions")
    if args.chunk_size <= 0 or args.num_evaluation_directions <= 0:
        raise ValueError("Chunk size and evaluation-direction count must be positive")

    device = torch.device(args.device)
    dtype = torch.float64
    model = GB5DOF(system="Ni", device=device, dtype=dtype).eval()
    boundaries = make_boundaries(device, dtype, args.seed)
    evaluation_normals = random_unit_normals(
        args.num_evaluation_directions, args.seed + 1
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_filename = args.output_prefix.with_suffix(".csv")
    reference_filename = args.output_prefix.with_name(
        args.output_prefix.name + "_reference_predictions.npz"
    )

    records: list[dict[str, float | int | str]] = []
    reference_predictions: dict[str, np.ndarray] = {}
    reference_count = max(sample_sizes)

    for boundary_name in BOUNDARY_ORDER:
        p, q = boundaries[boundary_name]
        print(f"\n{'=' * 72}\n{boundary_name}: reference N = {reference_count:,}\n{'=' * 72}")

        for sample_count in sorted(sample_sizes, reverse=True):
            print(f"  Building N = {sample_count:,} hull", flush=True)
            points = sample_reciprocal_surface(
                model, p, q, sample_count, args.chunk_size, device, dtype
            )
            equations = build_hull_equations(points)
            del points

            prediction = evaluate_hull(equations, evaluation_normals)
            facet_count = len(equations)
            del equations
            gc.collect()

            if sample_count == reference_count:
                reference_predictions[boundary_name] = prediction
                error = 0.0
                save_reference_predictions(
                    reference_filename,
                    evaluation_normals,
                    reference_predictions,
                    boundaries,
                    reference_count,
                    args.seed,
                )
            else:
                error = rmspe(prediction, reference_predictions[boundary_name])

            records.append(
                {
                    "boundary": boundary_name,
                    "sample_count": sample_count,
                    "rmspe_percent": error,
                    "hull_facets": facet_count,
                }
            )
            save_results(csv_filename, records)
            print(
                f"    facets = {facet_count:,}; RMSPE = {error:.6e}%",
                flush=True,
            )

    plot_convergence(records, args.output_prefix, sample_sizes)
    print(f"\nWrote {csv_filename}")
    print(f"Wrote {reference_filename}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
