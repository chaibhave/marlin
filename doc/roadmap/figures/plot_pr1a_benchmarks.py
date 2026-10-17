#!/usr/bin/env python3
"""Generate PR1a benchmark figures directly from the gtest log files.

Reads doc/roadmap/logs/petsc_torch_benchmark_{cpu,cuda}.log and produces:
  - timing_sweep.pdf: torch-native vs. PETSc zero-copy matrix-free solve
    wall-clock time vs. problem size n, CPU and CUDA, log-log.
  - anisotropic_amg_vs_jacobi.pdf: Jacobi vs. AMG iteration count and
    KSPSolve wall-clock time on the ill-conditioned (anisotropic) case,
    CPU and CUDA.

Run from doc/roadmap/figures/: python3 plot_pr1a_benchmarks.py
"""

import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"

COLOR_TORCH = "#DE8F05"
COLOR_PETSC = "#0173B2"

matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def parse_timing_sweep(log_text, label):
    """Parse the DISABLED_TimingSweep / DISABLED_CudaTimingSweep block.

    label is the literal text following "PETSc" and preceding the colon,
    e.g. " (zero-copy Vec)" or " CUDA (zero-copy Vec)".
    """
    pattern = re.compile(
        r"n=(\d+) torch-native: ([\d.]+)ms, \d+ iters \| PETSc"
        + re.escape(label)
        + r": ([\d.]+)ms, \d+ iters"
    )
    ns, torch_ms, petsc_ms = [], [], []
    for m in pattern.finditer(log_text):
        ns.append(int(m.group(1)))
        torch_ms.append(float(m.group(2)))
        petsc_ms.append(float(m.group(3)))
    return ns, torch_ms, petsc_ms


def parse_anisotropic(log_text, test_name):
    """Parse the AnisotropicAMGConvergesFasterThanJacobi(-Cuda) line."""
    pattern = re.compile(
        r"\[ RUN      \] PetscTorchMatrixFreeBenchmarkTest\."
        + re.escape(test_name)
        + r"\nanisotropic.*?: Jacobi (\d+) its, ([\d.]+)ms \| AMG \(PCGAMG\) (\d+) its, "
        r"([\d.]+)ms solve \+ ([\d.]+)ms Pmat assembly"
    )
    m = pattern.search(log_text)
    if not m:
        raise ValueError(f"Could not find {test_name} result in log")
    jacobi_its, jacobi_ms, amg_its, amg_solve_ms, amg_assembly_ms = m.groups()
    return {
        "jacobi_its": int(jacobi_its),
        "jacobi_ms": float(jacobi_ms),
        "amg_its": int(amg_its),
        "amg_solve_ms": float(amg_solve_ms),
        "amg_assembly_ms": float(amg_assembly_ms),
    }


def plot_timing_sweep(cpu_log, cuda_log):
    n_cpu, torch_cpu, petsc_cpu = parse_timing_sweep(cpu_log, " (zero-copy Vec)")
    n_cuda, torch_cuda, petsc_cuda = parse_timing_sweep(
        cuda_log, " CUDA (zero-copy Vec)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)

    ax = axes[0]
    ax.loglog(n_cpu, torch_cpu, "o-", color=COLOR_TORCH, label="torch-native")
    ax.loglog(n_cpu, petsc_cpu, "s-", color=COLOR_PETSC, label="PETSc (zero-copy)")
    ax.set_xlabel(r"$n$ (grid points per side)")
    ax.set_ylabel("CG solve wall-clock time (ms)")
    ax.set_title("CPU")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    # CUDA times span under one decade (~4 to ~32 ms), so a linear y-axis from
    # zero shows the launch-latency floor and the growth honestly; log-x stays
    # because n doubles between points.
    ax = axes[1]
    ax.semilogx(n_cuda, torch_cuda, "o-", color=COLOR_TORCH, label="torch-native")
    ax.semilogx(n_cuda, petsc_cuda, "s-", color=COLOR_PETSC, label="PETSc (zero-copy)")
    ax.set_ylim(0, max(petsc_cuda + torch_cuda) * 1.08)
    ax.set_xlabel(r"$n$ (grid points per side)")
    ax.set_ylabel("CG solve wall-clock time (ms)")
    ax.set_title("CUDA")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "timing_sweep.pdf")
    plt.close(fig)


def plot_anisotropic(cpu_log, cuda_log):
    cpu = parse_anisotropic(cpu_log, "AnisotropicAMGConvergesFasterThanJacobi")
    cuda = parse_anisotropic(cuda_log, "AnisotropicCudaAMGConvergesFasterThanJacobi")

    devices = ["CPU", "CUDA"]
    jacobi_its = [cpu["jacobi_its"], cuda["jacobi_its"]]
    amg_its = [cpu["amg_its"], cuda["amg_its"]]
    jacobi_ms = [cpu["jacobi_ms"], cuda["jacobi_ms"]]
    amg_total_ms = [
        cpu["amg_solve_ms"] + cpu["amg_assembly_ms"],
        cuda["amg_solve_ms"] + cuda["amg_assembly_ms"],
    ]

    x = range(len(devices))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    ax.bar(
        [i - width / 2 for i in x], jacobi_its, width, color=COLOR_TORCH, label="Jacobi"
    )
    ax.bar(
        [i + width / 2 for i in x],
        amg_its,
        width,
        color=COLOR_PETSC,
        label="AMG (PCGAMG)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(devices)
    ax.set_ylabel("CG iterations")
    ax.set_title("Iteration count")
    ax.set_ylim(0, max(jacobi_its) * 1.3)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    for i, v in zip(x, jacobi_its):
        ax.text(i - width / 2, v + 3, str(v), ha="center", va="bottom", fontsize=7)
    for i, v in zip(x, amg_its):
        ax.text(i + width / 2, v + 3, str(v), ha="center", va="bottom", fontsize=7)

    ax = axes[1]
    ax.bar(
        [i - width / 2 for i in x], jacobi_ms, width, color=COLOR_TORCH, label="Jacobi"
    )
    ax.bar(
        [i + width / 2 for i in x],
        amg_total_ms,
        width,
        color=COLOR_PETSC,
        label="AMG (solve + Pmat assembly)",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(devices)
    ax.set_ylabel("KSPSolve wall-clock time (ms)")
    ax.set_title("Wall-clock time")
    ax.set_ylim(0, max(amg_total_ms) * 1.3)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.0), fontsize=7)
    for i, v in zip(x, jacobi_ms):
        ax.text(
            i - width / 2,
            v + max(amg_total_ms) * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for i, v in zip(x, amg_total_ms):
        ax.text(
            i + width / 2,
            v + max(amg_total_ms) * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(HERE / "anisotropic_amg_vs_jacobi.pdf")
    plt.close(fig)


def main():
    cpu_log = (LOGS / "petsc_torch_benchmark_cpu.log").read_text()
    cuda_log = (LOGS / "petsc_torch_benchmark_cuda.log").read_text()
    plot_timing_sweep(cpu_log, cuda_log)
    plot_anisotropic(cpu_log, cuda_log)
    print("Wrote timing_sweep.pdf and anisotropic_amg_vs_jacobi.pdf")


if __name__ == "__main__":
    main()
