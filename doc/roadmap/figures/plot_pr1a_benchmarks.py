#!/usr/bin/env python3
"""Generate PR1a benchmark figures directly from the gtest log files.

Reads doc/roadmap/logs/petsc_torch_benchmark_{cpu,cuda}.log and produces:
  - timing_sweep.pdf: torch-native vs. PETSc zero-copy matrix-free solve
    wall-clock time vs. problem size n (CPU; CUDA absolute and per-unknown).
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
COLOR_COPY = "#029E73"

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


def parse_timing_sweep(log_text, petsc_label):
    """Parse the DISABLED_TimingSweep block (CPU log).

    petsc_label is the literal text between "PETSc" and the colon, e.g.
    " (zero-copy Vec)".
    """
    pattern = re.compile(
        r"n=(\d+) torch-native: ([\d.]+)ms, \d+ iters \| PETSc"
        + re.escape(petsc_label)
        + r": ([\d.]+)ms, \d+ iters"
    )
    rows = [(int(m[1]), float(m[2]), float(m[3])) for m in pattern.finditer(log_text)]
    ns, torch_ms, petsc_ms = (list(c) for c in zip(*rows))
    return ns, torch_ms, petsc_ms


def parse_cuda_sweep(log_text):
    """Parse DISABLED_CudaTimingSweep: torch-native, torch-native with a
    per-matvec device-to-device copy, and PETSc CUDA (setup, solve)."""
    pattern = re.compile(
        r"n=(\d+) torch-native: ([\d.]+)ms, \d+ iters \| "
        r"torch-native \+ per-matvec D2D copy: ([\d.]+)ms, \d+ iters \| "
        r"PETSc CUDA \(zero-copy Vec\): setup ([\d.]+)ms, solve ([\d.]+)ms"
    )
    rows = [
        (int(m[1]),) + tuple(float(g) for g in m.groups()[1:])
        for m in pattern.finditer(log_text)
    ]
    ns, torch_ms, copy_ms, setup_ms, petsc_ms = (list(c) for c in zip(*rows))
    return ns, torch_ms, copy_ms, setup_ms, petsc_ms


def parse_anisotropic(log_text, test_name):
    """Parse the AnisotropicAMGConvergesFasterThanJacobi(-Cuda) line."""
    pattern = re.compile(
        r"\[ RUN      \] PetscTorchMatrixFreeBenchmarkTest\."
        + re.escape(test_name)
        + r"\nanisotropic.*?: "
        r"Jacobi (\d+) its, setup ([\d.]+)ms, solve ([\d.]+)ms, repeat solve ([\d.]+)ms \| "
        r"AMG \(PCGAMG\) (\d+) its, Pmat assembly ([\d.]+)ms, setup ([\d.]+)ms, "
        r"solve ([\d.]+)ms, repeat solve ([\d.]+)ms"
    )
    m = pattern.search(log_text)
    if not m:
        raise ValueError(f"Could not find {test_name} result in log")
    keys = [
        "jacobi_its",
        "jacobi_setup_ms",
        "jacobi_solve_ms",
        "jacobi_repeat_ms",
        "amg_its",
        "amg_assembly_ms",
        "amg_setup_ms",
        "amg_solve_ms",
        "amg_repeat_ms",
    ]
    return {
        k: (int(v) if k.endswith("its") else float(v)) for k, v in zip(keys, m.groups())
    }


def plot_timing_sweep(cpu_log, cuda_log):
    n_cpu, torch_cpu, petsc_cpu = parse_timing_sweep(cpu_log, " (zero-copy Vec)")
    n_cu, torch_cu, copy_cu, _, petsc_cu = parse_cuda_sweep(cuda_log)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    ax = axes[0]
    ax.loglog(n_cpu, torch_cpu, "o-", color=COLOR_TORCH, label="torch-native")
    ax.loglog(n_cpu, petsc_cpu, "s-", color=COLOR_PETSC, label="PETSc (zero-copy)")
    ax.set_xlabel(r"$n$ (grid points per side)")
    ax.set_ylabel("CG solve wall-clock time (ms)")
    ax.set_title("CPU")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    # CUDA: absolute time spans five decades (4 ms to 1.8 s), so it is shown
    # log-log; the right panel divides by unknowns and iterations to expose
    # the launch-latency floor and the bandwidth-bound plateau.
    ax = axes[1]
    ax.loglog(n_cu, torch_cu, "o-", color=COLOR_TORCH, label="torch-native")
    ax.loglog(n_cu, copy_cu, "^--", color=COLOR_COPY, label="torch-native + D2D copy")
    ax.loglog(n_cu, petsc_cu, "s-", color=COLOR_PETSC, label="PETSc (zero-copy)")
    ax.set_xlabel(r"$n$ (grid points per side)")
    ax.set_ylabel("CG solve wall-clock time (ms)")
    ax.set_title("CUDA (H200)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    iters = 32
    per_elem = lambda ms, ns: [
        t * 1e-3 / iters / (n * n) * 1e12 for t, n in zip(ms, ns)
    ]
    ax = axes[2]
    ax.loglog(n_cu, per_elem(torch_cu, n_cu), "o-", color=COLOR_TORCH)
    ax.loglog(n_cu, per_elem(copy_cu, n_cu), "^--", color=COLOR_COPY)
    ax.loglog(n_cu, per_elem(petsc_cu, n_cu), "s-", color=COLOR_PETSC)
    ax.set_xlabel(r"$n$ (grid points per side)")
    ax.set_ylabel("time per unknown per iteration (ps)")
    ax.set_title("CUDA, normalized")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "timing_sweep.pdf")
    plt.close(fig)


def plot_anisotropic(cpu_log, cuda_log):
    cpu = parse_anisotropic(cpu_log, "AnisotropicAMGConvergesFasterThanJacobi")
    cuda = parse_anisotropic(cuda_log, "AnisotropicCudaAMGConvergesFasterThanJacobi")
    devices = ["CPU", "CUDA"]
    runs = [cpu, cuda]
    x = range(len(devices))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    jac = [r["jacobi_its"] for r in runs]
    amg = [r["amg_its"] for r in runs]
    ax.bar([i - width / 2 for i in x], jac, width, color=COLOR_TORCH, label="Jacobi")
    ax.bar(
        [i + width / 2 for i in x], amg, width, color=COLOR_PETSC, label="AMG (PCGAMG)"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(devices)
    ax.set_ylabel("CG iterations")
    ax.set_title("Iteration count")
    ax.set_ylim(0, max(jac) * 1.3)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    for i, v in zip(x, jac):
        ax.text(i - width / 2, v + 3, str(v), ha="center", va="bottom", fontsize=7)
    for i, v in zip(x, amg):
        ax.text(i + width / 2, v + 3, str(v), ha="center", va="bottom", fontsize=7)

    # Stacked: one-time costs (Pmat assembly, PC setup) below the per-solve
    # cost, which is the repeat solve with the PC already built.
    ax = axes[1]
    jac_setup = [r["jacobi_setup_ms"] for r in runs]
    jac_solve = [r["jacobi_repeat_ms"] for r in runs]
    amg_asm = [r["amg_assembly_ms"] for r in runs]
    amg_setup = [r["amg_setup_ms"] for r in runs]
    amg_solve = [r["amg_repeat_ms"] for r in runs]
    xj = [i - width / 2 for i in x]
    xa = [i + width / 2 for i in x]
    ax.bar(xj, jac_solve, width, color=COLOR_TORCH, label="Jacobi: solve")
    ax.bar(
        xj,
        jac_setup,
        width,
        bottom=jac_solve,
        color=COLOR_TORCH,
        alpha=0.45,
        label="Jacobi: PC setup",
    )
    ax.bar(xa, amg_solve, width, color=COLOR_PETSC, label="AMG: solve")
    bottom = amg_solve
    ax.bar(
        xa,
        amg_setup,
        width,
        bottom=bottom,
        color=COLOR_PETSC,
        alpha=0.6,
        label="AMG: PC setup",
    )
    bottom = [a + b for a, b in zip(bottom, amg_setup)]
    ax.bar(
        xa,
        amg_asm,
        width,
        bottom=bottom,
        color=COLOR_PETSC,
        alpha=0.3,
        label=r"AMG: $P_{mat}$ assembly",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(devices)
    ax.set_ylabel("wall-clock time (ms)")
    ax.set_title("Solve (PC reused) + one-time costs")
    top = max(s + u + a for s, u, a in zip(amg_solve, amg_setup, amg_asm))
    ax.set_ylim(0, top * 1.55)
    ax.legend(loc="upper left", fontsize=6.5, ncol=1)
    for i, v in zip(x, jac_solve):
        ax.text(
            i - width / 2,
            v + jac_setup[i] + top * 0.02,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for i, v in zip(x, amg_solve):
        total = v + amg_setup[i] + amg_asm[i]
        ax.text(
            i + width / 2,
            total + top * 0.02,
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
