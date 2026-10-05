#!/usr/bin/env python3
# *                    DO NOT MODIFY THIS HEADER
# *            Marlin, a Fourier spectral solver for MOOSE
# *
# *            Copyright 2024 Battelle Energy Alliance, LLC
# *                        ALL RIGHTS RESERVED
# *
# *        Licensed under LGPL 2.1, please see LICENSE for details
# *             https://www.gnu.org/licenses/lgpl-2.1.html

"""
Export GB5DOF as a traced TorchScript module.

torch.jit.script(GB5DOF(...)) fails: GB5DOF.mat2quat does
`q.reshape(*original_shape, 4)`, and TorchScript cannot statically infer the
size of a list built from unpacking a dynamic-length tuple inside a call.
GB5DOF.py is not modified to work around this (its construction is managed
elsewhere), so this instead traces the model, which TorchScript lowers
correctly: GB5DOF.forward branches with torch.where rather than data-dependent
Python `if`, so a traced graph reproduces the eager model on any input shape,
not just the one traced on. Verified by comparing traced vs. eager output on
randomly sampled orientations distinct from the tracing batch (max abs
difference 0.0 in float64).

Run from this directory so the GB5DOF import resolves:

    python export_gb5dof.py [system]
"""

import sys

import torch

from GB5DOF import GB5DOF

OUTPUT_DIR = "../../data/anisotropic_gb_energy"

# Batch size used only to record the traced graph; the exported module
# generalizes to any batch size (verified separately, see module docstring).
TRACE_BATCH_SIZE = 16


def random_orientation_batch(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Random proper-orthogonal matrices via QR decomposition of Gaussian noise."""
    a = torch.randn(n, 3, 3, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # Fix determinant sign so every matrix is a proper rotation.
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1).prod(dim=-1))
    q = q * sign[:, None, None]
    return q


def export(system: str) -> None:
    dtype = torch.float64

    model = GB5DOF(system=system, device="cpu", dtype=dtype)
    model.eval()

    P = random_orientation_batch(TRACE_BATCH_SIZE, dtype)
    Q = random_orientation_batch(TRACE_BATCH_SIZE, dtype)

    traced = torch.jit.trace(model, (P, Q), check_trace=False)

    # Sanity check against the eager model on inputs distinct from tracing.
    P_check = random_orientation_batch(257, dtype)
    Q_check = random_orientation_batch(257, dtype)
    with torch.no_grad():
        eager_out = model(P_check, Q_check)
        traced_out = traced(P_check, Q_check)
    max_diff = (eager_out - traced_out).abs().max().item()
    print(f"Max |traced - eager| on held-out batch: {max_diff:.3e}")
    if max_diff > 1e-10:
        raise RuntimeError(
            "Traced GB5DOF module disagrees with the eager model; "
            "refusing to save a TorchScript module that may not generalize."
        )

    output_path = f"{OUTPUT_DIR}/{system}_GB5DOF.pt"
    torch.jit.save(traced, output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    export(sys.argv[1] if len(sys.argv) > 1 else "Ni")
