from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

import torch

from .mesh import Mesh
from .halo import Face, HaloFiller


_UNSET = object()


@dataclass
class Field:
    """A tensor living on a `Mesh`, stored at ghosted shape.

    BCs are a property of the FIELD, not the mesh -- different variables on
    the same mesh (velocity, pressure, temperature, ...) can have different
    boundary conditions, and many tensors (postprocessing, visualization,
    intermediate diagnostics) don't need BCs or ghost-filling at all. A
    Field's `fillers` is therefore optional and per-instance:

    * Fields that go into stencil/residual evaluation: construct with
      `fillers=...` and call `.with_halos()` when the padded tensor is
      needed.
    * Everything else: leave `fillers=None`. Such a Field still gets
      ghosted storage and `.interior` (useful for e.g. writing IC/output
      data into the right slice), but `.with_halos()` will raise -- there's
      no way to accidentally treat an unassigned field as if it had BCs.

    Always construct via `Field.zeros(...)` rather than the raw dataclass
    constructor with a hand-built tensor -- that's what pins the tensor's
    shape to the ghost/dim conventions of the specific `Mesh` it belongs to.
    """

    mesh: Mesh
    data: torch.Tensor  # ghosted shape: see Mesh.zeros_field for shape rules
    fillers: Optional[dict[Face, HaloFiller]] = None

    def __post_init__(self):
        if self.fillers is None:
            return
        active = set(self.mesh.config.active_faces)
        given = set(self.fillers.keys())
        if given != active:
            raise ValueError(
                f"fillers must cover exactly the mesh's active faces "
                f"{sorted(f.name for f in active)}; got "
                f"{sorted(f.name for f in given)}"
            )
        # Periodic must be paired, checked only over faces that exist at
        # this dim.
        pairs = ((Face.X_LO, Face.X_HI), (Face.Y_LO, Face.Y_HI), (Face.Z_LO, Face.Z_HI))
        for lo, hi in pairs:
            if lo not in active:
                continue
            lo_periodic = type(self.fillers[lo]).__name__ == "Periodic"
            hi_periodic = type(self.fillers[hi]).__name__ == "Periodic"
            if lo_periodic != hi_periodic:
                raise ValueError(f"periodic BC must be set on both {lo} and {hi}")

    # ------------------------------------------------------------ factory

    @classmethod
    def zeros(
        cls, mesh: Mesh, *extra_shape: int, fillers: Optional[dict[Face, HaloFiller]] = None
    ) -> "Field":
        """Allocate a new zero-initialized field on `mesh`.

        extra_shape: optional trailing dims for vector/tensor-valued fields,
        e.g. Field.zeros(mesh) -> scalar, Field.zeros(mesh, 3) -> vector.
        fillers: per-face BCs, required only if `.with_halos()` will be
        called on this field. Leave None for postprocessing/intermediate
        tensors that never need ghost-filling.
        """
        return cls(mesh=mesh, data=mesh.zeros_field(*extra_shape), fillers=fillers)

    @classmethod
    def like(cls, other: "Field", fillers=_UNSET) -> "Field":
        """Zero field with the same mesh/shape as `other`.

        If `fillers` isn't passed at all, inherits `other.fillers`. Pass
        `fillers=None` explicitly to clear BCs, or a real dict to set
        different ones -- `None` as the argument value is distinguishable
        from "not provided" via the `_UNSET` sentinel default.
        """
        resolved = other.fillers if fillers is _UNSET else fillers
        return cls(
            mesh=other.mesh,
            data=torch.zeros_like(other.data),
            fillers=resolved,
        )

    # ----------------------------------------------------------- interior

    @property
    def interior(self) -> torch.Tensor:
        """View into the non-ghost region of this field's data."""
        return self.mesh.interior(self.data)

    @interior.setter
    def interior(self, value: torch.Tensor) -> None:
        self.mesh.interior(self.data)[...] = value

    # -------------------------------------------------------------- halos

    def with_halos(self) -> torch.Tensor:
        """Functionally materialize this field's ghost values.

        Pure: builds a new padded-consistent tensor; never mutates
        `self.data`. Only touches axes that are active for `mesh.config.dim`
        -- e.g. for a 2D mesh, the z axis is left as its existing size-1,
        un-ghosted slice.

        Raises if this field has no fillers assigned: that's the signal
        this Field was never meant to participate in stencil/residual
        evaluation (e.g. it's a postprocessing/visualization tensor).
        """
        if self.fillers is None:
            raise RuntimeError(
                "with_halos() called on a Field with no BCs assigned. "
                "This field was likely created for postprocessing/"
                "visualization rather than solver use -- pass `fillers=` "
                "to Field.zeros(...) if it should support ghost-filling."
            )
        out = self.mesh.interior(self.data)
        cfg = self.mesh.config
        for face in cfg.active_faces:
            if face.side == "hi":
                continue  # handled together with its lo partner below
            lo_face, hi_face = face, Face[f"{face.axis_name.upper()}_HI"]
            ghost = cfg.ghost
            spacing = self.mesh.spacing(face.axis)
            lo = self.fillers[lo_face].ghost_slab(out, lo_face, ghost, spacing)
            hi = self.fillers[hi_face].ghost_slab(out, hi_face, ghost, spacing)
            out = torch.cat((lo, out, hi), dim=face.axis)
        return out

    # ----------------------------------------------------------- metadata

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        return self.data.device