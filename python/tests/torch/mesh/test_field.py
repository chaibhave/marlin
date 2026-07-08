import pytest
import torch

from marlin.torch.mesh import GridConfig, Mesh, Field, Face, Dirichlet, Neumann, Periodic


def make_mesh(dim=1, nx=4, ny=1, nz=1, ghost=1):
    cfg = GridConfig(dim=dim, nx=nx, ny=ny, nz=nz, ghost=ghost)
    return Mesh(cfg)


def full_fillers_1d(lo=None, hi=None):
    return {
        Face.X_LO: lo or Dirichlet(0.0),
        Face.X_HI: hi or Dirichlet(0.0),
    }


# --------------------------------------------------------------------- #
# construction / validation
# --------------------------------------------------------------------- #

def test_zeros_without_fillers_has_none():
    mesh = make_mesh()
    f = Field.zeros(mesh)
    assert f.fillers is None


def test_zeros_shape_matches_mesh_zeros_field():
    mesh = make_mesh(dim=3, nx=4, ny=4, nz=4, ghost=1)
    f = Field.zeros(mesh, 3)
    assert f.data.shape == mesh.zeros_field(3).shape


def test_fillers_must_cover_exactly_active_faces():
    mesh = make_mesh(dim=1, nx=4)
    with pytest.raises(ValueError):
        Field.zeros(mesh, fillers={Face.X_LO: Dirichlet(0.0)})  # missing X_HI

    with pytest.raises(ValueError):
        # extra face beyond dim=1's active faces
        Field.zeros(mesh, fillers={
            Face.X_LO: Dirichlet(0.0),
            Face.X_HI: Dirichlet(0.0),
            Face.Y_LO: Dirichlet(0.0),
        })


def test_fillers_exactly_matching_active_faces_ok():
    mesh = make_mesh(dim=1, nx=4)
    f = Field.zeros(mesh, fillers=full_fillers_1d())
    assert set(f.fillers.keys()) == {Face.X_LO, Face.X_HI}


def test_periodic_must_be_paired():
    mesh = make_mesh(dim=1, nx=4)
    with pytest.raises(ValueError):
        Field.zeros(mesh, fillers={
            Face.X_LO: Periodic(),
            Face.X_HI: Dirichlet(0.0),
        })
    # paired periodic is fine
    f = Field.zeros(mesh, fillers={Face.X_LO: Periodic(), Face.X_HI: Periodic()})
    assert f.fillers is not None


# --------------------------------------------------------------------- #
# interior
# --------------------------------------------------------------------- #

def test_interior_getter_setter_roundtrip():
    mesh = make_mesh(dim=1, nx=4, ghost=1)
    f = Field.zeros(mesh)
    f.interior = torch.arange(4, dtype=mesh.config.dtype).view(4, 1, 1)
    assert torch.equal(f.interior.flatten(), torch.tensor([0.0, 1.0, 2.0, 3.0]))
    # ghosts remain untouched (still zero)
    assert f.data[0, 0, 0].item() == 0.0
    assert f.data[-1, 0, 0].item() == 0.0


# --------------------------------------------------------------------- #
# with_halos
# --------------------------------------------------------------------- #

def test_with_halos_raises_without_fillers():
    mesh = make_mesh()
    f = Field.zeros(mesh)
    with pytest.raises(RuntimeError):
        f.with_halos()


def test_with_halos_dirichlet_1d_shape_and_values():
    mesh = make_mesh(dim=1, nx=4, ghost=1)
    f = Field.zeros(mesh, fillers=full_fillers_1d(lo=Dirichlet(10.0), hi=Dirichlet(20.0)))
    f.interior = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=mesh.config.dtype).view(4, 1, 1)
    padded = f.with_halos()
    assert padded.shape == (6, 1, 1)
    flat = padded.flatten()
    assert torch.allclose(flat[0], torch.tensor(2 * 10.0 - 1.0, dtype=mesh.config.dtype))
    assert torch.allclose(flat[1:5], torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=mesh.config.dtype))
    assert torch.allclose(flat[5], torch.tensor(2 * 20.0 - 4.0, dtype=mesh.config.dtype))


def test_with_halos_is_pure_does_not_mutate_data():
    mesh = make_mesh(dim=1, nx=4, ghost=1)
    f = Field.zeros(mesh, fillers=full_fillers_1d())
    before = f.data.clone()
    f.with_halos()
    assert torch.equal(f.data, before)


def test_with_halos_2d_only_touches_active_axes():
    mesh = make_mesh(dim=2, nx=4, ny=4, ghost=1)
    fillers = {
        Face.X_LO: Periodic(), Face.X_HI: Periodic(),
        Face.Y_LO: Periodic(), Face.Y_HI: Periodic(),
    }
    f = Field.zeros(mesh, fillers=fillers)
    padded = f.with_halos()
    # z axis untouched (size 1, no ghost padding) since dim=2
    assert padded.shape == (6, 6, 1)


# --------------------------------------------------------------------- #
# Field.like
# --------------------------------------------------------------------- #

def test_like_inherits_mesh_shape_and_fillers_by_default():
    mesh = make_mesh(dim=1, nx=4, ghost=1)
    f = Field.zeros(mesh, fillers=full_fillers_1d())
    g = Field.like(f)
    assert g.mesh is f.mesh
    assert g.data.shape == f.data.shape
    assert g.fillers == f.fillers
    assert g.data is not f.data  # independent storage


def test_like_can_override_fillers():
    mesh = make_mesh(dim=1, nx=4, ghost=1)
    f = Field.zeros(mesh, fillers=full_fillers_1d())
    g = Field.like(f, fillers=None)
    assert g.fillers is None