import pytest
import torch

from marlin.torch.mesh import GridConfig, Mesh, Face


# --------------------------------------------------------------------- #
# GridConfig
# --------------------------------------------------------------------- #

def test_dim_must_be_1_2_or_3():
    with pytest.raises(ValueError):
        GridConfig(dim=0, nx=8, ny=8, nz=8)
    with pytest.raises(ValueError):
        GridConfig(dim=4, nx=8, ny=8, nz=8)


@pytest.mark.parametrize("dim,nx,ny,nz", [
    (1, 8, 1, 1),   # ny, nz inactive for dim=1 -- exempt even though < 2*ghost
    (2, 8, 8, 1),   # nz inactive for dim=2 -- exempt
])
def test_inactive_axes_exempt_from_ghost_check(dim, nx, ny, nz):
    GridConfig(dim=dim, nx=nx, ny=ny, nz=nz)  # must not raise


@pytest.mark.parametrize("dim,nx,ny,nz", [
    (1, 1, 8, 8),
    (2, 1, 8, 8),
    (2, 8, 1, 8),
    (3, 1, 8, 8),
    (3, 8, 1, 8),
    (3, 8, 8, 1),
])
def test_active_axis_below_ghost_raises(dim, nx, ny, nz):
    with pytest.raises(ValueError):
        GridConfig(dim=dim, nx=nx, ny=ny, nz=nz)


def test_active_faces_matches_dim():
    cfg1 = GridConfig(dim=1, nx=8, ny=1, nz=1)
    assert set(cfg1.active_faces) == {Face.X_LO, Face.X_HI}

    cfg2 = GridConfig(dim=2, nx=8, ny=8, nz=1)
    assert set(cfg2.active_faces) == {Face.X_LO, Face.X_HI, Face.Y_LO, Face.Y_HI}

    cfg3 = GridConfig(dim=3, nx=8, ny=8, nz=8)
    assert set(cfg3.active_faces) == set(Face)


def test_global_shape_defaults_to_local():
    cfg = GridConfig(dim=3, nx=4, ny=5, nz=6)
    assert cfg.global_shape == (4, 5, 6)


def test_global_shape_explicit_zero_is_not_treated_as_unset():
    # regression test: `self.global_nx or self.nx` would incorrectly fall
    # back to nx here; `is not None` is required.
    cfg = GridConfig(dim=3, nx=4, ny=5, nz=6, global_nx=0)
    assert cfg.global_shape[0] == 0


# --------------------------------------------------------------------- #
# Mesh coordinates
# --------------------------------------------------------------------- #

def test_coords_are_cell_centered_3d():
    cfg = GridConfig(dim=3, nx=4, ny=4, nz=4, xmin=0.0, xmax=1.0)
    mesh = Mesh(cfg)
    dx = 0.25
    expected = torch.tensor([0.125, 0.375, 0.625, 0.875], dtype=cfg.dtype)
    assert torch.allclose(mesh.X.flatten(), expected)
    assert torch.allclose(mesh.dx, torch.tensor(dx, dtype=cfg.dtype))


def test_coords_shapes_by_dim():
    cfg3 = GridConfig(dim=3, nx=4, ny=5, nz=6)
    mesh3 = Mesh(cfg3)
    assert mesh3.X.shape == (4, 1, 1)
    assert mesh3.Y.shape == (1, 5, 1)
    assert mesh3.Z.shape == (1, 1, 6)

    cfg2 = GridConfig(dim=2, nx=4, ny=5, nz=1)
    mesh2 = Mesh(cfg2)
    assert mesh2.X.shape == (4, 1, 1)
    assert mesh2.Y.shape == (1, 5, 1)
    assert mesh2.Z.shape == (1, 1, 1)
    assert torch.equal(mesh2.Z, torch.zeros(1, 1, 1, dtype=cfg2.dtype))

    cfg1 = GridConfig(dim=1, nx=4, ny=1, nz=1)
    mesh1 = Mesh(cfg1)
    assert mesh1.X.shape == (4, 1, 1)
    assert mesh1.Y.shape == (1, 1, 1)
    assert mesh1.Z.shape == (1, 1, 1)


def test_default_device_is_cpu():
    cfg = GridConfig(dim=1, nx=4, ny=1, nz=1)
    assert cfg.device == "cpu"
    mesh = Mesh(cfg)
    assert mesh.X.device.type == "cpu"


# --------------------------------------------------------------------- #
# zeros_field / interior
# --------------------------------------------------------------------- #

def test_zeros_field_scalar_shape_3d():
    cfg = GridConfig(dim=3, nx=4, ny=5, nz=6, ghost=1)
    mesh = Mesh(cfg)
    u = mesh.zeros_field()
    assert u.shape == (6, 7, 8)  # nx+2g, ny+2g, nz+2g


def test_zeros_field_vector_and_tensor_trailing_dims():
    cfg = GridConfig(dim=3, nx=4, ny=4, nz=4, ghost=1)
    mesh = Mesh(cfg)
    vec = mesh.zeros_field(3)
    assert vec.shape == (6, 6, 6, 3)
    ten = mesh.zeros_field(3, 3)
    assert ten.shape == (6, 6, 6, 3, 3)


def test_zeros_field_unused_axes_have_no_ghost_padding():
    cfg = GridConfig(dim=2, nx=4, ny=5, nz=1, ghost=1)
    mesh = Mesh(cfg)
    u = mesh.zeros_field()
    assert u.shape == (6, 7, 1)  # z axis: no +2*ghost since dim=2


def test_interior_shape_roundtrips_to_nx_ny_nz():
    cfg = GridConfig(dim=2, nx=4, ny=5, nz=1, ghost=2)
    mesh = Mesh(cfg)
    u = mesh.zeros_field()
    interior = mesh.interior(u)
    assert interior.shape == (4, 5, 1)


def test_interior_works_with_trailing_component_dims():
    cfg = GridConfig(dim=3, nx=4, ny=4, nz=4, ghost=1)
    mesh = Mesh(cfg)
    vec = mesh.zeros_field(3)
    interior = mesh.interior(vec)
    assert interior.shape == (4, 4, 4, 3)

def test_ghost_below_one_raises():
    with pytest.raises(ValueError):
        GridConfig(dim=1, nx=8, ny=1, nz=1, ghost=0)