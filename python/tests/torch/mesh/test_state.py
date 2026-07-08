import pytest
import torch

from marlin.torch.mesh import GridConfig, Mesh, Field, State, Face, Dirichlet


def make_state(dim=1, nx=4, ghost=1):
    cfg = GridConfig(dim=dim, nx=nx, ny=1, nz=1, ghost=ghost)
    mesh = Mesh(cfg)
    return State(mesh=mesh)


def dirichlet_1d(lo=0.0, hi=0.0):
    return {Face.X_LO: Dirichlet(lo), Face.X_HI: Dirichlet(hi)}


def test_add_registers_field_and_returns_it():
    state = make_state()
    f = state.add("phi")
    assert isinstance(f, Field)
    assert state["phi"] is f
    assert "phi" in state


def test_add_duplicate_name_raises():
    state = make_state()
    state.add("phi")
    with pytest.raises(ValueError):
        state.add("phi")


def test_heterogeneous_shapes_scalar_and_tensor():
    cfg = GridConfig(dim=3, nx=4, ny=4, nz=4, ghost=1)
    mesh = Mesh(cfg)
    state = State(mesh=mesh)
    scalar = state.add("temperature")
    stress = state.add("stress", 3, 3)
    assert scalar.data.shape == (6, 6, 6)
    assert stress.data.shape == (6, 6, 6, 3, 3)


def test_names_and_iter():
    state = make_state()
    state.add("a")
    state.add("b")
    assert state.names() == ("a", "b")
    assert set(iter(state)) == {"a", "b"}


def test_interior_returns_all_variables():
    state = make_state(nx=4)
    state.add("a")
    state.add("b")
    interior = state.interior()
    assert set(interior.keys()) == {"a", "b"}
    assert interior["a"].shape == (4, 1, 1)


def test_with_halos_only_includes_fields_with_fillers():
    state = make_state(nx=4, ghost=1)
    state.add("has_bc", fillers=dirichlet_1d())
    state.add("no_bc")  # postprocessing-style, no fillers
    padded = state.with_halos()
    assert set(padded.keys()) == {"has_bc"}
    assert padded["has_bc"].shape == (6, 1, 1)


def test_to_positional_and_update_from_positional_roundtrip():
    state = make_state(nx=4, ghost=1)
    state.add("a")
    state.add("b")
    names = ("a", "b")
    tensors = state.to_positional(names)
    assert len(tensors) == 2
    assert tensors[0] is state["a"].data

    new_a = torch.ones_like(tensors[0])
    new_b = torch.full_like(tensors[1], 2.0)
    state.update_from_positional(names, (new_a, new_b))
    assert torch.equal(state["a"].data, new_a)
    assert torch.equal(state["b"].data, new_b)


def test_update_from_positional_length_mismatch_raises():
    state = make_state(nx=4)
    state.add("a")
    with pytest.raises(ValueError):
        state.update_from_positional(("a",), tuple())


def test_stacked_same_shape_ok():
    state = make_state(nx=4, ghost=1)
    state.add("a")
    state.add("b")
    stacked = state.stacked(("a", "b"))
    assert stacked.shape == (2, *state["a"].data.shape)


def test_stacked_mismatched_shapes_raises():
    cfg = GridConfig(dim=3, nx=4, ny=4, nz=4, ghost=1)
    mesh = Mesh(cfg)
    state = State(mesh=mesh)
    state.add("scalar")
    state.add("tensor", 3, 3)
    with pytest.raises(ValueError):
        state.stacked(("scalar", "tensor"))