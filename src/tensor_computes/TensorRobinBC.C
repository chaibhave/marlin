/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "TensorRobinBC.h"
#include "DomainAction.h"
#include "Function.h"
#include "MarlinUtils.h"

registerMooseObject("MarlinApp", TensorRobinBC);

InputParameters
TensorRobinBC::validParams()
{
  InputParameters params = TensorBoundaryCondition::validParams();
  params.addParam<Real>("a", 1, "Robin coefficient on u (a*u + b*du/dn = c).");
  params.addParam<Real>("b", 0, "Robin coefficient on du/dn (a*u + b*du/dn = c).");
  params.addParam<Real>("c", 0, "Robin right-hand-side value (a*u + b*du/dn = c).");
  params.addParam<FunctionName>(
      "c_function",
      "Function giving c along the face, evaluated at the face position at the solve time.");
  params.addParam<TensorInputBufferName>(
      "c_buffer",
      "Buffer whose first owned cell layer next to the face holds c along the face. The compute "
      "that writes it must evaluate c at the face position, not at the cell centres.");
  params.addClassDescription(
      "General Robin boundary condition a*u + b*du/dn = c on a non-periodic domain face.");
  return params;
}

TensorRobinBC::TensorRobinBC(const InputParameters & parameters)
  : TensorBoundaryCondition(parameters),
    FunctionInterface(this),
    _a(getParam<Real>("a")),
    _b(getParam<Real>("b")),
    _c_scalar(torch::tensor(getParam<Real>("c"), MooseTensor::floatTensorOptions())),
    _c_function(isParamValid("c_function") ? &getFunction("c_function") : nullptr),
    _c_buffer(isParamValid("c_buffer") ? &getInputBuffer("c_buffer") : nullptr),
    _c_function_time(std::numeric_limits<Real>::quiet_NaN())
{
  if (unsigned(isParamSetByUser("c")) + bool(_c_function) + bool(_c_buffer) > 1)
    paramError("c",
               "Give only one of ",
               parameters.checkForRename("c"),
               ", ",
               parameters.checkForRename("c_function"),
               " and ",
               parameters.checkForRename("c_buffer"),
               ".");

  // the ghost value divides by a/2 + b/dx (see ghostCoefficientOnU0); this also rejects a = b = 0
  const auto dx = _domain.getGridSpacing()(faceDim());
  if (std::abs(_a / 2.0 + _b / dx) <= 1e-12 * (std::abs(_a) / 2.0 + std::abs(_b) / dx))
    paramError(
        "a", "a/2 + b/dx must be nonzero (dx = ", dx, " is the grid spacing normal to the face).");
}

Real
TensorRobinBC::ghostCoefficientOnU0(Real a, Real b, Real dx)
{
  // ughost = [c - u0*(a/2 - b/dx)] / (a/2 + b/dx)  =>  coefficient on u0
  return -(a / 2.0 - b / dx) / (a / 2.0 + b / dx);
}

Real
TensorRobinBC::ghostConstantTerm(Real a, Real b, Real c, Real dx)
{
  return c / (a / 2.0 + b / dx);
}

void
TensorRobinBC::fillGhostCellsStatic(torch::Tensor & buffer,
                                    bool is_upper_face,
                                    int64_t face_dim,
                                    unsigned int ghost_layers,
                                    int64_t halo,
                                    int64_t owned_extent,
                                    Real a,
                                    Real b,
                                    const torch::Tensor & c,
                                    Real dx)
{
  if (ghost_layers > 2)
    ::mooseError(
        "TensorRobinBC fills at most two ghost layers, but ", ghost_layers, " were requested.");
  // the second ghost layer divides by a/2 + b/(3dx) (the constructor checks a/2 + b/dx)
  if (ghost_layers == 2 &&
      std::abs(a / 2.0 + b / (3.0 * dx)) <= 1e-12 * (std::abs(a) / 2.0 + std::abs(b) / (3.0 * dx)))
    ::mooseError("TensorRobinBC with two ghost layers requires a/2 + b/(3*dx) to be nonzero (dx = ",
                 dx,
                 " is the grid spacing normal to the face).");

  // Marlin's real-space grid is cell-centered (axis built from
  // min+dx/2 ... max-dx/2), so for either face the boundary sits at the
  // midpoint between the first interior cell (u0) and the first ghost cell
  // (ughost), and the outward normal derivative at that midpoint is well
  // approximated by a central difference between u0 and ughost. Both the
  // lower and upper face reduce to the SAME formula (ghostCoefficientOnU0 /
  // ghostConstantTerm) because "ughost" always means "the cell on the
  // outward side of u0" for whichever face is being filled.
  const auto u0 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent - 1, 1)
                                : buffer.narrow(face_dim, halo, 1);

  const auto coeff = ghostCoefficientOnU0(a, b, dx);
  // c / (a/2 + b/dx), see ghostConstantTerm
  const auto rhs = c * ghostConstantTerm(a, b, 1.0, dx);

  // first ghost row, directly from the Robin relation at the face
  auto ghost1 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent, 1)
                              : buffer.narrow(face_dim, halo - 1, 1);
  ghost1.copy_(coeff * u0 + rhs);

  if (ghost_layers >= 2)
  {
    // second ghost row: the same Robin relation on the pair (u1, ghost2), which is also
    // centred on the face but 3*dx apart. Exact for linear fields; the even mirror
    // ghost2 = u1 for zero flux and the odd mirror ghost2 = 2c - u1 for Dirichlet.
    const auto u1 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent - 2, 1)
                                  : buffer.narrow(face_dim, halo + 1, 1);
    auto ghost2 = is_upper_face ? buffer.narrow(face_dim, halo + owned_extent + 1, 1)
                                : buffer.narrow(face_dim, halo - 2, 1);
    ghost2.copy_(ghostCoefficientOnU0(a, b, 3.0 * dx) * u1 +
                 c * ghostConstantTerm(a, b, 1.0, 3.0 * dx));
  }
}

void
TensorRobinBC::fillGhostCells(torch::Tensor & buffer,
                              unsigned int ghost_layers,
                              int64_t halo,
                              int64_t owned_extent,
                              Real dx) const
{
  fillGhostCellsStatic(buffer,
                       isUpperFace(),
                       static_cast<int64_t>(faceDim()),
                       ghost_layers,
                       halo,
                       owned_extent,
                       _a,
                       _b,
                       faceValue(),
                       dx);
}

void
TensorRobinBC::prepareFaceData()
{
  if (_c_buffer)
    _tensor_problem.exchangeGhostLayers(
        getParam<TensorInputBufferName>("c_buffer"), _tensor_problem.getMaxGhostLayer(), faceDim());
}

torch::Tensor
TensorRobinBC::faceValue() const
{
  if (_c_buffer)
  {
    // the first owned layer next to the face, over the whole padded extent of the other axes
    // (its halo is filled by prepareFaceData)
    const auto halo = int64_t(_tensor_problem.getMaxGhostLayer());
    const auto d = faceDim();
    return _c_buffer->narrow(d, isUpperFace() ? halo + _domain.getLocalGridSize()[d] - 1 : halo, 1);
  }
  if (_c_function)
  {
    const auto t = _tensor_problem.boundaryTime();
    if (t != _c_function_time)
    {
      _c_function_face = functionFaceValue(t);
      _c_function_time = t;
    }
    return _c_function_face;
  }
  return _c_scalar;
}

torch::Tensor
TensorRobinBC::functionFaceValue(Real t) const
{
  // cell centres of the padded local grid along every axis; the face axis is set to the face
  // position. Ghost positions along the other axes are true cell positions, so later axes'
  // fills see the right c at the corners.
  std::array<int64_t, 3> begin{{0, 0, 0}}, end{{0, 0, 0}};
  _domain.getLocalBounds(_domain.comm().rank(), begin, end);
  const auto & dx = _domain.getGridSpacing();
  const auto & x0 = _domain.getDomainMin();
  const auto halo = int64_t(_tensor_problem.getMaxGhostLayer());
  const auto shape = _tensor_problem.getLocalTensorShape({});
  const auto d = faceDim();

  std::vector<int64_t> face_shape(shape.begin(), shape.end());
  face_shape[d] = 1;
  const auto position = [&](unsigned int axis, int64_t i)
  {
    if (axis == d)
      return isUpperFace() ? _domain.getDomainMax()(d) : x0(d);
    return x0(axis) + (begin[axis] + i - halo + 0.5) * dx(axis);
  };

  // row-major (last axis fastest), as the tensor below
  face_shape.resize(3, 1);
  std::vector<double> values;
  values.reserve(face_shape[0] * face_shape[1] * face_shape[2]);
  for (const auto i : make_range(face_shape[0]))
    for (const auto j : make_range(face_shape[1]))
      for (const auto k : make_range(face_shape[2]))
        values.push_back(_c_function->value(t,
                                            Point(position(0, i),
                                                  _dim > 1 ? position(1, j) : 0.0,
                                                  _dim > 2 ? position(2, k) : 0.0)));
  face_shape.resize(_dim);
  return torch::tensor(values, torch::TensorOptions().dtype(torch::kDouble))
      .reshape(face_shape)
      .to(MooseTensor::floatTensorOptions());
}
