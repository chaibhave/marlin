/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "MultiCircleTensorCompute.h"

#include "MooseEnum.h"
#include "MooseError.h"
#include "SwiftUtils.h"

#include "libmesh/libmesh_common.h"

#include <algorithm>
#include <vector>

registerMooseObject("SwiftApp", MultiCircleTensorCompute);

InputParameters
MultiCircleTensorCompute::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Smoothly interpolate between inside and outside values for multiple circles/spheres.");
  params.addRequiredParam<std::vector<Real>>("centers",
                                             "Flattened list of circle/sphere centers.");
  params.addRequiredParam<std::vector<Real>>("radii", "Radius for each circle/sphere.");
  params.addRequiredParam<std::vector<Real>>("inside_values",
                                             "Value to apply inside each circle/sphere.");
  params.addRequiredParam<std::vector<Real>>("outside_values",
                                             "Value to apply outside each circle/sphere.");
  params.addParam<Real>("interface_thickness",
                        0.0,
                        "Width of the smooth blending region for cosine/tanh profiles.");
  params.addParam<MooseEnum>("profile",
                             MooseEnum("cos tanh sharp", "cos"),
                             "Transition profile to use (cos, tanh, or sharp).");
  params.addParam<Real>("background_value",
                        "Optional background value used away from all circles.");
  return params;
}

MultiCircleTensorCompute::MultiCircleTensorCompute(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _profile(ProfileType::Cosine),
    _interface_thickness(getParam<Real>("interface_thickness")),
    _use_background(isParamValid("background_value")),
    _background_value(_use_background ? getParam<Real>("background_value") : 0.0),
    _num_circles(0)
{
  if (_dim < 2)
    paramError("centers", "MultiCircleTensorCompute requires at least a 2D domain.");

  const auto centers_flat = getParam<std::vector<Real>>("centers");
  const auto radii_vec = getParam<std::vector<Real>>("radii");

  if (centers_flat.empty())
    paramError("centers", "At least one circle/sphere center must be provided.");

  if (centers_flat.size() % _dim != 0)
    paramError("centers", "Center coordinates must be provided in multiples of the dimension (",
               _dim,
               ").");

  _num_circles = centers_flat.size() / _dim;

  if (radii_vec.size() != _num_circles)
    paramError("radii", "Provide exactly ", _num_circles, " radii.");

  if (std::any_of(radii_vec.begin(), radii_vec.end(), [](Real r) { return r <= 0.0; }))
    paramError("radii", "Radii must be strictly positive.");

  const auto inside_vec = getParam<std::vector<Real>>("inside_values");
  if (!(inside_vec.size() == 1 || inside_vec.size() == _num_circles))
    paramError("inside_values", "Provide either one inside value or one per circle.");

  const auto outside_vec = getParam<std::vector<Real>>("outside_values");
  if (!(outside_vec.size() == 1 || outside_vec.size() == _num_circles))
    paramError("outside_values", "Provide either one outside value or one per circle.");

  std::vector<Real> inside_full(_num_circles);
  std::vector<Real> outside_full(_num_circles);

  for (std::size_t i = 0; i < _num_circles; ++i)
  {
    inside_full[i] = inside_vec.size() == 1 ? inside_vec[0] : inside_vec[i];
    outside_full[i] = outside_vec.size() == 1 ? outside_vec[0] : outside_vec[i];
  }

  const auto & profile_param = getParam<MooseEnum>("profile");
  if (profile_param == "cos")
    _profile = ProfileType::Cosine;
  else if (profile_param == "tanh")
    _profile = ProfileType::Tanh;
  else if (profile_param == "sharp")
    _profile = ProfileType::Sharp;
  else
    paramError("profile", "Unsupported profile selection.");

  if (_profile != ProfileType::Sharp && _interface_thickness <= 0.0)
    paramError("interface_thickness", "Interface thickness must be positive for smooth profiles.");

  if (!_use_background && !outside_full.empty())
    _background_value = outside_full.front();

  _centers = torch::tensor(centers_flat, MooseTensor::floatTensorOptions())
                 .view({static_cast<int64_t>(_num_circles), static_cast<int64_t>(_dim)});
  _radii = torch::tensor(radii_vec, MooseTensor::floatTensorOptions());
  _inside_values = torch::tensor(inside_full, MooseTensor::floatTensorOptions());
  _outside_values = torch::tensor(outside_full, MooseTensor::floatTensorOptions());
}

void
MultiCircleTensorCompute::computeBuffer()
{
  const auto & x_grid = _domain.getXGrid();
  const auto coord_tensor = (_dim == 1) ? x_grid.unsqueeze(-1) : x_grid;

  const auto options = coord_tensor.options();
  auto centers = _centers.to(options);
  auto radii = _radii.to(options);
  auto inside = _inside_values.to(options);
  auto outside = _outside_values.to(options);

  auto x_flat = coord_tensor.reshape({-1, static_cast<int64_t>(_dim)});
  auto diff = x_flat.unsqueeze(1) - centers.unsqueeze(0);
  auto distances = diff.pow(2).sum(-1).sqrt() - radii.view({1, -1});

  torch::Tensor smooth;
  switch (_profile)
  {
    case ProfileType::Cosine:
    {
      auto xi = torch::clamp((distances / _interface_thickness + 1.0) * 0.5, 0.0, 1.0);
      smooth = 0.5 * (1.0 - torch::cos(xi * libMesh::pi));
      break;
    }
    case ProfileType::Tanh:
    {
      auto scaled = distances / _interface_thickness;
      smooth = torch::clamp(0.5 * (1.0 + torch::tanh(scaled)), 0.0, 1.0);
      break;
    }
    case ProfileType::Sharp:
    {
      smooth = (distances >= 0).to(distances.dtype());
      break;
    }
  }

  auto values = inside.view({1, -1}) * (1.0 - smooth) + outside.view({1, -1}) * smooth;

  auto abs_dist = distances.abs();
  auto min_result = abs_dist.min(1, true);
  auto min_indices = std::get<1>(min_result);
  auto field = values.gather(1, min_indices).squeeze(-1);

  if (_use_background)
  {
    auto min_abs = std::get<0>(min_result).squeeze(-1);
    auto background = torch::full_like(field, _background_value);
    if (_profile == ProfileType::Sharp)
      field = torch::where(min_abs > 0.0, background, field);
    else
      field = torch::where(min_abs > _interface_thickness, background, field);
  }

  _u = field.reshape(_domain.getShape());
}
