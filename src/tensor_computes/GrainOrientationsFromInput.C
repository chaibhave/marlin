/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainOrientationsFromInput.h"

#include "MarlinUtils.h"

registerMooseObject("MarlinApp", GrainOrientationsFromInput);

InputParameters
GrainOrientationsFromInput::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Builds a [N,3,3] grain-orientation buffer from rotation matrices listed in the input "
      "file, one row of 9 values (row-major 3x3 matrix) per grain.");
  params.addRequiredParam<std::vector<std::vector<Real>>>(
      "rotation_matrices", "One row of 9 values (row-major 3x3 rotation matrix) per grain.");
  return params;
}

GrainOrientationsFromInput::GrainOrientationsFromInput(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _rotation_matrices(getParam<std::vector<std::vector<Real>>>("rotation_matrices"))
{
  for (const auto & row : _rotation_matrices)
    if (row.size() != 9)
      paramError("rotation_matrices", "Each row must list exactly 9 values (a 3x3 matrix).");
}

void
GrainOrientationsFromInput::computeBuffer()
{
  const auto n_grains = _rotation_matrices.size();

  std::vector<double> flat;
  flat.reserve(n_grains * 9);
  for (const auto & row : _rotation_matrices)
    for (const auto & value : row)
      flat.push_back(value);

  _u = torch::from_blob(flat.data(), {static_cast<int64_t>(n_grains), 3, 3}, torch::kFloat64)
           .to(MooseTensor::floatTensorOptions())
           .clone();
}
