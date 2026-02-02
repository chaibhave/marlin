/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"
// moose headers
#include "DataFileUtils.h"
// libtorch headers
#include <torch/script.h>

class LibtorchPhaseFieldModel : public TensorOperator<>
{
public:
  static InputParameters validParams();

  LibtorchPhaseFieldModel(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  const torch::Tensor & _c_hat;
  torch::Tensor * _linear;

  Moose::DataFileUtils::Path _file_path;
  // We need to use a pointer here because forward is not const qualified
  std::unique_ptr<torch::jit::script::Module> _surrogate;
};
