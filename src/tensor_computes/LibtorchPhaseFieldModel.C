/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "LibtorchPhaseFieldModel.h"
#include "MarlinUtils.h"
#include <torch/torch.h>

registerMooseObject("MarlinApp", LibtorchPhaseFieldModel);

InputParameters
LibtorchPhaseFieldModel::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Evaluates a phase field model using a Libtorch model. "
                             "Takes concentration in Fourier space and k_x as inputs, "
                             "returns dc_hat_dt and its JVP with respect to c_hat.");
  params.addRequiredParam<TensorInputBufferName>("concentration_hat",
                                                 "Concentration tensor in Fourier space");
  params.addRequiredParam<TensorOutputBufferName>(
      "linear", "Buffer name for the Jacobian-Vector Product linear term.");
  params.addRequiredParam<DataFileName>(
      "libtorch_model_file", "Path to the Libtorch file containing the phase field model.");
  return params;
}

LibtorchPhaseFieldModel::LibtorchPhaseFieldModel(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _c_hat(getInputBuffer("concentration_hat")),
    _linear(&getOutputBuffer("linear")),
    _file_path(Moose::DataFileUtils::getPath(getParam<DataFileName>("libtorch_model_file"))),
    _surrogate(std::make_unique<torch::jit::script::Module>(torch::jit::load(_file_path.path)))
{
  const auto opts = MooseTensor::floatTensorOptions();

  auto ref = torch::empty({0}, opts);
  const auto dev = ref.device();
  const auto dt = ref.scalar_type();
  _surrogate->to(dev, dt, /*non_blocking=*/false);
  // Keep in train mode to allow gradient computation for JVP
  _surrogate->train();
}

void
LibtorchPhaseFieldModel::computeBuffer()
{

  //
  // For when we want to use backward autograd
  //
  auto c_hat = _c_hat.detach().requires_grad_(true);

  // Call the model for the main output
  _u = _surrogate->forward({c_hat, _domain.getKGrid(),_domain.getGridSize()}).toTensor();

  auto VJP = torch::autograd::grad({torch::real(_u).sum()},
                                   {c_hat},
                                   /*grad_outputs=*/{},
                                   /*retain_graph=*/false,
                                   /*create_graph=*/true,
                                   /*allow_unused=*/true)[0];
  *_linear = VJP;

  //   auto c_hat = _c_hat.detach().requires_grad_(true);
  //   auto K_grid = _domain.getKGrid().detach().requires_grad_(true);
  //   auto tangent = torch::ones_like(c_hat);
  //   auto level = torch::autograd::forward_ad::enter_dual_level();

  //   auto dual_c_hat = torch::_make_dual(c_hat, tangent, level);

  //   // 4. Forward pass
  //   auto u_dual = _surrogate->forward({dual_c_hat, K_grid}).toTensor();

  //   // 5. Use torch::_unpack_dual and handle the std::tuple return
  //   // _unpack_dual returns std::tuple<at::Tensor, at::Tensor>
  //   auto unpacked = torch::_unpack_dual(torch::real(u_dual), level);

  //   auto jvp = std::get<1>(unpacked);
  //   auto complex_primal = std::get<0>(unpacked);
  //   auto complex_tangent = std::get<1>(unpacked);
  //   // 2. Now extract the real parts for your simulation
  //   _u = complex_primal.detach();

  //   // Ensure the tangent exists before calling real()
  //   if (complex_tangent.defined() && complex_tangent.numel() > 0)
  //   {
  //     *_linear = torch::real(complex_tangent).detach();
  //   }
  //   else
  //   {
  //     // This would indicate the surrogate itself is stripping the dual properties
  //     std::cerr << "Critical: Complex tangent is undefined!" << std::endl;
  //   }
  //   // 7. Exit level
  //   torch::autograd::forward_ad::exit_dual_level(level);
}
