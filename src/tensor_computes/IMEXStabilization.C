/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "IMEXStabilization.h"

registerMooseObject("SwiftApp", IMEXStabilization);

InputParameters
IMEXStabilization::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription("Calculates the linear stabilization term for IMEX time integration.");
  params.addRequiredParam<TensorInputBufferName>(
      "nonlinear_reciprocal",
      "Reciprocal space buffer containing non-linear term in time integration.");
  params.addRequiredParam<TensorInputBufferName>(
      "v_bar",
      "Reciprocal space buffer of variable 'v' whose time integration is being performed.");
  return params;
}
IMEXStabilization::IMEXStabilization(const InputParameters & parameters)
  : TensorOperator<>(parameters),
    _NL(getInputBuffer("nonlinear_reciprocal")),
    _v_bar(getInputBuffer("v_bar")),
    _k2(_domain.getKSquare())
{
}

void
IMEXStabilization::computeBuffer()
{
  auto numerator = _domain.ifft(_NL);
  auto denominator = _domain.ifft(-_k2 * _v_bar);
  auto D_s = numerator / denominator;
  std::cout << "Max = " << D_s.max() << ", Min = " << D_s.min() << std::endl;
//   auto mask = torch::abs(denominator) > 1e-15;

//   auto safe_result = torch::where(mask, _NL / denominator, 0.0);

//   _u = safe_result;
//   auto temp = _domain.ifft(safe_result);



//   auto stabilization_coeff = torch::mean(torch::abs(temp));

//   std::cout << "Stab coeff = " << temp.min() << " , " << temp.max() << std::endl;
//   std::cout << "Stab coeff mean = " << torch::mean(torch::abs(temp)) << std::endl;

  _u = _k2;
}
