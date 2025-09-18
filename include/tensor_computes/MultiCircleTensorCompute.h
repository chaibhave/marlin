/**********************************************************************/
/*                    DO NOT MODIFY THIS HEADER                       */
/*             Swift, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Tensor compute that smoothly interpolates inside/outside values for multiple circles/spheres.
 */
class MultiCircleTensorCompute : public TensorOperator<>
{
public:
  static InputParameters validParams();

  MultiCircleTensorCompute(const InputParameters & parameters);

  virtual void computeBuffer() override;

private:
  enum class ProfileType
  {
    Cosine,
    Tanh,
    Sharp
  };

  /// Centers of the circles (num_circles x dim)
  torch::Tensor _centers;

  /// Radii of each circle (num_circles)
  torch::Tensor _radii;

  /// Inside value per circle (num_circles)
  torch::Tensor _inside_values;

  /// Outside value per circle (num_circles)
  torch::Tensor _outside_values;

  /// Selected smoothing profile
  ProfileType _profile;

  /// Interface thickness for smooth profiles
  const Real _interface_thickness;

  /// Optional background value toggle
  bool _use_background;

  /// Background value applied when enabled
  Real _background_value;

  /// Number of circles encoded by the input parameters
  std::size_t _num_circles;
};
