/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#pragma once

#include "TensorOperator.h"

/**
 * Builds a [N,3,3] orientation buffer (lab-frame rotation matrices, one per
 * grain) from rotation matrices listed directly in the input file. Produces
 * the same buffer convention as GrainOrientationsFromFile, so downstream
 * objects (e.g. PolycrystalAnisotropicGBEnergy) can consume orientations
 * without caring how they were specified.
 */
class GrainOrientationsFromInput : public TensorOperator<>
{
public:
  static InputParameters validParams();

  GrainOrientationsFromInput(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  /// One row of 9 values (row-major 3x3 rotation matrix) per grain.
  const std::vector<std::vector<Real>> & _rotation_matrices;
};
