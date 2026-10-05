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
 * grain) by reading rows of 9 values (row-major 3x3 matrices) from a
 * delimited text file, one row per grain. Produces the same buffer
 * convention as GrainOrientationsFromInput, so downstream objects (e.g.
 * PolycrystalAnisotropicGBEnergy) can consume orientations without caring
 * how they were specified.
 */
class GrainOrientationsFromFile : public TensorOperator<>
{
public:
  static InputParameters validParams();

  GrainOrientationsFromFile(const InputParameters & parameters);

  virtual void computeBuffer() override;

protected:
  /// Path to the delimited file, one row of 9 values per grain.
  const FileName & _orientation_file;
};
