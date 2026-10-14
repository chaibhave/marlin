/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainOrientationsFromFile.h"

#include "DelimitedFileReader.h"

registerMooseObject("MarlinApp", GrainOrientationsFromFile);

InputParameters
GrainOrientationsFromFile::validParams()
{
  InputParameters params = TensorOperator<>::validParams();
  params.addClassDescription(
      "Builds a [N,3,3] grain-orientation buffer by reading rotation matrices from a delimited "
      "file, one row of 9 values (row-major 3x3 matrix) per grain.");
  params.addRequiredParam<FileName>("orientation_file",
                                    "Delimited file with one row of 9 values per grain.");
  return params;
}

GrainOrientationsFromFile::GrainOrientationsFromFile(const InputParameters & parameters)
  : TensorOperator<>(parameters), _orientation_file(getParam<FileName>("orientation_file"))
{
}

void
GrainOrientationsFromFile::computeBuffer()
{
  MooseUtils::DelimitedFileReader reader(_orientation_file);
  reader.setFormatFlag(MooseUtils::DelimitedFileReader::FormatFlag::ROWS);
  reader.setHeaderFlag(MooseUtils::DelimitedFileReader::HeaderFlag::OFF);
  reader.read();

  // With FormatFlag::ROWS, getData() is [row][column]: each outer entry is one file
  // line (one grain), and its inner vector holds that line's 9 matrix entries
  // (DelimitedFileReader.C's readRowData()/getData(), not the [column][row] layout
  // COLUMNS format would give).
  const auto & data = reader.getData();

  const auto n_grains = data.size();

  for (const auto & row : data)
    if (row.size() != 9)
      mooseError("orientation_file '",
                 _orientation_file,
                 "' must have exactly 9 columns (a row-major 3x3 matrix per grain), found ",
                 row.size(),
                 ".");

  std::vector<double> flat;
  flat.reserve(n_grains * 9);
  for (std::size_t g = 0; g < n_grains; ++g)
    for (std::size_t c = 0; c < 9; ++c)
      flat.push_back(data[g][c]);

  // Kept in float64 on CPU, independent of the simulation precision/device: the only consumer
  // (PolycrystalAnisotropicGBEnergy) builds its GB-energy hulls from these, and near special CSL
  // misorientations a float32 rounding of the matrices visibly changes GB5DOF's energies.
  _u = torch::from_blob(flat.data(), {static_cast<int64_t>(n_grains), 3, 3}, torch::kFloat64)
           .clone();
}
