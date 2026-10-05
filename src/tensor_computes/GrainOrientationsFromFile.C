/**********************************************************************/
/*                     DO NOT MODIFY THIS HEADER                      */
/*            Marlin, a Fourier spectral solver for MOOSE             */
/*                                                                    */
/*            Copyright 2024 Battelle Energy Alliance, LLC            */
/*                        ALL RIGHTS RESERVED                         */
/**********************************************************************/

#include "GrainOrientationsFromFile.h"

#include "DelimitedFileReader.h"
#include "MarlinUtils.h"

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

  // getData() is always [column][row]; with ROWS input that means
  // columns 0..8 are the 9 matrix entries and rows are grains.
  const auto & data = reader.getData();

  if (data.size() != 9)
    mooseError("orientation_file '",
               _orientation_file,
               "' must have exactly 9 columns (a row-major 3x3 matrix per grain), found ",
               data.size(),
               ".");

  const auto n_grains = data[0].size();

  std::vector<double> flat;
  flat.reserve(n_grains * 9);
  for (std::size_t g = 0; g < n_grains; ++g)
    for (std::size_t c = 0; c < 9; ++c)
      flat.push_back(data[c][g]);

  _u = torch::from_blob(flat.data(), {static_cast<int64_t>(n_grains), 3, 3}, torch::kFloat64)
           .to(MooseTensor::floatTensorOptions())
           .clone();
}
