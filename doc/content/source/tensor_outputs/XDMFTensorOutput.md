# XDMFTensorOutput

!syntax description /TensorOutputs/XDMFTensorOutput

This TensorOutput object writes tensors to an [XDMF](https://www.xdmf.org/) data file set.
The parameter [!param](/TensorOutputs/XDMFTensorOutput/enable_hdf5) enables writing of the raw
tensor data to an HDF5 file. The resulting data items will be bamed `/var.n`, where `var` is the name of the
exported tensor and `n` is the simulation step. Output as cell or node centered data is supported
and can be selected using the [!param](/TensorOutputs/XDMFTensorOutput/output_mode) parameter. Cell centered output
results in a value per simulation grid cell (e.g. `N[0] * N[1] * N[2]` entries), while for node centered output
the cell edge nodes are periodically replicated, resulting in `(N[0]+1) * (N[1]+1) * (N[2]+1)` exported entries.

The [!param](/TensorOutputs/XDMFTensorOutput/interval) parameter limits the frequency of
time-step based executions in the same manner as the [!param](/Outputs/interval) option in the
standard [Moose Outputs block](syntax/Outputs/index.md). When set to `n`, the output object writes
results on every `n`-th time step while still honoring any `execute_on` settings.

## Overview

!! Replace these lines with information regarding the XDMFTensorOutput object.

## Example Input File Syntax

!! Describe and include an example of how to use the XDMFTensorOutput object.

!syntax parameters /TensorOutputs/XDMFTensorOutput

!syntax inputs /TensorOutputs/XDMFTensorOutput

!syntax children /TensorOutputs/XDMFTensorOutput
