# ETDRK4Solver

!alert construction title=Undocumented Class
The ETDRK4Solver has not been documented. The content listed below should be used as a starting point for documenting the class.

!syntax description /TensorSolver/ETDRK4Solver

## Overview

An exponential time differencing fourth-order Runge-Kutta solver. A one-dimensional diffusion example with the analytical solution $u(x,t)=\sin(x)\exp(-t)$ enables direct comparison with existing diffusion solvers.

## Example Input File Syntax

!listing test/tests/solvers/diffusion_etdrk4.i line_numbers=false

!syntax parameters /TensorSolver/ETDRK4Solver

!syntax inputs /TensorSolver/ETDRK4Solver

!syntax children /TensorSolver/ETDRK4Solver
