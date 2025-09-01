# ETDRK4Solver

!alert construction title=Undocumented Class
The ETDRK4Solver has not been documented. The content listed below should be used as a starting point for documenting the class, which includes the typical automatic documentation associated with a
MooseObject; however, what is contained is ultimately determined by what is necessary to make the
documentation clear for users.

!syntax description /TensorSolver/ETDRK4Solver

## Overview

An exponential time differencing fourth-order Runge-Kutta solver. A one-dimensional diffusion example with the analytical solution $u(x,t)=\sin(x)\exp(-t)$ enables direct comparison with existing diffusion solvers.

## Example Input File Syntax

!listing test/tests/solvers/diffusion_etdrk4.i line_numbers=false

Performs fourth-order exponential time differencing Runge--Kutta (ETDRK4) time integration.

## Example Input File Syntax

!! Describe and include an example of how to use the ETDRK4Solver object.

!syntax parameters /TensorSolver/ETDRK4Solver

!syntax inputs /TensorSolver/ETDRK4Solver

!syntax children /TensorSolver/ETDRK4Solver
