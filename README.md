# turbine_opt
Balancing turbines by optimizing blade distribution

Python package, with optimization object -Simulated Annealing model-.

This package help dispose blades in a turbine in order to minimize global
unbalance. Disposing blades with different masses randomly would lead
to unbalance which can break the turbine when it rotates at high speed.

Another constraint is that the different weights should form up 4 lobes of
mass, in order to maximize turbine lifespan.

Please run lib.py in ipython for testing, results will show up as images.
A script is being written to call the optimzation automatically from command line.
