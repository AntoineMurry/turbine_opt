# turbine_opt

Balancing turbines by optimizing blade distribution

Python package, with optimization object -Simulated Annealing model-.

This package help distribute blades on a turbine in order to minimize global
unbalance. Distributing blades with different masses randomly would lead
to a global unbalance which can break the turbine when it rotates at high
 speed.

Another constraint is that the different weights should form up 4 lobes of
mass, in order to maximize turbine lifespan.

Please run lib.py in ipython for testing, results will show up as images.
A script is being written to call the optimzation automatically from command line.


Below is the tree of the package. 

For now, all code in in lib.py, TODO: split per function, e.g. Optimization
objects should be in a separated python file.

Unit tests are in the tests/ folder.

In order for the user to be able to play with the package, a sample of data
(input-data.csv) has be placed in turbine-opt/data/. This dataset
corresponds to a set of blades which needs to be distributed in an optimal
manner on the turbine, in order to minimize global unbalance.

.
├── Makefile
├── README.md
├── scripts
│   └── turbine-opt-example
├── setup.py
├── tests
│   ├── __init__.py
│   └── lib_tests.py
├── turbine_opt
    ├── data
    │   ├── __init__.py
    │   ├── input-data.csv
    │   └── test_data
    │       ├── input-data.csv
    ├── __init__.py
    ├── lib.py
    └── version.txt
