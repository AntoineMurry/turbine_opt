# turbine_opt

Balancing turbines by optimizing blade distribution

Python package, with optimization object -Simulated Annealing model-.

This package help distribute blades on a turbine in order to minimize global
unbalance. Distributing blades with different masses randomly would lead
to a global unbalance which can break the turbine when it rotates at high
 speed.

Another constraint is that the different weights should form up 4 lobes of
mass, in order to maximize turbine lifespan.

A script has been created for demo. Install the package by typing
 "$ make install" at package root, then simply type "$ turbine-opt-example".

Below is the tree of the package. 

Optimization code is in optimization.py, other code specific to the
turbine problem is in lib.py.

Unit tests are in the tests/ folder.

In order for the user to be able to play with the package, a sample of data
(input-data.csv) has be placed in turbine-opt/data/. This dataset
corresponds to a set of blades which needs to be distributed in an optimal
manner on the turbine, in order to minimize global unbalance.

To DO: wrap model into web app (flask or django) to create user-friendly interface.


```
.
├── build
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
    ├── optimization.py
    └── version.txt

```
