[![GitHub Actions Test Badge][1]][2]
[![Coveralls Badge][3]][4]

CADRE CubeSat design problem
----------------------------

This is an implementation of the CADRE CubeSat problem for OpenMDAO 3.x.

> [!NOTE]
> CADRE is an interesting optimization problem, but this code should not be used as an example of how to use OpenMDAO. This problem was implemented in the early days of OpenMDAO and has not been in active development for a long time, so it doesn't reflect the proper use of the current OpenMDAO API.

Instructions for the latest development version:

  `git clone git@github.com:OpenMDAO/CADRE`

  `cd CADRE`

  `pip install -e .[all]`

This will install CADRE with most of the required dependencies for testing, including [MBI][5].

Some examples use the [pyOptSparse][6] package with the [SNOPT][7] optimizer.
These will require that you have SNOPT, which you may be able to get [here][8].
Once you have SNOPT, you can follow the instructions [here][9] or use the [build_pyoptsparse][10] script to install it for use with OpenMDAO. e.g.

  `pip install git+https://github.com/OpenMDAO/build_pyoptsparse`

  `build_pyoptsparse -s /path/to/SNOPT/src`

For parallel execution you will also need MPI and [PETSc4py][11]:

  `pip install mpi4py petsc4py`
or
  `conda install mpi4py petsc4py`



[1]:  https://github.com/OpenMDAO/CADRE/actions/workflows/CADRE_test_workflow.yml/badge.svg "Github Actions Badge"
[2]:  https://github.com/OpenMDAO/CADRE/actions "Github Actions"

[3]:  https://coveralls.io/repos/github/OpenMDAO/CADRE/badge.svg?branch=master "Coverage Badge"
[4]:  https://coveralls.io/github/OpenMDAO/CADRE?branch=master "CADRE @Coveralls"

[5]:  https://github.com/OpenMDAO/MBI "MBI"

[6]:  https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/ "pyOptSparse"
[7]:  https://ccom.ucsd.edu/~optimizers/solvers/snopt/ "SNOPT"
[8]:  https://ccom.ucsd.edu/~optimizers/downloads/request/academic/ "UCSD"
[9]:  https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/SNOPT.html "MDO Lab"
[10]: https://github.com/OpenMDAO/build_pyoptsparse "build_pyoptsparse"


[11]: https://petsc.org/release/petsc4py/ "PETSc4py"
