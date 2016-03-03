""" Benchmark: Gradient calculation of full CADRE MDP. Parallel version."""

from __future__ import print_function

import numpy as np

from openmdao.core.mpi_wrap import MPI
from openmdao.core.problem import Problem
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.solvers.petsc_ksp import PetscKSP
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl

from CADRE.CADRE_mdp import CADRE_MDP_Group


class BenchmarkDerivsParallel(MPITestCase):
    N_PROCS=6

    def benchmark_cadre_mdp_derivs_full_par(self):

        # These numbers are for the CADRE problem in the paper.
        n = 1500
        m = 300
        npts = 6

        # Instantiate
        model = Problem(impl=impl)
        root = model.root = CADRE_MDP_Group(n=n, m=m, npts=npts)

        # Add parameters and constraints to each CADRE instance.
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):

            # add parameters to driver
            model.driver.add_desvar("%s.CP_Isetpt" % name, lower=0., upper=0.4)
            model.driver.add_desvar("%s.CP_gamma" % name, lower=0, upper=np.pi/2.)
            model.driver.add_desvar("%s.CP_P_comm" % name, lower=0., upper=25.)
            model.driver.add_desvar("%s.iSOC" % name, indices=[0], lower=0.2, upper=1.)

            model.driver.add_constraint('%s.ConCh'% name, upper=0.0)
            model.driver.add_constraint('%s.ConDs'% name, upper=0.0)
            model.driver.add_constraint('%s.ConS0'% name, upper=0.0)
            model.driver.add_constraint('%s.ConS1'% name, upper=0.0)
            model.driver.add_constraint('%s_con5.val'% name, equals=0.0)

        # Add Parameter groups
        model.driver.add_desvar("bp1.cellInstd", lower=0., upper=1.0)
        model.driver.add_desvar("bp2.finAngle", lower=0., upper=np.pi/2.)
        model.driver.add_desvar("bp3.antAngle", lower=-np.pi/4, upper=np.pi/4)


        # Add objective
        model.driver.add_objective('obj.val')

        model.root.ln_solver = LinearGaussSeidel()
        model.root.parallel.ln_solver = LinearGaussSeidel()
        model.root.parallel.pt0.ln_solver = LinearGaussSeidel()
        model.root.parallel.pt1.ln_solver = LinearGaussSeidel()
        model.root.parallel.pt2.ln_solver = LinearGaussSeidel()
        model.root.parallel.pt3.ln_solver = LinearGaussSeidel()
        model.root.parallel.pt4.ln_solver = LinearGaussSeidel()
        model.root.parallel.pt5.ln_solver = LinearGaussSeidel()

        model.setup(check=False)
        model.run()

        params = list(model.driver.get_desvars().keys())
        unks = list(model.driver.get_objectives().keys()) + list(model.driver.get_constraints().keys())

        #----------------------------------------
        # This is what we are really profiling
        #----------------------------------------
        J = model.calc_gradient(params, unks, mode='rev', return_format='dict')

        assert_rel_error(self, J['obj.val']['bp3.antAngle'], 67.13247594, 1e-5)
        assert_rel_error(self, J['obj.val']['pt1.CP_gamma'][-1][-1], -0.62480529972074561, 1e-5)

