""" Benchmark: Execution of full CADRE MDP. Serial version."""

from __future__ import print_function

import numpy as np

from openmdao.core.mpi_wrap import MPI
from openmdao.core.problem import Problem
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.solvers.petsc_ksp import PetscKSP
from openmdao.test.mpi_util import MPITestCase
from openmdao.test.util import assert_rel_error

try:
    from openmdao.core.petsc_impl import PetscImpl as impl
except ImportError:
    impl = None

from CADRE.CADRE_mdp import CADRE_MDP_Group


class BenchmarkExecSerial(MPITestCase):

    def benchmark_cadre_mdp_exec_full(self):

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

        model.setup(check=False)

        #----------------------------------------
        # This is what we are really profiling
        #----------------------------------------
        model.run()
        print(model['obj.val'])
        assert_rel_error(self, model['obj.val'], -393.805453491, 1.0e-5)

