""" Benchmark: Gradient calculation of full CADRE MDP. Parallel version."""

from __future__ import print_function

import unittest

import numpy as np

from openmdao.core.problem import Problem
from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.utils.assert_utils import assert_rel_error

from CADRE.CADRE_mdp import CADRE_MDP_Group


class BenchmarkDerivsParallel(unittest.TestCase):

    N_PROCS = 6

    def benchmark_cadre_mdp_derivs_full_par(self):

        # These numbers are for the CADRE problem in the paper.
        n = 1500
        m = 300
        npts = 6

        # Instantiate
        model = CADRE_MDP_Group(n=n, m=m, npts=npts)

        # Add parameters and constraints to each CADRE instance.
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):

            # add parameters to driver
            model.add_desvar("%s.CP_Isetpt" % name, lower=0., upper=0.4)
            model.add_desvar("%s.CP_gamma" % name, lower=0, upper=np.pi/2.)
            model.add_desvar("%s.CP_P_comm" % name, lower=0., upper=25.)
            model.add_desvar("%s.iSOC" % name, indices=[0], lower=0.2, upper=1.)

            model.add_constraint('%s.ConCh' % name, upper=0.0)
            model.add_constraint('%s.ConDs' % name, upper=0.0)
            model.add_constraint('%s.ConS0' % name, upper=0.0)
            model.add_constraint('%s.ConS1' % name, upper=0.0)
            model.add_constraint('%s_con5.val' % name, equals=0.0)

        # Add Parameter groups
        model.add_desvar("bp.cellInstd", lower=0., upper=1.0)
        model.add_desvar("bp.finAngle", lower=0., upper=np.pi/2.)
        model.add_desvar("bp.antAngle", lower=-np.pi/4, upper=np.pi/4)

        # Add objective
        model.driver.add_objective('obj.val')

        prob = Problem(model)

        model.linear_solver = LinearGaussSeidel()
        model.parallel.linear_solver = LinearGaussSeidel()
        model.parallel.pt0.linear_solver = LinearGaussSeidel()
        model.parallel.pt1.linear_solver = LinearGaussSeidel()
        model.parallel.pt2.linear_solver = LinearGaussSeidel()
        model.parallel.pt3.linear_solver = LinearGaussSeidel()
        model.parallel.pt4.linear_solver = LinearGaussSeidel()
        model.parallel.pt5.linear_solver = LinearGaussSeidel()

        prob.setup(check=False)
        prob.run_driver()

        # ----------------------------------------
        # This is what we are really profiling
        # ----------------------------------------
        J = prob.compute_totals()

        assert_rel_error(self, J['obj.val']['bp.antAngle'][0][0], 67.15777407, 1e-4)
        assert_rel_error(self, J['obj.val']['pt1.CP_gamma'][-1][-1], -0.62410223816776056, 1e-4)
