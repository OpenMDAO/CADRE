""" Benchmark: Gradient calculation of full CADRE MDP. Serial version."""

import unittest

import numpy as np

from openmdao.api import Problem, LinearBlockGS

from openmdao.utils.assert_utils import assert_rel_error

from CADRE.CADRE_mdp import CADRE_MDP_Group


class BenchmarkDerivsSerial(unittest.TestCase):

    def benchmark_cadre_mdp_derivs_full(self):
        # These numbers are for the CADRE problem in the paper.
        n = 1500
        m = 300
        npts = 6

        # Instantiate
        model = CADRE_MDP_Group(n=n, m=m, npts=npts)

        # Add design variables and constraints to each CADRE instance.
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):
            model.add_design_var("%s.CP_Isetpt" % name, lower=0., upper=0.4)
            model.add_design_var("%s.CP_gamma" % name, lower=0, upper=np.pi/2.)
            model.add_design_var("%s.CP_P_comm" % name, lower=0., upper=25.)
            model.add_design_var("%s.iSOC" % name, indices=[0], lower=0.2, upper=1.)

            model.add_constraint('%s.ConCh' % name, upper=0.0)
            model.add_constraint('%s.ConDs' % name, upper=0.0)
            model.add_constraint('%s.ConS0' % name, upper=0.0)
            model.add_constraint('%s.ConS1' % name, upper=0.0)
            model.add_constraint('%s_con5.val' % name, equals=0.0)

        # Add broadcast parameters
        model.add_design_var("bp.cellInstd", lower=0., upper=1.0)
        model.add_design_var("bp.finAngle", lower=0., upper=np.pi/2.)
        model.add_design_var("bp.antAngle", lower=-np.pi/4, upper=np.pi/4)

        # Add objective
        model.add_objective('obj.val')

        # create problem
        prob = Problem(model)
        prob.setup()

        # For serial execution, we will use LinearBlockGS
        model.linear_solver = LinearBlockGS()
        model.parallel.linear_solver = LinearBlockGS()
        model.parallel.pt0.linear_solver = LinearBlockGS()
        model.parallel.pt1.linear_solver = LinearBlockGS()
        model.parallel.pt2.linear_solver = LinearBlockGS()
        model.parallel.pt3.linear_solver = LinearBlockGS()
        model.parallel.pt4.linear_solver = LinearBlockGS()
        model.parallel.pt5.linear_solver = LinearBlockGS()

        # run
        prob.run_driver()

        #----------------------------------------
        # This is what we are really profiling
        #----------------------------------------
        J = prob.compute_totals()

        assert_rel_error(self, J['obj.val', 'bp.antAngle'][0][0],
                         67.15777407, 1e-4)
        assert_rel_error(self, J['obj.val', 'parallel.pt1.design.CP_gamma'][-1][-1],
                         0.62410223816776056, 1e-4)
