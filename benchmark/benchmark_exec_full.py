"""
Benchmark: Execution of full CADRE MDP. Serial version.
"""
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_rel_error

from CADRE.CADRE_mdp import CADRE_MDP_Group


class BenchmarkExecSerial(unittest.TestCase):

    def benchmark_cadre_mdp_exec_full(self):

        # These numbers are for the CADRE problem in the paper.
        n = 1500
        m = 300
        npts = 6

        # instantiate model
        model = CADRE_MDP_Group(n=n, m=m, npts=npts)

        # add design variables and constraints to each CADRE instance.
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):
            model.add_design_var('%s.CP_Isetpt' % name, lower=0., upper=0.4)
            model.add_design_var('%s.CP_gamma' % name, lower=0, upper=np.pi/2.)
            model.add_design_var('%s.CP_P_comm' % name, lower=0., upper=25.)
            model.add_design_var('%s.iSOC' % name, indices=[0], lower=0.2, upper=1.)

            model.add_constraint('%s.ConCh' % name, upper=0.0)
            model.add_constraint('%s.ConDs' % name, upper=0.0)
            model.add_constraint('%s.ConS0' % name, upper=0.0)
            model.add_constraint('%s.ConS1' % name, upper=0.0)
            model.add_constraint('%s_con5.val' % name, equals=0.0)

        # add broadcast parameters
        model.add_design_var('bp.cellInstd', lower=0., upper=1.0)
        model.add_design_var('bp.finAngle', lower=0., upper=np.pi/2.)
        model.add_design_var('bp.antAngle', lower=-np.pi/4, upper=np.pi/4)

        # add objective
        model.add_objective('obj.val')

        # create problem
        prob = Problem(model)
        prob.setup()

        # ----------------------------------------
        # This is what we are really profiling
        # ----------------------------------------
        prob.run_driver()

        assert_rel_error(self, prob['obj.val'], -393.789941398, 1.0e-4)
