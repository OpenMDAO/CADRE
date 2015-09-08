""" Poor man's test_total_derivatives for CADRE."""

import os
import unittest

import numpy as np

from openmdao.core.group import Group
from openmdao.core.problem import Problem

from CADRE.CADRE_group import CADRE


class Testcase_CADRE_deriv(unittest.TestCase):
    """ Test run/step/stop aspects of a simple workflow. """

    def test_RK4_issue(self):

        n = 60
        m = 20

        LDs = [5233.5, 5294.5, 5356.5, 5417.5, 5478.5, 5537.5]

        r_e2b_I0s = [np.array([4505.29362, -3402.16069, -3943.74582,
                               4.1923899, -1.56280012,  6.14347427]),
                     np.array(
                         [-1005.46693,  -597.205348, -6772.86532, -0.61047858,
                          -7.54623146,  0.75907455]),
                     np.array(
                         [4401.10539,  2275.95053, -4784.13188, -5.26605537,
                          -1.08194926, -5.37013745]),
                     np.array(
                         [-4969.91222,  4624.84149,  1135.9414,  0.1874654,
                          -1.62801666,  7.4302362]),
                     np.array(
                         [-235.021232,  2195.72976,  6499.79919, -2.55956031,
                          -6.82743519,  2.21628099]),
                     np.array(
                         [-690.314375, -1081.78239, -6762.90367,  7.44316722,
                          1.19745345, -0.96035904])]

        top = Problem(root=Group())

        i = 0
        init = {}
        init["LD"] =  LDs[i]
        init["r_e2b_I0"] =  r_e2b_I0s[i]

        top.root.add('pt', CADRE(n, m, initial_params=init), promotes=['*'])

        #from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
        #top.root.ln_solver = LinearGaussSeidel()
        #top.root.pt.ln_solver = LinearGaussSeidel()

        top.setup(check=False)
        top.run()

        inputs = ['CP_gamma']
        outputs = ['Data']

        from time import time
        t0 = time()
        J1 = top.calc_gradient(inputs, outputs, mode='fwd')
        print('Forward', time()-t0)
        t0 = time()
        J2 = top.calc_gradient(inputs, outputs, mode='rev')
        print('Reverse', time()-t0)
        t0 = time()
        Jfd = top.calc_gradient(inputs, outputs, mode='fd')
        print('Fd', time()-t0)

        np.set_printoptions(threshold='nan')
        #print np.nonzero(J1)
        #print np.nonzero(J2)
        #print np.nonzero(Jfd)
        #print J1
        #print J2
        #print Jfd
        print np.max(abs(J1-Jfd))
        print np.max(abs(J2-Jfd))
        print np.max(abs(J1-J2))

        self.assertTrue( np.max(abs(J1-J2)) < 1.0e-6 )
        self.assertTrue( np.max(abs(J1-Jfd)) < 1.0e-4 )
        self.assertTrue( np.max(abs(J2-Jfd)) < 1.0e-4 )

if __name__ == "__main__":

    unittest.main()
