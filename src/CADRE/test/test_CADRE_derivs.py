""" Test_total_derivatives for CADRE RK4 Component. """
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from CADRE.CADRE_group import CADRE


class TestCADRE(unittest.TestCase):

    def test_RK4_issue(self):

        n = 60
        m = 20

        LDs = [5233.5, 5294.5, 5356.5, 5417.5, 5478.5, 5537.5]

        r_e2b_I0s = [np.array([4505.29362, -3402.16069, -3943.74582,
                               4.1923899, -1.56280012, 6.14347427]),
                     np.array([-1005.46693, -597.205348, -6772.86532,
                               -0.61047858, -7.54623146, 0.75907455]),
                     np.array([4401.10539, 2275.95053, -4784.13188,
                               -5.26605537, -1.08194926, -5.37013745]),
                     np.array([-4969.91222, 4624.84149, 1135.9414,
                               0.1874654, -1.62801666, 7.4302362]),
                     np.array([-235.021232, 2195.72976, 6499.79919,
                               -2.55956031, -6.82743519, 2.21628099]),
                     np.array([-690.314375, -1081.78239, -6762.90367,
                               7.44316722, 1.19745345, -0.96035904])]

        prob = Problem()

        i = 0
        init = {
            'LD': LDs[i],
            'r_e2b_I0': r_e2b_I0s[i]
        }

        prob.model.add_subsystem('pt', CADRE(n, m, initial_inputs=init), promotes=['*'])

        prob.setup(check=True)
        prob.run_model()

        inputs = ['CP_gamma']
        outputs = ['Data']

        # check partials
        # partials = prob.check_partials()  # out_stream=None)
        # assert_check_partials(partials)

        # check totals
        J = prob.compute_totals(of=outputs, wrt=inputs)
        from pprint import pprint
        pprint(J)

        for outp in outputs:
            for inp in inputs:
                Jn = J[outp, inp]['J_fd']
                Jf = J[outp, inp]['J_fwd']
                np.set_prinprobtions(threshold='nan')
                print(np.nonzero(Jn))
                print(np.nonzero(Jf))
                diff = abs(Jf - Jn)
                assert_rel_error(self, diff.max(), 0.0, 1e-4)


if __name__ == '__main__':
    unittest.main()
