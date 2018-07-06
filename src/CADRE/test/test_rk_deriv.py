""" Testing the derivativs on the RK component. """

import unittest

import numpy as np

from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from CADRE.rk4 import RK4


class RKTest(RK4):
    """
    Simple case with 2 states, 2 time points, and a 2-wide time-invariant input.
    """
    def __init__(self, n_times):
        super(RKTest, self).__init__()

        self.n_times = n_times

        self.options['state_var'] = 'x'
        self.options['init_state_var'] = 'x0'
        self.options['external_vars'] = ['yv']
        self.options['fixed_external_vars'] = ['yi']

    def setup(self):
        self.add_input('yi', np.zeros((2, 3)))
        self.add_input('yv', np.zeros((2, self.n_times)))
        self.add_input('x0', np.array([1.6, 2.7]))

        self.add_output('x', np.zeros((2, self.n_times)))

        self.declare_partials(of='*', wrt='*')

    def f_dot(self, external, state):
        """
        Time rate of change of state variables.

        Parameters
        ----------
        external: ndarray
            array of external variables for a single time step

        state: ndarray
            array of state variables for a single time step.
        """
        df = np.zeros((2))
        yinv = external[2:]

        for i in range(0, 2):
            df[i] = 0.1 * state[i] * yinv[0] * yinv[1] * \
                yinv[4] - .32 * state[i] * yinv[2] * yinv[3] * yinv[5]

        return df

    def df_dy(self, external, state):
        """
        Derivatives of states with respect to states.

        Parameters
        ----------
        external: ndarray
            array or external variables for a single time step

        state: ndarray
            array of state variables for a single time step.
        """
        df = np.zeros((2, 2))
        yinv = external[2:]

        for i in range(0, 2):
            df[i, i] = 0.1 * yinv[0] * yinv[i] * \
                yinv[4] - .32 * yinv[2] * yinv[3] * yinv[5]

        return df

    def df_dx(self, external, state):
        """
        Derivatives of states with respect to external vars.

        Parameters
        ----------
        external: ndarray
            array or external variables for a single time step

        state: ndarray
            array of state variables for a single time step.
        """
        df = np.zeros((2, 8))
        yinv = external[2:]

        for i in range(0, 2):
            df[i, 2] = 0.1 * state[i] * yinv[1] * yinv[4]
            df[i, 3] = 0.1 * state[i] * yinv[0] * yinv[4]
            df[i, 4] = -.32 * state[i] * yinv[3] * yinv[5]
            df[i, 5] = -.32 * state[i] * yinv[2] * yinv[5]
            df[i, 6] = 0.1 * state[i] * yinv[0] * yinv[1]
            df[i, 7] = -.32 * state[i] * yinv[2] * yinv[3]

        return df


NTIME = 3


class Testcase_RK_deriv(unittest.TestCase):

    def test_with_time_invariant(self):
        np.random.seed(1)

        indeps = IndepVarComp()
        rktest = RKTest(NTIME)

        indeps.add_output('yi', np.random.random((2, 3)))
        indeps.add_output('yv', np.random.random((2, NTIME)))
        indeps.add_output('x0', np.random.random((2,)))

        prob = Problem()
        model = prob.model

        model.add_subsystem('indeps', indeps, promotes=['*'])
        model.add_subsystem('rktest', rktest, promotes=['*'])

        prob.setup()
        prob.run_model()

        partials = prob.check_partials(out_stream=None)

        assert_check_partials(partials, atol=6e-5, rtol=6e-5)


if __name__ == '__main__':
    unittest.main()
