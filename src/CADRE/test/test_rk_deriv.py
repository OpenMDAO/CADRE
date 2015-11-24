""" Testing the derivativs on the RK component. """

import unittest

import numpy as np

from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.util import assert_rel_error

from CADRE.rk4 import RK4


class RKTest(RK4):
    ''' Simple case with 2 states, 2 time points, and a 2-wide time-invariant
    input'''

    def __init__(self, n_times):
        super(RKTest, self).__init__(0.01)

        self.add_param("yi", np.zeros((2, 3)))
        self.add_param("yv", np.zeros((2, n_times)))
        self.add_param("x0", np.array([1.6, 2.7]))

        self.add_output("x", np.zeros((2, n_times)))

        self.options['state_var'] = "x"
        self.options['init_state_var'] = "x0"
        self.options['external_vars'] = ["yv"]
        self.options['fixed_external_vars'] = ["yi"]

    def f_dot(self, external, state):

        df = np.zeros((2))
        yvar = external[:2]
        yinv = external[2:]

        for i in range(0, 2):
            df[i] = 0.1 * state[i] * yinv[0] * yinv[1] * \
                yinv[4] - .32 * state[i] * yinv[2] * yinv[3] * yinv[5]

        return df

    def df_dy(self, external, state):

        df = np.zeros((2, 2))
        yvar = external[:2]
        yinv = external[2:]

        for i in range(0, 2):
            df[i, i] = 0.1 * yinv[0] * yinv[i] * \
                yinv[4] - .32 * yinv[2] * yinv[3] * yinv[5]

        return df

    def df_dx(self, external, state):

        df = np.zeros((2, 8))
        yvar = external[:2]
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

    """ Test run/step/stop aspects of a simple workflow. """

    def setUp(self):
        """ Called before each test. """
        self.model = Problem()
        self.model.root = Group()

    def tearDown(self):
        """ Called after each test. """
        self.model = None

    def setup(self, compname, inputs, state0):

        self.model.root.add('comp', eval('%s(NTIME)' % compname),
                            promotes=['*'])

        for item in inputs + state0:
            pshape = self.model.root.comp._init_params_dict[item]['shape']
            self.model.root.add('p_%s' % item, IndepVarComp(item, np.zeros((pshape))),
                                promotes=['*'])

        self.model.setup(check=False)

        for item in inputs + state0:
            val = self.model['%s' % item]
            if hasattr(val, 'shape'):
                shape1 = val.shape
                self.model['%s' % item] = np.random.random(shape1)
            else:
                self.model['%s' % item] = np.random.random()

    def run_model(self):

        self.model.run()

    def compare_derivatives(self, inputs, outputs):

        model = self.model

        # Numeric
        Jn = model.calc_gradient(inputs, outputs, mode="fd",
                                 return_format='array')
        # print Jn

        # Analytic forward
        Jf = model.calc_gradient(inputs, outputs, mode='fwd',
                                 return_format='array')

        diff = abs(Jf - Jn)
        #print Jfz
        #print Jn
        assert_rel_error(self, diff.max(), 0.0, 6e-5)

        # Analytic adjoint
        Ja = model.calc_gradient(inputs, outputs, mode='rev',
                                 return_format='array')

        diff = abs(Ja - Jn)
        assert_rel_error(self, diff.max(), 0.0, 6e-5)

    def test_with_time_invariant(self):

        compname = 'RKTest'
        inputs = ['yi', 'yv']
        outputs = ['x']
        state0 = ['x0']
        np.random.seed(1)
        self.setup(compname, inputs, state0)
        self.run_model()
        self.compare_derivatives(inputs+state0, outputs)


if __name__ == "__main__":

    unittest.main()
