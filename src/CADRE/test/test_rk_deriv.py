""" Testing the derivativs on the RK component. """

import unittest

import numpy as np

from openmdao.core.problem import Problem
from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from CADRE.rk4 import RK4


class RKTest(RK4):
    """
    Simple case with 2 states, 2 time points, and a 2-wide time-invariant input.
    """

    def __init__(self, n_times):
        super(RKTest, self).__init__(0.01)

        self.add_input("yi", np.zeros((2, 3)))
        self.add_input("yv", np.zeros((2, n_times)))
        self.add_input("x0", np.array([1.6, 2.7]))

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
        self.prob = Problem()

    def tearDown(self):
        """ Called after each test. """
        self.prob = None

    def setup(self, compname, inputs, state0):

        self.prob.model.add_subsystem('comp', eval('%s(NTIME)' % compname), promotes=['*'])

        for item in inputs + state0:
            pshape = self.prob.model.comp._init_inputs_dict[item]['shape']
            self.prob.model.add_subsystem('p_%s' % item,
                                          IndepVarComp(item, np.zeros((pshape))),
                                          promotes=['*'])

        self.prob.setup(check=False)

        for item in inputs + state0:
            val = self.prob['%s' % item]
            if hasattr(val, 'shape'):
                shape1 = val.shape
                self.prob['%s' % item] = np.random.random(shape1)
            else:
                self.prob['%s' % item] = np.random.random()

    def run_model(self):

        self.prob.run()

    def compare_derivatives(self, inputs, outputs):

        prob = self.prob

        # Numeric
        Jn = prob.calc_gradient(inputs, outputs, mode="fd",
                                 return_format='array')
        # print Jn

        # Analytic forward
        Jf = prob.calc_gradient(inputs, outputs, mode='fwd',
                                 return_format='array')

        diff = abs(Jf - Jn)
        #print Jfz
        #print Jn
        assert_rel_error(self, diff.max(), 0.0, 6e-5)

        # Analytic adjoint
        Ja = prob.calc_gradient(inputs, outputs, mode='rev',
                                 return_format='array')

        diff = abs(Ja - Jn)
        assert_rel_error(self, diff.max(), 0.0, 6e-5)

    def old_test_with_time_invariant(self):

        compname = 'RKTest'
        inputs = ['yi', 'yv']
        outputs = ['x']
        state0 = ['x0']
        np.random.seed(1)

        self.setup(compname, inputs, state0)
        self.run_model()
        self.compare_derivatives(inputs+state0, outputs)

    def test_with_time_invariant(self):
        np.random.seed(1)

        prob = Problem()
        model = prob.model

        model.add_subsystem('comp', RKTest(NTIME), promotes=['*'])

        inputs = ['yi', 'yv']
        outputs = ['x']
        state0 = ['x0']

        for name, meta in model.comp.list_inputs():
            print(name, meta)
            model.add_subsystem('p_%s' % name,
                                IndepVarComp(item, np.zeros(meta['value'].shape)),
                                promotes=['*'])

        prob.setup(check=True)

        for name, meta in model.comp.list_inputs():
            val = self.prob['%s' % name]
            self.prob[name] = np.random.random(val.shape)


if __name__ == "__main__":
    unittest.main()
