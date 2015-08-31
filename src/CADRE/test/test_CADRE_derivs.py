""" Poor man's test_total_derivatives for CADRE."""

import os
import unittest

import numpy as np

from openmdao.components.param_comp import ParamComp
from openmdao.core.group import Group
from openmdao.core.problem import Problem

from CADRE.CADRE_group import CADRE


class Testcase_CADRE_deriv(unittest.TestCase):
    """ Test run/step/stop aspects of a simple workflow. """

    def test_RK4_issue(self):

        n = 60
        m = 20
        h = 43200.0 / (n - 1)

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
        root = top.root

        i = 0
        initial_params = {}
        initial_params["LD"] =  LDs[i]
        initial_params["r_e2b_I0"] =  r_e2b_I0s[i]

        root.add('bp1', ParamComp('cellInstd', np.ones((7, 12))))
        root.add('bp2', ParamComp('finAngle', np.pi/4.0))
        root.add('bp3', ParamComp('antAngle', 0.0))

        # User-defined initial parameters
        if initial_params is None:
            initial_params = {}
        if 't1' not in initial_params:
            initial_params['t1'] = 0.0
        if 't2' not in initial_params:
            initial_params['t2'] = 43200.0
        if 't' not in initial_params:
            initial_params['t'] = np.array(range(0, n))*h
        if 'CP_Isetpt' not in initial_params:
            initial_params['CP_Isetpt'] = 0.2 * np.ones((12, m))
        if 'CP_gamma' not in initial_params:
            initial_params['CP_gamma'] = np.pi/4 * np.ones((m, ))
        if 'CP_P_comm' not in initial_params:
            initial_params['CP_P_comm'] = 0.1 * np.ones((m, ))

        if 'iSOC' not in initial_params:
            initial_params['iSOC'] = np.array([0.5])

        # Fixed Station Parameters for the CADRE problem.
        # McMurdo station: -77.85, 166.666667
        # Ann Arbor: 42.2708, -83.7264
        if 'LD' not in initial_params:
            initial_params['LD'] = 5000.0
        if 'lat' not in initial_params:
            initial_params['lat'] = 42.2708
        if 'lon' not in initial_params:
            initial_params['lon'] = -83.7264
        if 'alt' not in initial_params:
            initial_params['alt'] = 0.256

        # Initial Orbital Elements
        if 'r_e2b_I0' not in initial_params:
            initial_params['r_e2b_I0'] = np.zeros((6, ))

        name = 'pt'

        top.root.add('pt', CADRE(n, m), promotes=['*'])
        #from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
        #top.root.ln_solver = LinearGaussSeidel()
        #top.root.pt.ln_solver = LinearGaussSeidel()

        # Some initial setup.
        root.add('%s_t1' % name, ParamComp('t1', initial_params['t1']))
        root.connect("%s_t1.t1" % name, '%s.t1'% name)
        root.add('%s_t2' % name, ParamComp('t2', initial_params['t2']))
        root.connect("%s_t2.t2" % name, '%s.t2'% name)
        root.add('%s_t' % name, ParamComp('t', initial_params['t']))
        root.connect("%s_t.t" % name, '%s.t'% name)

        # Design parameters
        root.add('%s_CP_Isetpt' % name, ParamComp('CP_Isetpt',
                                                    initial_params['CP_Isetpt']))
        root.connect("%s_CP_Isetpt.CP_Isetpt" % name,
                     '%s.CP_Isetpt'% name)
        root.add('%s_CP_gamma' % name, ParamComp('CP_gamma',
                                         initial_params['CP_gamma']))
        root.connect("%s_CP_gamma.CP_gamma" % name,
                     '%s.CP_gamma'% name)
        root.add('%s_CP_P_comm' % name, ParamComp('CP_P_comm',
                                          initial_params['CP_P_comm']))
        root.connect("%s_CP_P_comm.CP_P_comm" % name,
                     '%s.CP_P_comm'% name)
        root.add('%s_iSOC' % name, ParamComp('iSOC', initial_params['iSOC']))
        root.connect("%s_iSOC.iSOC" % name,
                     '%s.iSOC'% name)

        root.add('%s_param_LD' % name, ParamComp('LD', initial_params['LD']))
        root.connect("%s_param_LD.LD" % name,
                     '%s.LD'% name)
        root.add('%s_param_lat' % name, ParamComp('lat', initial_params['lat']))
        root.connect("%s_param_lat.lat" % name,
                     '%s.lat'% name)
        root.add('%s_param_lon' % name, ParamComp('lon', initial_params['lon']))
        root.connect("%s_param_lon.lon" % name,
                     '%s.lon'% name)
        root.add('%s_param_alt' % name, ParamComp('alt', initial_params['alt']))
        root.connect("%s_param_alt.alt" % name,
                     '%s.alt'% name)
        root.add('%s_param_r_e2b_I0' % name, ParamComp('r_e2b_I0',
                                             initial_params['r_e2b_I0']))
        root.connect("%s_param_r_e2b_I0.r_e2b_I0" % name,
                     '%s.r_e2b_I0'% name)

        # Hook up broadcast params
        root.connect('bp1.cellInstd', "%s.cellInstd" % name)
        root.connect('bp2.finAngle', "%s.finAngle" % name)
        root.connect('bp3.antAngle', "%s.antAngle" % name)

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
