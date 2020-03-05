"""
Optimization of the CADRE MDP.
"""
from __future__ import print_function

import unittest

import os
import pickle

import numpy as np

from openmdao.api import Problem, Group, ParallelGroup, ExplicitComponent, IndepVarComp, \
    pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error

from CADRE.power import Power_SolarPower, Power_CellVoltage
from CADRE.parameters import BsplineParameters
import CADRE.test


class Perf(ExplicitComponent):

    def __init__(self, n):
        super(Perf, self).__init__()

        self.n = n
        self.J = -np.ones((1, n))

    def setup(self):
        self.add_input('P_sol1', np.zeros((self.n, )), units='W',
                       desc='Solar panels power over time')

        self.add_input('P_sol2', np.zeros((self.n, )), units='W',
                       desc='Solar panels power over time')

        self.add_output('result', 0.0)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        outputs['result'] = -np.sum(inputs['P_sol1']) - np.sum(inputs['P_sol2'])

    def compute_partials(self, inputs, partials):
        partials['result', 'P_sol1'] = self.J
        partials['result', 'P_sol2'] = self.J


class MPPT(Group):

    def __init__(self, LOS, temp, area, m, n):
        super(MPPT, self).__init__()

        self.LOS = LOS
        self.temp = temp
        self.area = area
        self.m = m
        self.n = n

    def setup(self):
        m = self.m
        n = self.n

        param = IndepVarComp()
        param.add_output('LOS', self.LOS, units=None)
        param.add_output('temperature', self.temp, units='degK')
        param.add_output('exposedArea', self.area, units='m**2')
        param.add_output('CP_Isetpt', np.zeros((12, m)), units='A')

        self.add_subsystem('param', param)
        self.add_subsystem('bspline', BsplineParameters(n, m))
        self.add_subsystem('voltage', Power_CellVoltage(n))
        self.add_subsystem('power', Power_SolarPower(n))

        # self.add_subsystem('perf', Perf(n))

        self.connect('param.LOS', 'voltage.LOS')
        self.connect('param.temperature', 'voltage.temperature')
        self.connect('param.exposedArea', 'voltage.exposedArea')

        self.connect('param.CP_Isetpt', 'bspline.CP_Isetpt')
        self.connect('bspline.Isetpt', 'voltage.Isetpt')

        self.connect('bspline.Isetpt', 'power.Isetpt')
        self.connect('voltage.V_sol', 'power.V_sol')

        # self.connect('power.P_sol', 'perf.P_sol')


class MPPT_MDP(Group):

    def __init__(self):
        super(MPPT_MDP, self).__init__()
        self.n = 1500
        self.m = 300

    def setup(self):
        n = self.n
        m = self.m

        fpath = os.path.dirname(os.path.realpath(CADRE.test.__file__))
        data = pickle.load(open(fpath + '/data1346.pkl', 'rb'))

        pt0 = MPPT(data['0:LOS'], data['0:temperature'], data['0:exposedArea'], m, n)
        pt1 = MPPT(data['1:LOS'], data['1:temperature'], data['1:exposedArea'], m, n)

        # CADRE instances go into a Parallel Group
        para = self.add_subsystem('parallel', ParallelGroup(), promotes=['*'])

        para.add_subsystem('pt0', pt0)
        para.add_subsystem('pt1', pt1)

        self.add_subsystem('perf', Perf(1500))

        self.connect('pt0.power.P_sol', 'perf.P_sol1')
        self.connect('pt1.power.P_sol', 'perf.P_sol2')


class BenchmarkMPPT(unittest.TestCase):

    N_PROCS = 2

    def benchmark_mppt(self):
        model = MPPT_MDP()

        model.add_design_var('pt0.param.CP_Isetpt', lower=0., upper=0.4)
        model.add_design_var('pt1.param.CP_Isetpt', lower=0., upper=0.4)
        model.add_objective('perf.result')

        # create problem and add optimizer
        prob = Problem(model)
        prob.driver = pyOptSparseDriver(optimizer='SNOPT')
        prob.driver.opt_settings = {
            'Major optimality tolerance': 1e-3,
            'Major feasibility tolerance': 1.0e-5,
            'Iterations limit': 500000000,
            'New basis file': 10
        }
        prob.setup()
        prob.run_driver()

        # NOTE: adjusting tolerance from 1e-6 to 1e-5 because the answer is slightly
        #       different when running with Python 3.x and the latest numpy/scipy
        assert_rel_error(self, prob['perf.result'], -9.4308562238E+03, 1e-5)

