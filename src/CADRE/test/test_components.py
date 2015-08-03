""" Unit test each component in CADRE using some saved data from John's CMF
implementation."""

import os
import pickle
import random
import unittest
import warnings

import numpy as np

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.test.util import assert_rel_error

from CADRE.attitude import Attitude_Angular, Attitude_AngularRates, \
     Attitude_Attitude, Attitude_Roll, Attitude_RotationMtx, \
     Attitude_RotationMtxRates, Attitude_Sideslip, Attitude_Torque
from CADRE.battery import BatterySOC, BatteryPower, BatteryConstraints
from CADRE.comm import Comm_DataDownloaded, Comm_AntRotation, Comm_AntRotationMtx, \
     Comm_BitRate, Comm_Distance, Comm_EarthsSpin, Comm_EarthsSpinMtx, Comm_GainPattern, \
     Comm_GSposEarth, Comm_GSposECI, Comm_LOS, Comm_VectorAnt, Comm_VectorBody, \
     Comm_VectorECI, Comm_VectorSpherical
from CADRE.orbit import Orbit_Dynamics, Orbit_Initial
from CADRE.parameters import BsplineParameters
from CADRE.power import Power_CellVoltage, Power_SolarPower, Power_Total
from CADRE.reactionwheel import ReactionWheel_Motor, ReactionWheel_Power, \
     ReactionWheel_Torque, ReactionWheel_Dynamics
from CADRE.solar import Solar_ExposedArea
from CADRE.sun import Sun_LOS, Sun_PositionBody, Sun_PositionECI, \
     Sun_PositionSpherical
from CADRE.thermal_temperature import ThermalTemperature

# Ignore the numerical warnings from performing the rel error calc.
warnings.simplefilter("ignore")

idx = '5'

setd = {}
fpath = os.path.dirname(os.path.realpath(__file__))
data = pickle.load(open(fpath + "/data1346.pkl", 'rb'))

for key in data.keys():
    if key[0] == idx or not key[0].isdigit():
        if not key[0].isdigit():
            shortkey = key
        else:
            shortkey = key[2:]
        # set floats correctly
        if data[key].shape == (1,) and shortkey != "iSOC":
            setd[shortkey] = data[key][0]
        else:
            setd[shortkey] = data[key]

n = setd['P_comm'].size
m = setd['CP_P_comm'].size
h = 43200. / (n - 1)

setd['r_e2b_I0'] = np.zeros(6)
setd['r_e2b_I0'][:3] = data[idx + ":r_e2b_I0"]
setd['r_e2b_I0'][3:] = data[idx + ":v_e2b_I0"]
setd['Gamma'] = data[idx + ":gamma"]


class Testcase_CADRE(unittest.TestCase):

    """ Test run/step/stop aspects of a simple workflow. """

    def setUp(self):
        """ Called before each test. """
        self.model = Problem()
        self.model.root = Group()

    def tearDown(self):
        """ Called after each test. """
        self.model = None

    def setup(self, compname, inputs):

        try:
            self.model.root.add('comp', eval('%s(n)' % compname))
        except TypeError:
            # At least one comp has no args.
            try:
                self.model.root.add('comp', eval('%s()' % compname))
            except TypeError:
                self.model.root.add('comp', eval('%s(n, 300)' % compname))

        self.model.setup(check=False)

        for var in inputs:
            self.model['comp.%s' % var] = setd[var]

    def run_model(self):
        self.model.run()

    def compare_results(self, outputs):
        for var in outputs:
            tval = setd[var]
            assert np.linalg.norm(tval - self.model[
                'comp.%s' % var]) / np.linalg.norm(tval) < 1e-3

    def test_Attitude_Angular(self):

        compname = 'Attitude_Angular'
        inputs = ['O_BI', 'Odot_BI']
        outputs = ['w_B']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Attitude_AngularRates(self):

        compname = 'Attitude_AngularRates'
        inputs = ['w_B']
        outputs = ['wdot_B']

        self.setup(compname, inputs)
        self.model.root.comp.h = h
        self.run_model()
        self.compare_results(outputs)

    def test_Attitude_Attitude(self):

        compname = 'Attitude_Attitude'
        inputs = ['r_e2b_I']
        outputs = ['O_RI']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Attitude_Roll(self):

        compname = 'Attitude_Roll'
        inputs = ['Gamma']
        outputs = ['O_BR']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Attitude_RotationMtx(self):

        compname = 'Attitude_RotationMtx'
        inputs = ['O_BR', 'O_RI']
        outputs = ['O_BI']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Attitude_RotationMtxRates(self):

        compname = 'Attitude_RotationMtxRates'
        inputs = ['O_BI']
        outputs = ['Odot_BI']

        self.setup(compname, inputs)
        self.model.root.comp.h = h
        self.run_model()
        self.compare_results(outputs)

    def test_Attitude_Sideslip(self):

        compname = 'Attitude_Sideslip'
        inputs = ['r_e2b_I', 'O_BI']
        outputs = ['v_e2b_B']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Attitude_Torque(self):

        compname = 'Attitude_Torque'
        inputs = ['w_B', 'wdot_B']
        outputs = ['T_tot']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_BatterySOC(self):

        compname = 'BatterySOC'
        inputs = ['P_bat', 'temperature', 'iSOC']
        outputs = ['SOC']

        self.setup(compname, inputs)
        self.model.root.comp.h = h
        self.run_model()
        self.compare_results(outputs)

    def test_BatteryPower(self):

        compname = 'BatteryPower'
        inputs = ['SOC', 'temperature', 'P_bat']
        outputs = ['I_bat']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_BatteryConstraints(self):

        compname = 'BatteryConstraints'
        inputs = ['I_bat', 'SOC']
        outputs = ['ConCh', 'ConDs', 'ConS0', 'ConS1']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_DataDownloaded(self):

        compname = 'Comm_DataDownloaded'
        inputs = ['Dr']
        outputs = ['Data']

        self.setup(compname, inputs)
        self.model.root.comp.h = h
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_AntRotation(self):

        compname = 'Comm_AntRotation'
        inputs = ['antAngle']
        outputs = ['q_A']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_AntRotationMtx(self):

        compname = 'Comm_AntRotationMtx'
        inputs = ['q_A']
        outputs = ['O_AB']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_BitRate(self):

        compname = 'Comm_BitRate'
        inputs = ['P_comm', 'gain', 'GSdist', 'CommLOS']
        outputs = ['Dr']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_Distance(self):

        compname = 'Comm_Distance'
        inputs = ['r_b2g_A']
        outputs = ['GSdist']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_EarthsSpin(self):

        compname = 'Comm_EarthsSpin'
        inputs = ['t']
        outputs = ['q_E']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_EarthsSpinMtx(self):

        compname = 'Comm_EarthsSpinMtx'
        inputs = ['q_E']
        outputs = ['O_IE']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_GainPattern(self):

        compname = 'Comm_GainPattern'
        inputs = ['azimuthGS', 'elevationGS']
        outputs = ['gain']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_GSposEarth(self):

        compname = 'Comm_GSposEarth'
        inputs = ['lon', 'lat', 'alt']
        outputs = ['r_e2g_E']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_GSposECI(self):

        compname = 'Comm_GSposECI'
        inputs = ['O_IE', 'r_e2g_E']
        outputs = ['r_e2g_I']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_LOS(self):

        compname = 'Comm_LOS'
        inputs = ['r_b2g_I', 'r_e2g_I']
        outputs = ['CommLOS']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_VectorAnt(self):

        compname = 'Comm_VectorAnt'
        inputs = ['r_b2g_B', 'O_AB']
        outputs = ['r_b2g_A']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_VectorBody(self):

        compname = 'Comm_VectorBody'
        inputs = ['r_b2g_I', 'O_BI']
        outputs = ['r_b2g_B']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_VectorECI(self):

        compname = 'Comm_VectorECI'
        inputs = ['r_e2g_I', 'r_e2b_I']
        outputs = ['r_b2g_I']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Comm_VectorSpherical(self):

        compname = 'Comm_VectorSpherical'
        inputs = ['r_b2g_A']
        outputs = ['azimuthGS', 'elevationGS']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Orbit_Dynamics(self):

        compname = 'Orbit_Dynamics'
        inputs = ['r_e2b_I0']
        outputs = ['r_e2b_I']

        self.setup(compname, inputs)
        self.model.root.comp.h = h
        self.run_model()
        self.compare_results(outputs)

    def test_Orbit_Initial(self):

        # This component was not recorder in John's pickle.

        pass
        #compname = 'Orbit_Initial'
        #inputs = ['altPerigee', 'altApogee', 'RAAN', 'Inc', 'argPerigee',
                  #'trueAnomaly']
        #outputs = ['r_e2b_I0']

        #self.setup(compname, inputs)
        #self.run_model()
        #self.compare_results(outputs)

    def test_bspline_parameters(self):

        compname = 'BsplineParameters'
        inputs = ['CP_P_comm', 'CP_gamma', 'CP_Isetpt']
        outputs = ['P_comm', 'Gamma', 'Isetpt']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Power_CellVoltage(self):
        compname = 'Power_CellVoltage'
        inputs = ['LOS', 'temperature', 'exposedArea', 'Isetpt']
        outputs = ['V_sol']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Power_SolarPower(self):

        compname = 'Power_SolarPower'
        inputs = ['V_sol', 'Isetpt']
        outputs = ['P_sol']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Power_Total(self):

        compname = 'Power_Total'
        inputs = ['P_sol', 'P_comm', 'P_RW']
        outputs = ['P_bat']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_ReactionWheel_Motor(self):

        compname = 'ReactionWheel_Motor'
        inputs = ['T_RW', 'w_B', 'w_RW']
        outputs = ['T_m']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_ReactionWheel_Power(self):

        compname = 'ReactionWheel_Power'
        inputs = ['w_RW', 'T_RW']
        outputs = ['P_RW']

        self.setup(compname, inputs)

        self.run_model()
        self.compare_results(outputs)

    def test_ReactionWheel_Torque(self):

        compname = 'ReactionWheel_Torque'
        inputs = ['T_tot']
        outputs = ['T_RW']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_ReactionWheel_Dynamics(self):

        compname = 'ReactionWheel_Dynamics'
        inputs = ['w_B', 'T_RW']
        outputs = ['w_RW']

        self.setup(compname, inputs)
        self.model.root.comp.h = h
        self.run_model()
        self.compare_results(outputs)

    def test_Solar_ExposedArea(self):
        compname = 'Solar_ExposedArea'
        inputs = ['finAngle', 'azimuth', 'elevation']
        outputs = ['exposedArea']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Sun_LOS(self):

        compname = 'Sun_LOS'
        inputs = ['r_e2b_I', 'r_e2s_I']
        outputs = ['LOS']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Sun_PositionBody(self):

        compname = 'Sun_PositionBody'
        inputs = ['O_BI', 'r_e2s_I']
        outputs = ['r_e2s_B']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Sun_PositionECI(self):

        compname = 'Sun_PositionECI'
        inputs = ['t', 'LD']
        outputs = ['r_e2s_I']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_Sun_PositionSpherical(self):

        compname = 'Sun_PositionSpherical'
        inputs = ['r_e2s_B']
        outputs = ['azimuth', 'elevation']

        self.setup(compname, inputs)
        self.run_model()
        self.compare_results(outputs)

    def test_ThermalTemperature(self):

        compname = 'ThermalTemperature'
        inputs = ['exposedArea', 'cellInstd', 'LOS', 'P_comm']
        outputs = ['temperature']

        self.setup(compname, inputs)
        self.model.root.comp.h = h
        self.run_model()
        self.compare_results(outputs)

if __name__ == "__main__":

    unittest.main()
