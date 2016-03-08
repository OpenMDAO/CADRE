""" Run the CADRE model and make sure the value compare to the saved pickle."""

from __future__ import print_function

import sys
import os
import pickle
import unittest
import warnings

import numpy as np

from openmdao.core.problem import Problem

from CADRE.CADRE_group import CADRE

# Ignore the numerical warnings from performing the rel error calc.
warnings.simplefilter("ignore")


idx = '0'

setd = {}
fpath = os.path.dirname(os.path.realpath(__file__))
if sys.version_info.major == 2:
    data = pickle.load(open(fpath + "/data1346_py2.pkl", 'rb'))
else:
    data = pickle.load(open(fpath + "/data1346_py2.pkl", 'rb'))

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

assembly = Problem(root=CADRE(n, m))
#assembly.setup()
assembly.setup(check=False)

setd['r_e2b_I0'] = np.zeros(6)
setd['r_e2b_I0'][:3] = data[idx + ":r_e2b_I0"]
setd['r_e2b_I0'][3:] = data[idx + ":v_e2b_I0"]
setd['Gamma'] = data[idx + ":gamma"]

assembly['CP_P_comm'] = setd['CP_P_comm']
assembly['LD'] = setd['LD']
assembly['cellInstd'] = setd['cellInstd']
assembly['CP_gamma'] = setd['CP_gamma']
assembly['finAngle'] = setd['finAngle']
assembly['lon'] = setd['lon']
assembly['CP_Isetpt'] = setd['CP_Isetpt']
assembly['antAngle'] = setd['antAngle']
assembly['t'] = setd['t']
assembly['r_e2b_I0'] = setd['r_e2b_I0']
assembly['lat'] = setd['lat']
assembly['alt'] = setd['alt']
assembly['iSOC'] = setd['iSOC']

assembly.run()


class Testcase_CADRE_assembly(unittest.TestCase):

    """ Tests the CADRE assembly. """

    def compare(self, compname, inputs, outputs):
        for var in inputs + outputs:
            computed = assembly[var]
            actual = setd[var]
            if isinstance(computed, np.ndarray):
                rel = np.linalg.norm(
                    actual - computed) / np.linalg.norm(actual)
            else:
                rel = np.abs(actual - computed) / np.abs(actual)
            if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                #print(var)
                #print(computed)
                #print(actual)
                assert rel <= 1e-3

    def test_Comm_DataDownloaded(self):

        compname = 'Comm_DataDownloaded'
        inputs = ['Dr']
        outputs = ['Data']

        self.compare(compname, inputs, outputs)

    def test_Comm_AntRotation(self):

        compname = 'Comm_AntRotation'
        inputs = ['antAngle']
        outputs = ['q_A']

        self.compare(compname, inputs, outputs)

    def test_Comm_BitRate(self):

        compname = 'Comm_BitRate'
        inputs = ['P_comm', 'gain', 'GSdist', 'CommLOS']
        outputs = ['Dr']

        self.compare(compname, inputs, outputs)

    def test_Comm_Distance(self):

        compname = 'Comm_Distance'
        inputs = ['r_b2g_A']
        outputs = ['GSdist']

        self.compare(compname, inputs, outputs)

    def test_Comm_EarthsSpin(self):

        compname = 'Comm_EarthsSpin'
        inputs = ['t']
        outputs = ['q_E']

        self.compare(compname, inputs, outputs)

    def test_Comm_EarthsSpinMtx(self):

        compname = 'Comm_EarthsSpinMtx'
        inputs = ['q_E']
        outputs = ['O_IE']

        self.compare(compname, inputs, outputs)

    def test_Comm_GainPattern(self):
        compname = 'Comm_GainPattern'
        inputs = ['azimuthGS', 'elevationGS']
        outputs = ['gain']

        self.compare(compname, inputs, outputs)

    def test_Comm_GSposEarth(self):

        compname = 'Comm_GSposEarth'
        inputs = ['lon', 'lat', 'alt']
        outputs = ['r_e2g_E']

        self.compare(compname, inputs, outputs)

    def test_Comm_GSposECI(self):

        compname = 'Comm_GSposECI'
        inputs = ['O_IE', 'r_e2g_E']
        outputs = ['r_e2g_I']

        self.compare(compname, inputs, outputs)

    def test_Comm_LOS(self):

        compname = 'Comm_LOS'
        inputs = ['r_b2g_I', 'r_e2g_I']
        outputs = ['CommLOS']

        self.compare(compname, inputs, outputs)

    def test_Comm_VectorAnt(self):

        compname = 'Comm_VectorAnt'
        inputs = ['r_b2g_B', 'O_AB']
        outputs = ['r_b2g_A']

        self.compare(compname, inputs, outputs)

    def test_Comm_VectorBody(self):

        compname = 'Comm_VectorBody'
        inputs = ['r_b2g_I', 'O_BI']
        outputs = ['r_b2g_B']

        self.compare(compname, inputs, outputs)

    def test_Comm_VectorECI(self):

        compname = 'Comm_VectorECI'
        inputs = ['r_e2g_I', 'r_e2b_I']
        outputs = ['r_b2g_I']

        self.compare(compname, inputs, outputs)

    def test_Comm_VectorSpherical(self):

        compname = 'Comm_VectorSpherical'
        inputs = ['r_b2g_A']
        outputs = ['azimuthGS', 'elevationGS']

        self.compare(compname, inputs, outputs)

    def test_ThermalTemperature(self):

        compname = 'ThermalTemperature'
        inputs = ['exposedArea', 'cellInstd', 'LOS', 'P_comm']
        outputs = ['temperature']

        self.compare(compname, inputs, outputs)

    def test_Attitude_Angular(self):

        compname = 'Attitude_Angular'
        inputs = ['O_BI', 'Odot_BI']
        outputs = ['w_B']

        self.compare(compname, inputs, outputs)

    def test_Attitude_AngularRates(self):

        compname = 'Attitude_AngularRates'
        inputs = ['w_B']
        outputs = ['wdot_B']

        self.compare(compname, inputs, outputs)

    def test_Attitude_Attitude(self):

        compname = 'Attitude_Attitude'
        inputs = ['r_e2b_I']
        outputs = ['O_RI']

        self.compare(compname, inputs, outputs)

    def test_Attitude_Roll(self):

        compname = 'Attitude_Roll'
        inputs = ['Gamma']
        outputs = ['O_BR']

        self.compare(compname, inputs, outputs)

    def test_Attitude_RotationMtx(self):

        compname = 'Attitude_RotationMtx'
        inputs = ['O_BR', 'O_RI']
        outputs = ['O_BI']

        self.compare(compname, inputs, outputs)

    def test_Attitude_RotationMtxRates(self):

        compname = 'Attitude_RotationMtxRates'
        inputs = ['O_BI']
        outputs = ['Odot_BI']

        self.compare(compname, inputs, outputs)

    # def test_Attitude_Sideslip(self):

    #     compname = 'Attitude_Sideslip'
    #     inputs = ['r_e2b_I', 'O_BI']
    #     outputs = ['v_e2b_B']

    #     self.compare(compname, inputs, outputs)

    def test_Attitude_Torque(self):

        compname = 'Attitude_Torque'
        inputs = ['w_B', 'wdot_B']
        outputs = ['T_tot']

        self.compare(compname, inputs, outputs)

    def test_Sun_LOS(self):

        compname = 'Sun_LOS'
        inputs = ['r_e2b_I', 'r_e2s_I']
        outputs = ['LOS']

        self.compare(compname, inputs, outputs)

    def test_Sun_PositionBody(self):

        compname = 'Sun_PositionBody'
        inputs = ['O_BI', 'r_e2s_I']
        outputs = ['r_e2s_B']

        self.compare(compname, inputs, outputs)

    def test_Sun_PositionECI(self):

        compname = 'Sun_PositionECI'
        inputs = ['t', 'LD']
        outputs = ['r_e2s_I']

        self.compare(compname, inputs, outputs)

    def test_Sun_PositionSpherical(self):

        compname = 'Sun_PositionSpherical'
        inputs = ['r_e2s_B']
        outputs = ['azimuth', 'elevation']

        self.compare(compname, inputs, outputs)

    def test_Solar_ExposedArea(self):
        compname = 'Solar_ExposedArea'
        inputs = ['finAngle', 'azimuth', 'elevation']
        outputs = ['exposedArea']

        self.compare(compname, inputs, outputs)

    def test_Power_CellVoltage(self):
        compname = 'Power_CellVoltage'
        inputs = ['LOS', 'temperature', 'exposedArea', 'Isetpt']
        outputs = ['V_sol']

        self.compare(compname, inputs, outputs)

    def test_Power_SolarPower(self):

        compname = 'Power_SolarPower'
        inputs = ['V_sol', 'Isetpt']
        outputs = ['P_sol']

        self.compare(compname, inputs, outputs)

    def test_Power_Total(self):

        compname = 'Power_Total'
        inputs = ['P_sol', 'P_comm', 'P_RW']
        outputs = ['P_bat']

        self.compare(compname, inputs, outputs)

    # def test_ReactionWheel_Motor(self):

    #     compname = 'ReactionWheel_Motor'
    #     inputs = ['T_RW', 'w_B', 'w_RW']
    #     outputs = ['T_m']

    #     self.compare(compname, inputs, outputs)

    def test_ReactionWheel_Dynamics(self):

        compname = 'ReactionWheel_Dynamics'
        inputs = ['w_B', 'T_RW']
        outputs = ['w_RW']

        self.compare(compname, inputs, outputs)

    def test_ReactionWheel_Power(self):

        compname = 'ReactionWheel_Power'
        inputs = ['w_RW', 'T_RW']
        outputs = ['P_RW']

        self.compare(compname, inputs, outputs)

    def test_ReactionWheel_Torque(self):

        compname = 'ReactionWheel_Torque'
        inputs = ['T_tot']
        outputs = ['T_RW']

        self.compare(compname, inputs, outputs)

    def test_BatterySOC(self):

        compname = 'BatterySOC'
        inputs = ['P_bat', 'temperature']
        outputs = ['SOC']

        self.compare(compname, inputs, outputs)

    def test_BatteryPower(self):

        compname = 'BatteryPower'
        inputs = ['SOC', 'temperature', 'P_bat']
        outputs = ['I_bat']

        self.compare(compname, inputs, outputs)

    def test_BatteryConstraints(self):

        compname = 'BatteryConstraints'
        inputs = ['I_bat', 'SOC']
        outputs = ['ConCh', 'ConDs', 'ConS0', 'ConS1']

        self.compare(compname, inputs, outputs)

if __name__ == "__main__":

    unittest.main()
