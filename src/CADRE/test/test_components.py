"""
Unit test each component in CADRE using some saved data from John's CMF implementation.
"""

import sys
import os
import pickle
import unittest

from parameterized import parameterized

import numpy as np

from openmdao.core.problem import Problem

from CADRE.attitude import Attitude_Angular, Attitude_AngularRates, \
    Attitude_Attitude, Attitude_Roll, Attitude_RotationMtx, \
    Attitude_RotationMtxRates, Attitude_Sideslip, Attitude_Torque
from CADRE.battery import BatterySOC, BatteryPower, BatteryConstraints
from CADRE.comm import Comm_DataDownloaded, Comm_AntRotation, Comm_AntRotationMtx, \
    Comm_BitRate, Comm_Distance, Comm_EarthsSpin, Comm_EarthsSpinMtx, Comm_GainPattern, \
    Comm_GSposEarth, Comm_GSposECI, Comm_LOS, Comm_VectorAnt, Comm_VectorBody, \
    Comm_VectorECI, Comm_VectorSpherical
from CADRE.orbit import Orbit_Dynamics  # , Orbit_Initial
from CADRE.parameters import BsplineParameters
from CADRE.power import Power_CellVoltage, Power_SolarPower, Power_Total
from CADRE.reactionwheel import ReactionWheel_Motor, ReactionWheel_Power, \
    ReactionWheel_Torque, ReactionWheel_Dynamics
from CADRE.solar import Solar_ExposedArea
from CADRE.sun import Sun_LOS, Sun_PositionBody, Sun_PositionECI, Sun_PositionSpherical
from CADRE.thermal_temperature import ThermalTemperature

#
# load data for validation
#
setd = {}
fpath = os.path.dirname(os.path.realpath(__file__))
if sys.version_info.major == 2:
    data = pickle.load(open(fpath + "/data1346_py2.pkl", 'rb'))
else:
    data = pickle.load(open(fpath + "/data1346.pkl", 'rb'))

idx = '5'
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

#
# component types to test
#
component_types = [
    # from CADRE.attitude
    Attitude_Angular, Attitude_AngularRates,
    Attitude_Attitude, Attitude_Roll, Attitude_RotationMtx,
    Attitude_RotationMtxRates, Attitude_Sideslip, Attitude_Torque,
    # from CADRE.battery
    BatterySOC, BatteryPower, BatteryConstraints,
    # from CADRE.comm
    Comm_DataDownloaded, Comm_AntRotation, Comm_AntRotationMtx,
    Comm_BitRate, Comm_Distance, Comm_EarthsSpin, Comm_EarthsSpinMtx, Comm_GainPattern,
    Comm_GSposEarth, Comm_GSposECI, Comm_LOS, Comm_VectorAnt, Comm_VectorBody,
    Comm_VectorECI, Comm_VectorSpherical,
    # from CADRE.orbit
    Orbit_Dynamics,   # Orbit_Initial was not recorded in John's pickle.
    # from CADRE.parameters
    BsplineParameters,
    # from CADRE.power
    Power_CellVoltage, Power_SolarPower, Power_Total,
    # from CADRE.reactionwheel
    ReactionWheel_Motor, ReactionWheel_Power,
    ReactionWheel_Torque, ReactionWheel_Dynamics,
    # from CADRE.solar
    Solar_ExposedArea,
    # from CADRE.sun
    Sun_LOS, Sun_PositionBody, Sun_PositionECI, Sun_PositionSpherical,
    # from CADRE.thermal_temperature
    ThermalTemperature
]


class TestComponents(unittest.TestCase):
    @parameterized.expand([(_class.__name__, _class) for _class in component_types],
                          testcase_func_name=lambda f, n, p: 'test_' + p.args[0])
    def test_component(self, name, comp_class):
        # create instance of component type
        try:
            comp = comp_class(n)
        except TypeError:
            # At least one comp has no args.
            try:
                comp = comp_class()
            except TypeError:
                comp = comp_class(n, 300)

        self.assertTrue(isinstance(comp, comp_class),
                        'Could not create instance of %s' % comp_class.__name__)

        # some components need this
        comp.h = h

        prob = Problem(comp)
        prob.setup()
        prob.final_setup()

        inputs = prob.model.list_inputs(out_stream=None)
        outputs = prob.model.list_outputs(out_stream=None)

        for var, meta in inputs:
            if var in setd:
                prob[var] = setd[var]

        prob.run_model()

        for var, meta in outputs:
            if var in setd:
                tval = setd[var]
                assert(np.linalg.norm(tval - prob[var]) / np.linalg.norm(tval) < 1e-3), \
                    '%s: Expected\n%s\nbut got\n%s' % (var, str(tval), str(prob[var]))


if __name__ == "__main__":
    unittest.main()
