"""
Unit test each component in CADRE using some saved data from John's CMF implementation.
"""

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

from CADRE.test.util import load_validation_data

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


#
# load saved data from John's CMF implementation.
#
n, m, h, setd = load_validation_data(idx='5')


class TestComponents(unittest.TestCase):
    @parameterized.expand([(_class.__name__, _class) for _class in component_types],
                          testcase_func_name=lambda f, n, p: 'test_' + p.args[0])
    def test_component(self, name, comp_class):
        # create instance of component type
        try:
            comp = comp_class(n)
        except TypeError:
            try:
                comp = comp_class()
            except TypeError:
                comp = comp_class(n, 300)

        self.assertTrue(isinstance(comp, comp_class),
                        'Could not create instance of %s' % comp_class.__name__)

        comp.h = h  # some components need this

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
