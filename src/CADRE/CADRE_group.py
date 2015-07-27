""" Group containing a single CADRE model."""

import numpy as np

from openmdao.components.paramcomp import ParamComp
from openmdao.core.group import Group

from CADRE.attitude import Attitude_Angular, Attitude_AngularRates, Attitude_Attitude, \
     Attitude_Roll, Attitude_RotationMtx, \
     Attitude_RotationMtxRates, Attitude_Sideslip, Attitude_Torque
from CADRE.battery import BatteryConstraints, BatteryPower, BatterySOC
from CADRE.parameters import BsplineParameters
from CADRE.comm import Comm_AntRotation, Comm_AntRotationMtx, Comm_BitRate, \
     Comm_DataDownloaded, Comm_Distance, Comm_EarthsSpin, Comm_EarthsSpinMtx, \
     Comm_GainPattern, Comm_GSposEarth, Comm_GSposECI, Comm_LOS, Comm_VectorAnt, \
     Comm_VectorBody, Comm_VectorECI, Comm_VectorSpherical
from CADRE.orbit import Orbit_Initial, Orbit_Dynamics
from CADRE.reactionwheel import ReactionWheel_Motor, ReactionWheel_Power, \
     ReactionWheel_Torque, ReactionWheel_Dynamics
from CADRE.solar import Solar_ExposedArea
from CADRE.sun import Sun_LOS, Sun_PositionBody, Sun_PositionECI, Sun_PositionSpherical
from CADRE.thermal_temperature import ThermalTemperature
from CADRE.power import Power_CellVoltage, Power_SolarPower, Power_Total

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103


class CADRE(Group):
    """ OpenMDAO implementation of the CADRE model
    """

    def __init__(self, n, m, solar_raw1=None, solar_raw2=None, comm_raw=None,
                 power_raw=None):

        super(CADRE, self).__init__()

        # Analysis parameters
        self.n = n
        self.m = m

        # Some initial setup.
        self.add('p_t1', ParamComp('t1', 0.0), promotes=['*'])
        self.add('p_t2', ParamComp('t2', 43200.0), promotes=['*'])
        h = 43200.0 / (self.n - 1)
        self.add('p_h', ParamComp('h', h), promotes=['*'])
        self.add('p_t', ParamComp('t', np.array(range(0, n))*h), promotes=['*'])

        # Design parameters
        self.add('p_CP_Isetpt', ParamComp('CP_Isetpt',
                                          0.2 * np.ones((12, self.m))),
                 promotes=['*'])
        self.add('p_CP_gamma', ParamComp('CP_gamma',
                                         np.pi/4 * np.ones((self.m, ))),
                 promotes=['*'])
        self.add('p_CP_P_comm', ParamComp('CP_P_comm',
                                          0.1 * np.ones((self.m, ))),
                 promotes=['*'])
        self.add('p_iSOC', ParamComp('iSOC', np.array([0.5])), promotes=['*'])
        self.add('p_cellInstd', ParamComp('cellInstd', np.ones((7, 12))),
                 promotes=['*'])
        self.add('p_finAngle', ParamComp('finAngle', np.pi / 4.), promotes=['*'])
        self.add('p_antAngle', ParamComp('antAngle', 0.0), promotes=['*'])

        # Fixed Station Parameters for the CADRE problem.
        self.add('param_LD', ParamComp('LD', 5000.0), promotes=['*'])
        self.add('param_lat', ParamComp('lat', 42.2708), promotes=['*'])
        self.add('param_lon', ParamComp('lon', -83.7264), promotes=['*'])
        self.add('param_alt', ParamComp('alt', 0.256), promotes=['*'])
        # McMurdo station: -77.85, 166.666667
        # Ann Arbor: 42.2708, -83.7264

        self.add('param_r_e2b_I0', ParamComp('r_e2b_I0', np.zeros((6,))),
                 promotes=['*'])

        # Add Component Models
        self.add("BsplineParameters", BsplineParameters(n, m), promotes=['*'])
        self.add("Attitude_Angular", Attitude_Angular(n), promotes=['*'])
        self.add("Attitude_AngularRates", Attitude_AngularRates(n), promotes=['*'])
        self.add("Attitude_Attitude", Attitude_Attitude(n), promotes=['*'])
        self.add("Attitude_Roll", Attitude_Roll(n), promotes=['*'])
        self.add("Attitude_RotationMtx", Attitude_RotationMtx(n), promotes=['*'])
        self.add("Attitude_RotationMtxRates", Attitude_RotationMtxRates(n),
                 promotes=['*'])

        # Not needed?
        #self.add("Attitude_Sideslip", Attitude_Sideslip(n))

        self.add("Attitude_Torque", Attitude_Torque(n), promotes=['*'])
        self.add("BatteryConstraints", BatteryConstraints(n), promotes=['*'])
        self.add("BatteryPower", BatteryPower(n), promotes=['*'])
        self.add("BatterySOC", BatterySOC(n), promotes=['*'])
        self.add("Comm_AntRotation", Comm_AntRotation(n), promotes=['*'])
        self.add("Comm_AntRotationMtx", Comm_AntRotationMtx(n), promotes=['*'])
        self.add("Comm_BitRate", Comm_BitRate(n), promotes=['*'])
        self.add("Comm_DataDownloaded", Comm_DataDownloaded(n), promotes=['*'])
        self.add("Comm_Distance", Comm_Distance(n), promotes=['*'])
        self.add("Comm_EarthsSpin", Comm_EarthsSpin(n), promotes=['*'])
        self.add("Comm_EarthsSpinMtx", Comm_EarthsSpinMtx(n), promotes=['*'])
        self.add("Comm_GainPattern", Comm_GainPattern(n, comm_raw), promotes=['*'])
        self.add("Comm_GSposEarth", Comm_GSposEarth(n), promotes=['*'])
        self.add("Comm_GSposECI", Comm_GSposECI(n), promotes=['*'])
        self.add("Comm_LOS", Comm_LOS(n), promotes=['*'])
        self.add("Comm_VectorAnt", Comm_VectorAnt(n), promotes=['*'])
        self.add("Comm_VectorBody", Comm_VectorBody(n), promotes=['*'])
        self.add("Comm_VectorECI", Comm_VectorECI(n), promotes=['*'])
        self.add("Comm_VectorSpherical", Comm_VectorSpherical(n), promotes=['*'])

        # Not needed?
        #self.add("Orbit_Initial", Orbit_Initial())

        self.add("Orbit_Dynamics", Orbit_Dynamics(n), promotes=['*'])
        self.add("Power_CellVoltage", Power_CellVoltage(n, power_raw),
                 promotes=['*'])
        self.add("Power_SolarPower", Power_SolarPower(n), promotes=['*'])
        self.add("Power_Total", Power_Total(n), promotes=['*'])

        # Not needed?
        #self.add("ReactionWheel_Motor", ReactionWheel_Motor(n))

        self.add("ReactionWheel_Power", ReactionWheel_Power(n), promotes=['*'])
        self.add("ReactionWheel_Torque", ReactionWheel_Torque(n), promotes=['*'])
        self.add("ReactionWheel_Dynamics", ReactionWheel_Dynamics(n), promotes=['*'])
        self.add("Solar_ExposedArea", Solar_ExposedArea(n, solar_raw1,
                                                        solar_raw2), promotes=['*'])
        self.add("Sun_LOS", Sun_LOS(n), promotes=['*'])
        self.add("Sun_PositionBody", Sun_PositionBody(n), promotes=['*'])
        self.add("Sun_PositionECI", Sun_PositionECI(n), promotes=['*'])
        self.add("Sun_PositionSpherical", Sun_PositionSpherical(n), promotes=['*'])
        self.add("ThermalTemperature", ThermalTemperature(n), promotes=['*'])



