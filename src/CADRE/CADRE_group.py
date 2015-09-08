""" Group containing a single CADRE model."""

import numpy as np

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
    """ OpenMDAO implementation of the CADRE model.

    Args
    ----
    n : int
        Number of time integration points.

    m : int
        Number of panels in fin discretization.

    solar_raw1 : string
        Name of file containing the angle map of solar exposed area.

    solar_raw2 : string
        Name of file containing the azimuth map of solar exposed area.

    comm_raw : string
        Name of file containing the az-el map of transmitter gain.

    power_raw : string
        Name of file containing a map of solar cell output voltage versus current,
        area, and temperature.

    initial_params : dict, optional
        Dictionary of initial values for all parameters. Set the keys you want to
        change.
    """

    def __init__(self, n, m, solar_raw1=None, solar_raw2=None, comm_raw=None,
                 power_raw=None, initial_params=None)):

        super(CADRE, self).__init__()

        self.ln_solver.options['mode'] = 'auto'

        # Analysis parameters
        self.n = n
        self.m = m
        h = 43200.0 / (self.n - 1)

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
            initial_params['CP_Isetpt'] = 0.2 * np.ones((12, self.m))
        if 'CP_gamma' not in initial_params:
            initial_params['CP_gamma'] = np.pi/4 * np.ones((self.m, ))
        if 'CP_P_comm' not in initial_params:
            initial_params['CP_P_comm'] = 0.1 * np.ones((self.m, ))

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

        # Some initial setup.
        self.add('p_t1', ParamComp('t1', initial_params['t1']), promotes=['*'])
        self.add('p_t2', ParamComp('t2', initial_params['t2']), promotes=['*'])
        self.add('p_t', ParamComp('t', initial_params['t']), promotes=['*'])

        # Design parameters
        self.add('p_CP_Isetpt', ParamComp('CP_Isetpt',
                                          initial_params['CP_Isetpt']),
                 promotes=['*'])
        self.add('p_CP_gamma', ParamComp('CP_gamma',
                                         initial_params['CP_gamma']),
                 promotes=['*'])
        self.add('p_CP_P_comm', ParamComp('CP_P_comm',
                                          initial_params['CP_P_comm']),
                 promotes=['*'])
        self.add('p_iSOC', ParamComp('iSOC', initial_params['iSOC']),
                 promotes=['*'])

        # These are broadcast params in the MDP.
        #self.add('p_cellInstd', ParamComp('cellInstd', np.ones((7, 12))),
        #         promotes=['*'])
        #self.add('p_finAngle', ParamComp('finAngle', np.pi / 4.), promotes=['*'])
        #self.add('p_antAngle', ParamComp('antAngle', 0.0), promotes=['*'])

        self.add('param_LD', ParamComp('LD', initial_params['LD']),
                 promotes=['*'])
        self.add('param_lat', ParamComp('lat', initial_params['lat']),
                 promotes=['*'])
        self.add('param_lon', ParamComp('lon', initial_params['lon']),
                 promotes=['*'])
        self.add('param_alt', ParamComp('alt', initial_params['alt']),
                 promotes=['*'])
        self.add('param_r_e2b_I0', ParamComp('r_e2b_I0',
                                             initial_params['r_e2b_I0']),
                 promotes=['*'])

        # Add Component Models
        self.add("BsplineParameters", BsplineParameters(n, m), promotes=['*'])
        self.add("Attitude_Angular", Attitude_Angular(n), promotes=['*'])
        self.add("Attitude_AngularRates", Attitude_AngularRates(n, h), promotes=['*'])
        self.add("Attitude_Attitude", Attitude_Attitude(n), promotes=['*'])
        self.add("Attitude_Roll", Attitude_Roll(n), promotes=['*'])
        self.add("Attitude_RotationMtx", Attitude_RotationMtx(n), promotes=['*'])
        self.add("Attitude_RotationMtxRates", Attitude_RotationMtxRates(n, h),
                 promotes=['*'])

        # Not needed?
        #self.add("Attitude_Sideslip", Attitude_Sideslip(n))

        self.add("Attitude_Torque", Attitude_Torque(n), promotes=['*'])
        self.add("BatteryConstraints", BatteryConstraints(n), promotes=['*'])
        self.add("BatteryPower", BatteryPower(n), promotes=['*'])
        self.add("BatterySOC", BatterySOC(n, h), promotes=['*'])
        self.add("Comm_AntRotation", Comm_AntRotation(n), promotes=['*'])
        self.add("Comm_AntRotationMtx", Comm_AntRotationMtx(n), promotes=['*'])
        self.add("Comm_BitRate", Comm_BitRate(n), promotes=['*'])
        self.add("Comm_DataDownloaded", Comm_DataDownloaded(n, h), promotes=['*'])
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

        self.add("Orbit_Dynamics", Orbit_Dynamics(n, h), promotes=['*'])
        self.add("Power_CellVoltage", Power_CellVoltage(n, power_raw),
                 promotes=['*'])
        self.add("Power_SolarPower", Power_SolarPower(n), promotes=['*'])
        self.add("Power_Total", Power_Total(n), promotes=['*'])

        # Not needed?
        #self.add("ReactionWheel_Motor", ReactionWheel_Motor(n))

        self.add("ReactionWheel_Power", ReactionWheel_Power(n), promotes=['*'])
        self.add("ReactionWheel_Torque", ReactionWheel_Torque(n), promotes=['*'])
        self.add("ReactionWheel_Dynamics", ReactionWheel_Dynamics(n, h), promotes=['*'])
        self.add("Solar_ExposedArea", Solar_ExposedArea(n, solar_raw1,
                                                        solar_raw2), promotes=['*'])
        self.add("Sun_LOS", Sun_LOS(n), promotes=['*'])
        self.add("Sun_PositionBody", Sun_PositionBody(n), promotes=['*'])
        self.add("Sun_PositionECI", Sun_PositionECI(n), promotes=['*'])
        self.add("Sun_PositionSpherical", Sun_PositionSpherical(n), promotes=['*'])
        self.add("ThermalTemperature", ThermalTemperature(n, h), promotes=['*'])



