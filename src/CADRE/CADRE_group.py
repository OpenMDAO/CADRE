""" Group containing a single CADRE model."""

import numpy as np

from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.core.group import Group

from CADRE.attitude import Attitude_Angular, Attitude_AngularRates, Attitude_Attitude, \
    Attitude_Roll, Attitude_RotationMtx, Attitude_RotationMtxRates, Attitude_Torque
from CADRE.battery import BatteryConstraints, BatteryPower, BatterySOC
from CADRE.parameters import BsplineParameters
from CADRE.comm import Comm_AntRotation, Comm_AntRotationMtx, Comm_BitRate, \
    Comm_DataDownloaded, Comm_Distance, Comm_EarthsSpin, Comm_EarthsSpinMtx, \
    Comm_GainPattern, Comm_GSposEarth, Comm_GSposECI, Comm_LOS, Comm_VectorAnt, \
    Comm_VectorBody, Comm_VectorECI, Comm_VectorSpherical
from CADRE.orbit import Orbit_Dynamics
from CADRE.reactionwheel import ReactionWheel_Power, ReactionWheel_Torque, ReactionWheel_Dynamics
from CADRE.solar import Solar_ExposedArea
from CADRE.sun import Sun_LOS, Sun_PositionBody, Sun_PositionECI, Sun_PositionSpherical
from CADRE.thermal_temperature import ThermalTemperature
from CADRE.power import Power_CellVoltage, Power_SolarPower, Power_Total

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103


class CADRE(Group):
    """
    OpenMDAO implementation of the CADRE model.

    Parameters
    ----------
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

    initial_inputs : dict, optional
        Dictionary of initial values for all parameters. Set the keys you want to
        change.
    """

    def __init__(self, n, m, solar_raw1=None, solar_raw2=None, comm_raw=None,
                 power_raw=None, initial_inputs=None):

        super(CADRE, self).__init__()

        # Analysis parameters
        self.n = n
        self.m = m
        h = 43200.0 / (self.n - 1)

        # User-defined initial parameters
        if initial_inputs is None:
            initial_inputs = {}
        if 't1' not in initial_inputs:
            initial_inputs['t1'] = 0.0
        if 't2' not in initial_inputs:
            initial_inputs['t2'] = 43200.0
        if 't' not in initial_inputs:
            initial_inputs['t'] = np.array(range(0, n))*h
        if 'CP_Isetpt' not in initial_inputs:
            initial_inputs['CP_Isetpt'] = 0.2 * np.ones((12, self.m))
        if 'CP_gamma' not in initial_inputs:
            initial_inputs['CP_gamma'] = np.pi/4 * np.ones((self.m, ))
        if 'CP_P_comm' not in initial_inputs:
            initial_inputs['CP_P_comm'] = 0.1 * np.ones((self.m, ))

        if 'iSOC' not in initial_inputs:
            initial_inputs['iSOC'] = np.array([0.5])

        # Fixed Station Parameters for the CADRE problem.
        # McMurdo station: -77.85, 166.666667
        # Ann Arbor: 42.2708, -83.7264
        if 'LD' not in initial_inputs:
            initial_inputs['LD'] = 5000.0
        if 'lat' not in initial_inputs:
            initial_inputs['lat'] = 42.2708
        if 'lon' not in initial_inputs:
            initial_inputs['lon'] = -83.7264
        if 'alt' not in initial_inputs:
            initial_inputs['alt'] = 0.256

        # Initial Orbital Elements
        if 'r_e2b_I0' not in initial_inputs:
            initial_inputs['r_e2b_I0'] = np.zeros((6, ))

        # Some initial setup.
        self.add_subsystem('p_t1', IndepVarComp('t1', initial_inputs['t1']), promotes=['*'])
        self.add_subsystem('p_t2', IndepVarComp('t2', initial_inputs['t2']), promotes=['*'])
        self.add_subsystem('p_t', IndepVarComp('t', initial_inputs['t']), promotes=['*'])

        # Design parameters
        self.add_subsystem('p_CP_Isetpt', IndepVarComp('CP_Isetpt', initial_inputs['CP_Isetpt']),
                           promotes=['*'])
        self.add_subsystem('p_CP_gamma', IndepVarComp('CP_gamma', initial_inputs['CP_gamma']),
                           promotes=['*'])
        self.add_subsystem('p_CP_P_comm', IndepVarComp('CP_P_comm', initial_inputs['CP_P_comm']),
                           promotes=['*'])
        self.add_subsystem('p_iSOC', IndepVarComp('iSOC', initial_inputs['iSOC']),
                           promotes=['*'])

        # These are broadcast inputs in the MDP.
        # self.add_subsystem('p_cellInstd', IndepVarComp('cellInstd', np.ones((7, 12))),
        #                    promotes=['*'])
        # self.add_subsystem('p_finAngle', IndepVarComp('finAngle', np.pi / 4.), promotes=['*'])
        # self.add_subsystem('p_antAngle', IndepVarComp('antAngle', 0.0), promotes=['*'])

        self.add_subsystem('param_LD', IndepVarComp('LD', initial_inputs['LD']), promotes=['*'])
        self.add_subsystem('param_lat', IndepVarComp('lat', initial_inputs['lat']), promotes=['*'])
        self.add_subsystem('param_lon', IndepVarComp('lon', initial_inputs['lon']), promotes=['*'])
        self.add_subsystem('param_alt', IndepVarComp('alt', initial_inputs['alt']), promotes=['*'])
        self.add_subsystem('param_r_e2b_I0', IndepVarComp('r_e2b_I0', initial_inputs['r_e2b_I0']),
                           promotes=['*'])

        # Add Component Models
        self.add_subsystem("BsplineParameters", BsplineParameters(n, m), promotes=['*'])
        self.add_subsystem("Attitude_Angular", Attitude_Angular(n), promotes=['*'])
        self.add_subsystem("Attitude_AngularRates", Attitude_AngularRates(n, h), promotes=['*'])
        self.add_subsystem("Attitude_Attitude", Attitude_Attitude(n), promotes=['*'])
        self.add_subsystem("Attitude_Roll", Attitude_Roll(n), promotes=['*'])
        self.add_subsystem("Attitude_RotationMtx", Attitude_RotationMtx(n), promotes=['*'])
        self.add_subsystem("Attitude_RotationMtxRates", Attitude_RotationMtxRates(n, h),
                           promotes=['*'])

        # Not needed?
        # self.add_subsystem("Attitude_Sideslip", Attitude_Sideslip(n))

        self.add_subsystem("Attitude_Torque", Attitude_Torque(n), promotes=['*'])
        self.add_subsystem("BatteryConstraints", BatteryConstraints(n), promotes=['*'])
        self.add_subsystem("BatteryPower", BatteryPower(n), promotes=['*'])
        self.add_subsystem("BatterySOC", BatterySOC(n, h), promotes=['*'])
        self.add_subsystem("Comm_AntRotation", Comm_AntRotation(n), promotes=['*'])
        self.add_subsystem("Comm_AntRotationMtx", Comm_AntRotationMtx(n), promotes=['*'])
        self.add_subsystem("Comm_BitRate", Comm_BitRate(n), promotes=['*'])
        self.add_subsystem("Comm_DataDownloaded", Comm_DataDownloaded(n, h), promotes=['*'])
        self.add_subsystem("Comm_Distance", Comm_Distance(n), promotes=['*'])
        self.add_subsystem("Comm_EarthsSpin", Comm_EarthsSpin(n), promotes=['*'])
        self.add_subsystem("Comm_EarthsSpinMtx", Comm_EarthsSpinMtx(n), promotes=['*'])
        self.add_subsystem("Comm_GainPattern", Comm_GainPattern(n, comm_raw), promotes=['*'])
        self.add_subsystem("Comm_GSposEarth", Comm_GSposEarth(n), promotes=['*'])
        self.add_subsystem("Comm_GSposECI", Comm_GSposECI(n), promotes=['*'])
        self.add_subsystem("Comm_LOS", Comm_LOS(n), promotes=['*'])
        self.add_subsystem("Comm_VectorAnt", Comm_VectorAnt(n), promotes=['*'])
        self.add_subsystem("Comm_VectorBody", Comm_VectorBody(n), promotes=['*'])
        self.add_subsystem("Comm_VectorECI", Comm_VectorECI(n), promotes=['*'])
        self.add_subsystem("Comm_VectorSpherical", Comm_VectorSpherical(n), promotes=['*'])

        # Not needed?
        # self.add_subsystem("Orbit_Initial", Orbit_Initial())

        self.add_subsystem("Orbit_Dynamics", Orbit_Dynamics(n, h), promotes=['*'])
        self.add_subsystem("Power_CellVoltage", Power_CellVoltage(n, power_raw),
                           promotes=['*'])
        self.add_subsystem("Power_SolarPower", Power_SolarPower(n), promotes=['*'])
        self.add_subsystem("Power_Total", Power_Total(n), promotes=['*'])

        # Not needed?
        # self.add_subsystem("ReactionWheel_Motor", ReactionWheel_Motor(n))

        self.add_subsystem("ReactionWheel_Power", ReactionWheel_Power(n), promotes=['*'])
        self.add_subsystem("ReactionWheel_Torque", ReactionWheel_Torque(n), promotes=['*'])
        self.add_subsystem("ReactionWheel_Dynamics", ReactionWheel_Dynamics(n, h), promotes=['*'])
        self.add_subsystem("Solar_ExposedArea", Solar_ExposedArea(n, solar_raw1, solar_raw2),
                           promotes=['*'])
        self.add_subsystem("Sun_LOS", Sun_LOS(n), promotes=['*'])
        self.add_subsystem("Sun_PositionBody", Sun_PositionBody(n), promotes=['*'])
        self.add_subsystem("Sun_PositionECI", Sun_PositionECI(n), promotes=['*'])
        self.add_subsystem("Sun_PositionSpherical", Sun_PositionSpherical(n), promotes=['*'])
        self.add_subsystem("ThermalTemperature", ThermalTemperature(n, h), promotes=['*'])
