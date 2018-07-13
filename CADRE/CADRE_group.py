"""
Group containing a single CADRE model.
"""

import numpy as np

from openmdao.api import Group, IndepVarComp

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
        indeps = IndepVarComp()
        indeps.add_output('t1', initial_inputs['t1'], units='s')
        indeps.add_output('t2', initial_inputs['t2'], units='s')
        indeps.add_output('t', initial_inputs['t'], units='s')

        # Design parameters
        design = IndepVarComp()
        design.add_output('CP_Isetpt', initial_inputs['CP_Isetpt'], units='A')
        design.add_output('CP_gamma', initial_inputs['CP_gamma'], units='rad')
        design.add_output('CP_P_comm', initial_inputs['CP_P_comm'], units='W')
        design.add_output('iSOC', initial_inputs['iSOC'])

        # These are broadcast inputs in the MDP.
        # mdp_in = IndepVarComp()
        # mdp_in.add_output('lat', 'cellInstd', np.ones((7, 12)))
        # mdp_in.add_output('lon', 'finAngle', np.pi / 4.)
        # mdp_in.add_output('alt', 'antAngle', 0.0)
        # self.add_subsystem('mdp_in', mdp_in, promotes=['*'])

        # params
        params = IndepVarComp()
        params.add_output('LD', initial_inputs['LD'])
        params.add_output('lat', initial_inputs['lat'], units='rad')
        params.add_output('lon', initial_inputs['lon'], units='rad')
        params.add_output('alt', initial_inputs['alt'], units='km')
        params.add_output('r_e2b_I0', initial_inputs['r_e2b_I0'])

        # add independent vars
        self.add_subsystem('indeps', indeps, promotes=['*'])
        self.add_subsystem('design', design, promotes=['*'])
        self.add_subsystem('params', params, promotes=['*'])

        # Add Component Models
        self.add_subsystem('BsplineParameters', BsplineParameters(n, m), promotes=['*'])
        self.add_subsystem('Attitude_Angular', Attitude_Angular(n), promotes=['*'])
        self.add_subsystem('Attitude_AngularRates', Attitude_AngularRates(n, h), promotes=['*'])
        self.add_subsystem('Attitude_Attitude', Attitude_Attitude(n), promotes=['*'])
        self.add_subsystem('Attitude_Roll', Attitude_Roll(n), promotes=['*'])
        self.add_subsystem('Attitude_RotationMtx', Attitude_RotationMtx(n), promotes=['*'])
        self.add_subsystem('Attitude_RotationMtxRates', Attitude_RotationMtxRates(n, h),
                           promotes=['*'])

        # Not needed?
        # self.add_subsystem('Attitude_Sideslip', Attitude_Sideslip(n))

        self.add_subsystem('Attitude_Torque', Attitude_Torque(n), promotes=['*'])
        self.add_subsystem('BatteryConstraints', BatteryConstraints(n), promotes=['*'])
        self.add_subsystem('BatteryPower', BatteryPower(n), promotes=['*'])
        self.add_subsystem('BatterySOC', BatterySOC(n, h), promotes=['*'])
        self.add_subsystem('Comm_AntRotation', Comm_AntRotation(n), promotes=['*'])
        self.add_subsystem('Comm_AntRotationMtx', Comm_AntRotationMtx(n), promotes=['*'])
        self.add_subsystem('Comm_BitRate', Comm_BitRate(n), promotes=['*'])
        self.add_subsystem('Comm_DataDownloaded', Comm_DataDownloaded(n, h), promotes=['*'])
        self.add_subsystem('Comm_Distance', Comm_Distance(n), promotes=['*'])
        self.add_subsystem('Comm_EarthsSpin', Comm_EarthsSpin(n), promotes=['*'])
        self.add_subsystem('Comm_EarthsSpinMtx', Comm_EarthsSpinMtx(n), promotes=['*'])
        self.add_subsystem('Comm_GainPattern', Comm_GainPattern(n, comm_raw), promotes=['*'])
        self.add_subsystem('Comm_GSposEarth', Comm_GSposEarth(n), promotes=['*'])
        self.add_subsystem('Comm_GSposECI', Comm_GSposECI(n), promotes=['*'])
        self.add_subsystem('Comm_LOS', Comm_LOS(n), promotes=['*'])
        self.add_subsystem('Comm_VectorAnt', Comm_VectorAnt(n), promotes=['*'])
        self.add_subsystem('Comm_VectorBody', Comm_VectorBody(n), promotes=['*'])
        self.add_subsystem('Comm_VectorECI', Comm_VectorECI(n), promotes=['*'])
        self.add_subsystem('Comm_VectorSpherical', Comm_VectorSpherical(n), promotes=['*'])

        # Not needed?
        # self.add_subsystem('Orbit_Initial', Orbit_Initial())

        self.add_subsystem('Orbit_Dynamics', Orbit_Dynamics(n, h), promotes=['*'])
        self.add_subsystem('Power_CellVoltage', Power_CellVoltage(n, power_raw),
                           promotes=['*'])
        self.add_subsystem('Power_SolarPower', Power_SolarPower(n), promotes=['*'])
        self.add_subsystem('Power_Total', Power_Total(n), promotes=['*'])

        # Not needed?
        # self.add_subsystem('ReactionWheel_Motor', ReactionWheel_Motor(n))

        self.add_subsystem('ReactionWheel_Power', ReactionWheel_Power(n), promotes=['*'])
        self.add_subsystem('ReactionWheel_Torque', ReactionWheel_Torque(n), promotes=['*'])
        self.add_subsystem('ReactionWheel_Dynamics', ReactionWheel_Dynamics(n, h), promotes=['*'])
        self.add_subsystem('Solar_ExposedArea', Solar_ExposedArea(n, solar_raw1, solar_raw2),
                           promotes=['*'])
        self.add_subsystem('Sun_LOS', Sun_LOS(n), promotes=['*'])
        self.add_subsystem('Sun_PositionBody', Sun_PositionBody(n), promotes=['*'])
        self.add_subsystem('Sun_PositionECI', Sun_PositionECI(n), promotes=['*'])
        self.add_subsystem('Sun_PositionSpherical', Sun_PositionSpherical(n), promotes=['*'])
        self.add_subsystem('ThermalTemperature', ThermalTemperature(n, h), promotes=['*'])

        self.set_order([
            # independent vars
            'indeps',
            'design',
            'params',

            # Component Models
            'BsplineParameters',

            'Orbit_Dynamics',

            'Attitude_Attitude',
            'Attitude_Roll',
            'Attitude_RotationMtx',
            'Attitude_RotationMtxRates',
            'Attitude_Angular',
            'Attitude_AngularRates',
            'Attitude_Torque',

            'ReactionWheel_Torque',
            'ReactionWheel_Dynamics',
            'ReactionWheel_Power',

            'Sun_PositionECI',
            'Sun_PositionBody',
            'Sun_PositionSpherical',
            'Sun_LOS',

            'Solar_ExposedArea',

            'ThermalTemperature',

            'Power_CellVoltage',
            'Power_SolarPower',
            'Power_Total',

            'BatterySOC',
            'BatteryPower',
            'BatteryConstraints',

            'Comm_AntRotation',
            'Comm_AntRotationMtx',
            'Comm_EarthsSpin',
            'Comm_EarthsSpinMtx',
            'Comm_GSposEarth',
            'Comm_GSposECI',
            'Comm_VectorECI',
            'Comm_VectorBody',
            'Comm_VectorAnt',
            'Comm_VectorSpherical',
            'Comm_LOS',
            'Comm_GainPattern',
            'Comm_Distance',
            'Comm_BitRate',
            'Comm_DataDownloaded',
        ])
