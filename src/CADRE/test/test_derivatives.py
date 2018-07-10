""" Test derivatives by comparing analytic to finite difference."""

import unittest
import warnings
from pprint import pprint

from parameterized import parameterized

import numpy as np

from openmdao.core.problem import Problem
from openmdao.core.indepvarcomp import IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error,assert_check_partials

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
from CADRE.sun import Sun_LOS, Sun_PositionBody, Sun_PositionECI, Sun_PositionSpherical
from CADRE.thermal_temperature import ThermalTemperature

# Ignore the numerical warnings from performing the rel error calc.
warnings.simplefilter("ignore")


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


NTIME = 5


class TestDerivatives(unittest.TestCase):
    @parameterized.expand([(_class.__name__, _class) for _class in component_types],
                          testcase_func_name=lambda f, n, p: 'test_' + p.args[0])
    def test_component(self, name, comp_class):
        # create instance of component type
        try:
            comp = comp_class(NTIME)
        except TypeError:
            try:
                comp = comp_class()
            except TypeError:
                comp = comp_class(NTIME, 300)

        self.assertTrue(isinstance(comp, comp_class),
                        'Could not create instance of %s' % comp_class.__name__)

        # add component to problem
        prob = Problem()
        prob.model.add_subsystem(name, comp, promotes=['*'])

        # do initial setup to get component inputs and outputs
        prob.setup()
        prob.final_setup()

        inputs = [name.split('.')[-1]  # promoted name
                  for name, meta in prob.model.list_inputs(out_stream=None)]
        outputs = [(name.split('.')[-1])  # promoted name
                   for name, meta in prob.model.list_outputs(out_stream=None)]
        print('inputs:', inputs)
        print('outputs:', outputs)

        # create IndepVarComp and add corresponding outputs
        indep = IndepVarComp()
        for var in inputs:
            indep.add_output(var, shape=prob[var].shape)
        prob.model.add_subsystem('indep', indep, promotes=['*'])

        # redo the setup
        prob.setup()
        prob.final_setup()

        # set random input values
        np.random.seed(1001)
        for var in inputs:
            prob[var] = np.random.random(prob[var].shape)*1e-1

        # shape = self.inputs_dict['T_RW']['value'].shape
        # self.prob['T_RW'] = np.random.random(shape) * 1e-1
        # shape = self.inputs_dict['w_RW']['value'].shape
        # self.prob['w_RW'] = np.random.random(shape) * 1e-1
        # self.prob.model.comp.deriv_options['step_calc'] = 'relative'

        # Some components have a time step as a non-differentiable input.
        if 'h' in inputs:
            prob['h'] = 0.01

        # run and check partials
        prob.run_model()

        np.set_printoptions(precision=3, linewidth=256)

        # J = prob.compute_totals(outputs, inputs, return_format='array')
        # pprint(J)
        # J = prob.compute_totals(outputs, inputs, return_format='dict')  #, debug_print=True)
        # assert_rel_error(self, J['temperature']['LOS'][0][0], -6.0, 1e-6)
        # for outp in outputs:
        #     for inp in inputs:
        #         print('\n=================================\n%s wrt %s: %s\n' % (outp, inp, str(J[outp][inp].shape)))
        #         print(J[outp][inp])
        # prob.check_totals(of=outputs, wrt=inputs)

        partials = prob.check_partials(out_stream=None)
        assert_check_partials(partials, atol=1e-3, rtol=1e-3)


# @unittest.skip('deprecated.')
class Testcase_CADRE(unittest.TestCase):

    def setUp(self):
        """ Called before each test. """
        self.prob = Problem()

    def tearDown(self):
        """ Called after each test. """
        self.prob = None

    def setup(self, compname, inputs, state0):
        try:
            comp = eval('%s(NTIME)' % compname)
        except TypeError:
            # At least one comp has no args.
            try:
                comp = eval('%s()' % compname)
            except TypeError:
                comp = eval('%s(NTIME, 300)' % compname)

        self.prob.model.add_subsystem('comp', comp, promotes=['*'])

        # do initial setup to get component inputs
        self.prob.setup()
        self.prob.final_setup()
        self.inputs_dict = {}
        for name, meta in self.prob.model.list_inputs(out_stream=None):
            self.inputs_dict[name.split('.')[-1]] = meta

        for item in inputs + state0:
            pshape = self.inputs_dict[item]['value'].shape
            if pshape == 1:
                initval = 0.0
            else:
                initval = np.zeros((pshape))
            self.prob.model.add_subsystem('p_%s' % item, IndepVarComp(item, initval),
                                          promotes=['*'])

        self.prob.setup(check=False)

        for item in inputs + state0:
            val = self.prob['%s' % item]
            if hasattr(val, 'shape'):
                shape1 = val.shape
                self.prob['%s' % item] = np.random.random(shape1)
            else:
                self.prob['%s' % item] = np.random.random()

    def run_prob(self):
        # Some components have a time step as a non-differentiable input.
        if 'h' in self.inputs_dict:
            self.prob['h'] = 0.01
        self.prob.run_model()

    def compare_derivatives(self, var_in, var_out, rel_error=False):
        partials = self.prob.check_partials()  # compact_print=True)  # out_stream=None)
        assert_check_partials(partials, atol=1e-3, rtol=1e-3)

    def test_Attitude_Angular(self):

        compname = 'Attitude_Angular'
        inputs = ['O_BI', 'Odot_BI']
        outputs = ['w_B']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Attitude_AngularRates(self):

        compname = 'Attitude_AngularRates'
        inputs = ['w_B']
        outputs = ['wdot_B']
        state0 = []

        self.setup(compname, inputs, state0)
        self.prob.model.comp.h = 0.01
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Attitude_Attitude(self):

        compname = 'Attitude_Attitude'
        inputs = ['r_e2b_I']
        outputs = ['O_RI']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Attitude_Roll(self):

        compname = 'Attitude_Roll'
        inputs = ['Gamma']
        outputs = ['O_BR']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Attitude_RotationMtx(self):

        compname = 'Attitude_RotationMtx'
        inputs = ['O_BR', 'O_RI']
        outputs = ['O_BI']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Attitude_RotationMtxRates(self):

        compname = 'Attitude_RotationMtxRates'
        inputs = ['O_BI']
        outputs = ['Odot_BI']
        state0 = []

        self.setup(compname, inputs, state0)
        self.prob.model.comp.h = 0.01
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Attitude_Sideslip(self):

        compname = 'Attitude_Sideslip'
        inputs = ['r_e2b_I', 'O_BI']
        outputs = ['v_e2b_B']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Attitude_Torque(self):

        compname = 'Attitude_Torque'
        inputs = ['w_B', 'wdot_B']
        outputs = ['T_tot']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_BatterySOC(self):

        compname = 'BatterySOC'
        inputs = ['P_bat', 'temperature']
        outputs = ['SOC']
        state0 = ['iSOC']

        self.setup(compname, inputs, state0)
        self.prob.model.comp.h = 0.01
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_BatteryPower(self):

        compname = 'BatteryPower'
        inputs = ['SOC', 'temperature', 'P_bat']
        outputs = ['I_bat']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_BatteryConstraints(self):

        compname = 'BatteryConstraints'
        inputs = ['I_bat', 'SOC']
        outputs = ['ConCh', 'ConDs', 'ConS0', 'ConS1']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_DataDownloaded(self):

        compname = 'Comm_DataDownloaded'
        inputs = ['Dr']
        outputs = ['Data']
        state0 = ['Data0']

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_AntRotation(self):

        compname = 'Comm_AntRotation'
        inputs = ['antAngle']
        outputs = ['q_A']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_AntRotationMtx(self):

        compname = 'Comm_AntRotationMtx'
        inputs = ['q_A']
        outputs = ['O_AB']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_BitRate(self):

        compname = 'Comm_BitRate'
        inputs = ['P_comm', 'gain', 'GSdist', 'CommLOS']
        outputs = ['Dr']
        state0 = []

        self.setup(compname, inputs, state0)

        # These need to be a certain magnitude so it doesn't blow up
        shape = self.inputs_dict['P_comm']['value'].shape
        self.prob['P_comm'] = np.ones(shape)
        shape = self.inputs_dict['GSdist']['value'].shape
        self.prob['GSdist'] = np.random.random(shape) * 1e3

        self.run_prob()

        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_Distance(self):

        compname = 'Comm_Distance'
        inputs = ['r_b2g_A']
        outputs = ['GSdist']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_EarthsSpin(self):

        compname = 'Comm_EarthsSpin'
        inputs = ['t']
        outputs = ['q_E']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_EarthsSpinMtx(self):

        compname = 'Comm_EarthsSpinMtx'
        inputs = ['q_E']
        outputs = ['O_IE']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_GainPattern(self):

        compname = 'Comm_GainPattern'
        inputs = ['azimuthGS', 'elevationGS']
        outputs = ['gain']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_GSposEarth(self):

        compname = 'Comm_GSposEarth'
        inputs = ['lon', 'lat', 'alt']
        outputs = ['r_e2g_E']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_GSposECI(self):

        compname = 'Comm_GSposECI'
        inputs = ['O_IE', 'r_e2g_E']
        outputs = ['r_e2g_I']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_LOS(self):

        compname = 'Comm_LOS'
        inputs = ['r_b2g_I', 'r_e2g_I']
        outputs = ['CommLOS']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_VectorAnt(self):

        compname = 'Comm_VectorAnt'
        inputs = ['r_b2g_B', 'O_AB']
        outputs = ['r_b2g_A']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_VectorBody(self):

        compname = 'Comm_VectorBody'
        inputs = ['r_b2g_I', 'O_BI']
        outputs = ['r_b2g_B']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_VectorECI(self):

        compname = 'Comm_VectorECI'
        inputs = ['r_e2g_I', 'r_e2b_I']
        outputs = ['r_b2g_I']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Comm_VectorSpherical(self):

        compname = 'Comm_VectorSpherical'
        inputs = ['r_b2g_A']
        outputs = ['azimuthGS', 'elevationGS']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Orbit_Dynamics(self):

        compname = 'Orbit_Dynamics'
        inputs = ['r_e2b_I0']
        outputs = ['r_e2b_I']
        state0 = []

        self.setup(compname, inputs, state0)
        self.prob.model.comp.h = 0.01
        self.prob['r_e2b_I0'][:3] = np.random.random((3)) * 1e6
        self.prob['r_e2b_I0'][3:] = np.random.random((3)) * 1e5
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Orbit_Initial(self):

        compname = 'Orbit_Initial'
        inputs = ['altPerigee', 'altApogee', 'RAAN', 'Inc', 'argPerigee', 'trueAnomaly']
        outputs = ['r_e2b_I0']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_bspline_parameters(self):

        inputs = ['CP_P_comm', 'CP_gamma', 'CP_Isetpt']
        outputs = ['P_comm', 'Gamma', 'Isetpt']
        state0 = []

        self.prob.model.add_subsystem('comp', BsplineParameters(NTIME, 5), promotes=['*'])

        # do initial setup to get component inputs
        self.prob.setup()
        self.prob.final_setup()
        self.inputs_dict = {}
        for name, meta in self.prob.model.comp.list_inputs(out_stream=None):
            self.inputs_dict[name.split('.')[-1]] = meta

        for item in inputs:
            pshape = self.inputs_dict[item]['value'].shape
            if pshape == 1:
                initval = 0.0
            else:
                initval = np.zeros((pshape))
            self.prob.model.add_subsystem('p_%s' % item, IndepVarComp(item, initval),
                                          promotes=['*'])

        self.prob.setup(check=False)

        for item in inputs:
            shape = self.inputs_dict[item]['value'].shape
            self.prob[item] = np.random.random(shape)

        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Power_CellVoltage(self):
        # fix
        compname = 'Power_CellVoltage'
        inputs = ['LOS', 'temperature', 'exposedArea', 'Isetpt']
        outputs = ['V_sol']
        state0 = []

        self.setup(compname, inputs, state0)

        shape = self.inputs_dict['temperature']['value'].shape
        self.prob['temperature'] = np.random.random(shape) * 40 + 240

        shape = self.inputs_dict['exposedArea']['value'].shape
        self.prob['exposedArea'] = np.random.random(shape) * 1e-4

        shape = self.inputs_dict['Isetpt']['value'].shape
        self.prob['Isetpt'] = np.random.random(shape) * 1e-2

        self.run_prob()
        self.compare_derivatives(inputs, outputs, rel_error=True)

    def test_Power_SolarPower(self):

        compname = 'Power_SolarPower'
        inputs = ['V_sol', 'Isetpt']
        outputs = ['P_sol']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Power_Total(self):

        compname = 'Power_Total'
        inputs = ['P_sol', 'P_comm', 'P_RW']
        outputs = ['P_bat']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_ReactionWheel_Motor(self):

        compname = 'ReactionWheel_Motor'
        inputs = ['T_RW', 'w_B', 'w_RW']
        outputs = ['T_m']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_ReactionWheel_Power(self):

        compname = 'ReactionWheel_Power'
        inputs = ['w_RW', 'T_RW']
        outputs = ['P_RW']
        state0 = []
        np.random.seed(1001)

        self.setup(compname, inputs, state0)

        shape = self.inputs_dict['T_RW']['value'].shape
        self.prob['T_RW'] = np.random.random(shape) * 1e-1
        shape = self.inputs_dict['w_RW']['value'].shape
        self.prob['w_RW'] = np.random.random(shape) * 1e-1
        # self.prob.model.comp.deriv_options['step_calc'] = 'relative'

        self.run_prob()
        self.compare_derivatives(inputs, outputs, rel_error=True)

    def test_ReactionWheel_Torque(self):

        compname = 'ReactionWheel_Torque'
        inputs = ['T_tot']
        outputs = ['T_RW']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_ReactionWheel_Dynamics(self):

        compname = 'ReactionWheel_Dynamics'
        inputs = ['w_B', 'T_RW']
        outputs = ['w_RW']

        # keep these at zeros
        state0 = []  # ['w_RW0']

        self.setup(compname, inputs, state0)
        self.prob.model.comp.h = 0.01

        shape = self.inputs_dict['w_B']['value'].shape
        self.prob['w_B'] = np.random.random(shape) * 1e-4
        shape = self.inputs_dict['T_RW']['value'].shape
        self.prob['T_RW'] = np.random.random(shape) * 1e-9

        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Solar_ExposedArea(self):

        compname = 'Solar_ExposedArea'
        inputs = ['finAngle', 'azimuth', 'elevation']
        outputs = ['exposedArea']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Sun_LOS(self):

        compname = 'Sun_LOS'
        inputs = ['r_e2b_I', 'r_e2s_I']
        outputs = ['LOS']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Sun_PositionBody(self):

        compname = 'Sun_PositionBody'
        inputs = ['O_BI', 'r_e2s_I']
        outputs = ['r_e2s_B']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Sun_PositionECI(self):

        compname = 'Sun_PositionECI'
        inputs = ['t', 'LD']
        outputs = ['r_e2s_I']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_Sun_PositionSpherical(self):

        compname = 'Sun_PositionSpherical'
        inputs = ['r_e2s_B']
        outputs = ['azimuth', 'elevation']
        state0 = []

        self.setup(compname, inputs, state0)
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)

    def test_ThermalTemperature(self):

        compname = 'ThermalTemperature'
        inputs = ['exposedArea', 'cellInstd', 'LOS', 'P_comm']
        outputs = ['temperature']
        state0 = ['T0']

        self.setup(compname, inputs, state0)
        self.prob.model.comp.h = 0.01
        self.run_prob()
        self.compare_derivatives(inputs+state0, outputs)


if __name__ == "__main__":
    unittest.main()
