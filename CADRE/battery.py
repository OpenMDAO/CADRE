"""
Battery discipline for CADRE
"""

import numpy as np

from openmdao.api import ExplicitComponent
from openmdao.components.ks_comp import KSfunction

from CADRE import rk4


# Constants
sigma = 1e-10
eta = 0.99
Cp = 2900.0*0.001*3600.0
IR = 0.9
T0 = 293.0
alpha = np.log(1/1.1**5)


class BatterySOC(rk4.RK4):
    """
    Computes the time history of the battery state of charge.

    Parameters
    ----------
    n_time: int
        number of time_steps to take

    time_step: float
        size of each timestep
    """

    def __init__(self, n_times, h):
        super(BatterySOC, self).__init__(n_times, h)

        self.n_times = n_times

    def setup(self):
        n_times = self.n_times

        # Inputs
        self.add_input('iSOC', np.zeros((1, )), units=None,
                       desc='Initial state of charge')

        self.add_input('P_bat', np.zeros((n_times, )), units='W',
                       desc='Battery power over time')

        self.add_input('temperature', np.zeros((5, n_times)), units='degK',
                       desc='Battery temperature over time')

        # Outputs
        self.add_output('SOC', np.zeros((1, n_times)), units=None,
                        desc='Battery state of charge over time')

        self.declare_partials('*', '*')

        self.options['state_var'] = 'SOC'
        self.options['init_state_var'] = 'iSOC'
        self.options['external_vars'] = ['P_bat', 'temperature']

    def f_dot(self, external, state):
        """
        Rate of change of SOC
        """
        SOC = state[0]
        P = external[0]
        T = external[5]

        voc = 3 + np.expm1(SOC) / (np.e-1)
        # dVoc_dSOC = np.exp(SOC) / (np.e-1)

        V = IR * voc * (2.0 - np.exp(alpha*(T-T0)/T0))
        I = P/V

        soc_dot = -sigma/24*SOC + eta/Cp*I

        return soc_dot

    def df_dy(self, external, state):
        """
        State derivative
        """
        SOC = state[0]
        P = external[0]
        T = external[5]

        voc = 3 + np.expm1(SOC) / (np.e-1)
        dVoc_dSOC = np.exp(SOC) / (np.e-1)

        tmp = 2 - np.exp(alpha*(T-T0)/T0)
        V = IR * voc * tmp
        # I = P/V

        dV_dSOC = IR * dVoc_dSOC * tmp
        dI_dSOC = -P/V**2 * dV_dSOC

        df_dy = -sigma/24 + eta/Cp*dI_dSOC

        return np.array([[df_dy]])

    def df_dx(self, external, state):
        """
        Output derivative
        """
        SOC = state[0]
        P = external[0]
        T = external[1]

        voc = 3 + np.expm1(SOC) / (np.e-1)
        # dVoc_dSOC = np.exp(SOC) / (np.e-1)

        tmp = 2 - np.exp(alpha*(T-T0)/T0)

        V = IR * voc * tmp
        # I = P/V

        dV_dT = - IR * voc * np.exp(alpha*(T-T0)/T0) * alpha/T0
        dI_dT = - P/V**2 * dV_dT
        dI_dP = 1.0/V

        return np.array([[eta/Cp*dI_dP, 0, 0, 0, 0, eta/Cp*dI_dT]])


class BatteryPower(ExplicitComponent):
    """
    Power supplied by the battery
    """
    def __init__(self, n=2):
        super(BatteryPower, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('SOC', np.zeros((1, n)), units=None,
                       desc='Battery state of charge over time')

        self.add_input('temperature', np.zeros((5, n)), units='degK',
                       desc='Battery temperature over time')

        self.add_input('P_bat', np.zeros((n, )), units='W',
                       desc='Battery power over time')

        # Outputs
        self.add_output('I_bat', np.zeros((n, )), units='A',
                        desc='Battery Current over time')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        SOC = inputs['SOC']
        temperature = inputs['temperature']
        P_bat = inputs['P_bat']

        self.exponential = (2.0 - np.exp(alpha*(temperature[4, :]-T0)/T0))
        self.voc = 3.0 + np.expm1(SOC[0, :]) / (np.e-1)
        self.V = IR * self.voc * self.exponential

        outputs['I_bat'] = P_bat / self.V

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        SOC = inputs['SOC']
        temperature = inputs['temperature']
        P_bat = inputs['P_bat']

        # dI_dP
        dV_dvoc = IR * self.exponential
        dV_dT = - IR * self.voc * np.exp(alpha*(temperature[4, :] - T0)/T0) * alpha / T0
        dVoc_dSOC = np.exp(SOC[0, :]) / (np.e-1)

        self.dI_dP = 1.0 / self.V
        tmp = -P_bat/(self.V**2)
        self.dI_dT = tmp * dV_dT
        self.dI_dSOC = tmp * dV_dvoc * dVoc_dSOC

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """

        dI_bat = d_outputs['I_bat']

        if mode == 'fwd':
            if 'P_bat' in d_inputs:
                dI_bat += self.dI_dP * d_inputs['P_bat']

            if 'temperature' in d_inputs:
                dI_bat += self.dI_dT * d_inputs['temperature'][4, :]

            if 'SOC' in d_inputs:
                dI_bat += self.dI_dSOC * d_inputs['SOC'][0, :]
        else:
            if 'P_bat' in d_inputs:
                d_inputs['P_bat'] += self.dI_dP * dI_bat

            if 'temperature' in d_inputs:
                d_inputs['temperature'][4, :] += self.dI_dT * dI_bat

            if 'SOC' in d_inputs:
                d_inputs['SOC'][0, :] += self.dI_dSOC * dI_bat


class BatteryConstraints(ExplicitComponent):
    """
    Some KS constraints for the battery. I believe this essentially
    replaces a cycle in the graph.
    """

    def __init__(self, n=2):
        super(BatteryConstraints, self).__init__()
        self.n = n

        self.rho = 50
        self.Imin = -10.0
        self.Imax = 5.0
        self.SOC0 = 0.2
        self.SOC1 = 1.0

        self.KS_ch = KSfunction()
        self.KS_ds = KSfunction()
        self.KS_s0 = KSfunction()
        self.KS_s1 = KSfunction()

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('I_bat', np.zeros((n, )), units='A',
                       desc='Battery current over time')

        self.add_input('SOC', np.zeros((1, n)), units=None,
                       desc='Battery state of charge over time')

        # Outputs
        self.add_output('ConCh', 0.0, units='A',
                        desc='Constraint on charging rate')

        self.add_output('ConDs', 0.0, units='A',
                        desc='Constraint on discharging rate')

        self.add_output('ConS0', 0.0, units=None,
                        desc='Constraint on minimum state of charge')

        self.add_output('ConS1', 0.0, units=None,
                        desc='Constraint on maximum state of charge')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        I_bat = inputs['I_bat']
        SOC = inputs['SOC']

        outputs['ConCh'] = self.KS_ch.compute(I_bat - self.Imax, self.rho)
        outputs['ConDs'] = self.KS_ds.compute(self.Imin - I_bat, self.rho)
        outputs['ConS0'] = self.KS_s0.compute(self.SOC0 - SOC, self.rho)
        outputs['ConS1'] = self.KS_s1.compute(SOC - self.SOC1, self.rho)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.dCh_dg, self.dCh_drho = self.KS_ch.derivatives()
        self.dDs_dg, self.dDs_drho = self.KS_ds.derivatives()
        self.dS0_dg, self.dS0_drho = self.KS_s0.derivatives()
        self.dS1_dg, self.dS1_drho = self.KS_s1.derivatives()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
         Matrix-vector product with the Jacobian.
         """
        if mode == 'fwd':
            if 'I_bat' in d_inputs:
                if 'ConCh' in d_outputs:
                    d_outputs['ConCh'] += np.dot(self.dCh_dg, d_inputs['I_bat'])
                if 'ConDs' in d_outputs:
                    d_outputs['ConDs'] -= np.dot(self.dDs_dg, d_inputs['I_bat'])
            if 'SOC' in d_inputs:
                if 'ConS0' in d_outputs:
                    d_outputs['ConS0'] -= np.dot(self.dS0_dg, d_inputs['SOC'][0, :])
                if 'ConS1' in d_outputs:
                    d_outputs['ConS1'] += np.dot(self.dS1_dg, d_inputs['SOC'][0, :])
        else:
            if 'I_bat' in d_inputs:
                dI_bat = d_inputs['I_bat']
                if 'ConCh' in d_outputs:
                    dI_bat += self.dCh_dg * d_outputs['ConCh']
                if 'ConDs' in d_outputs:
                    dI_bat -= self.dDs_dg * d_outputs['ConDs']
                d_inputs['I_bat'] = dI_bat
            if 'SOC' in d_inputs:
                dSOC = d_inputs['SOC']
                if 'ConS0' in d_outputs:
                    dSOC -= self.dS0_dg * d_outputs['ConS0']
                if 'ConS1' in d_outputs:
                    dSOC += self.dS1_dg * d_outputs['ConS1']
                d_inputs['SOC'] = dSOC
