''' Battery discipline for CADRE '''

import numpy as np

from openmdao.core.component import Component

from CADRE import KS
from CADRE import rk4

# Allow non-standard variable names for scientific calc
# pylint: disable-msg=C0103

#Constants
sigma = 1e-10
eta = 0.99
Cp = 2900.0*0.001*3600.0
IR = 0.9
T0 = 293.0
alpha = np.log(1/1.1**5)


class BatterySOC(rk4.RK4):
    """Computes the time history of the battery state of charge.

    n_time: int
        number of time_steps to take

    time_step: float
        size of each timestep
    """

    def __init__(self, n_times, h):
        super(BatterySOC, self).__init__(n_times, h)

        # Inputs
        self.add_input('iSOC', np.zeros((1, )), units="unitless",
                       desc="Initial state of charge")

        self.add_input('P_bat', np.zeros((n_times, )), units="W",
                       desc="Battery power over time")

        self.add_input('temperature', np.zeros((5, n_times )), units="degK",
                       desc="Battery temperature over time")

        # Outputs
        self.add_output('SOC', np.zeros((1, n_times)), units="unitless",
                        desc="Battery state of charge over time")

        self.options['state_var'] = "SOC"
        self.options['init_state_var'] = "iSOC"
        self.options['external_vars'] = ["P_bat", "temperature"]

    def f_dot(self, external, state):
        """Rate of change of SOC"""

        SOC = state[0]
        P = external[0]
        T = external[5]

        voc = 3 + np.expm1(SOC) / (np.e-1)
        #dVoc_dSOC = np.exp(SOC) / (np.e-1)

        V = IR * voc * (2.0 - np.exp(alpha*(T-T0)/T0))
        I = P/V

        soc_dot = -sigma/24*SOC + eta/Cp*I
        return soc_dot

    def df_dy(self, external, state):
        """ State deriative """

        SOC = state[0]
        P = external[0]
        T = external[5]

        voc = 3 + np.expm1(SOC) / (np.e-1)
        dVoc_dSOC = np.exp(SOC) / (np.e-1)

        tmp = 2 - np.exp(alpha*(T-T0)/T0)
        V = IR * voc * tmp
        #I = P/V

        dV_dSOC = IR * dVoc_dSOC * tmp
        dI_dSOC = -P/V**2 * dV_dSOC

        df_dy = -sigma/24 + eta/Cp*dI_dSOC

        return np.array([[df_dy]])

    def df_dx(self, external, state):
        """ Output deriative """

        SOC = state[0]
        P = external[0]
        T = external[1]

        voc = 3 + np.expm1(SOC) / (np.e-1)
        #dVoc_dSOC = np.exp(SOC) / (np.e-1)

        tmp = 2 - np.exp(alpha*(T-T0)/T0)

        V = IR * voc * tmp
        #I = P/V

        dV_dT = - IR * voc * np.exp(alpha*(T-T0)/T0) * alpha/T0
        dI_dT = - P/V**2 * dV_dT
        dI_dP = 1.0/V

        return np.array([[eta/Cp*dI_dP, 0 , 0, 0, 0, eta/Cp*dI_dT]])


class BatteryPower(Component):
    """ Power supplied by the battery"""

    def __init__(self, n=2):
        super(BatteryPower, self).__init__()

        self.n = n

        # Inputs
        self.add_input('SOC', np.zeros((1, n)), units="unitless",
                       desc="Battery state of charge over time")

        self.add_input('temperature', np.zeros((5, n)), units="degK",
                       desc="Battery temperature over time")

        self.add_input('P_bat', np.zeros((n, )), units="W",
                       desc="Battery power over time")

        # Outputs
        self.add_output('I_bat', np.zeros((n, )), units="A",
                        desc="Battery Current over time")

    def solve_nonlinear(self, inputs, outputs, resids):
        """ Calculate output. """

        SOC = inputs['SOC']
        temperature = inputs['temperature']
        P_bat = inputs['P_bat']

        self.exponential = (2.0 - np.exp(alpha*(temperature[4, :]-T0)/T0))
        self.voc = 3.0 + np.expm1(SOC[0, :]) / (np.e-1)
        self.V = IR * self.voc * self.exponential
        outputs['I_bat'] = P_bat/self.V

    def linearize(self, inputs, outputs, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        SOC = inputs['SOC']
        temperature = inputs['temperature']
        P_bat = inputs['P_bat']

        #dI_dP
        dV_dvoc = IR * self.exponential
        dV_dT = - IR * self.voc * np.exp(alpha*(temperature[4, :] -
                                                T0)/T0) * alpha / T0
        dVoc_dSOC = np.exp(SOC[0, :]) / (np.e-1)

        self.dI_dP = 1.0 / self.V
        tmp = -P_bat/(self.V**2)
        self.dI_dT = tmp * dV_dT
        self.dI_dSOC = tmp * dV_dvoc * dVoc_dSOC

    def apply_linear(self, inputs, outputs, dinputs, doutputs, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dI_bat = dresids['I_bat']

        if mode == 'fwd':

            if 'P_bat' in dinputs:
                dI_bat += self.dI_dP * dinputs['P_bat']

            if 'temperature' in dinputs:
                dI_bat += self.dI_dT * dinputs['temperature'][4, :]

            if 'SOC' in dinputs:
                dI_bat += self.dI_dSOC * dinputs['SOC'][0, :]

        else:

            if 'P_bat' in dinputs:
                dinputs['P_bat'] += self.dI_dP * dI_bat

            if 'temperature' in dinputs:
                dinputs['temperature'][4, :] += self.dI_dT * dI_bat

            if 'SOC' in dinputs:
                dinputs['SOC'][0, :] += self.dI_dSOC * dI_bat


class BatteryConstraints(Component):
    """ Some KS constraints for the battery. I believe this essentially
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

        # Inputs
        self.add_input('I_bat', np.zeros((n, )), units="A",
                       desc="Battery current over time")

        self.add_input('SOC', np.zeros((1, n)), units="unitless",
                       desc="Battery state of charge over time")

        # Outputs
        self.add_output('ConCh', 0.0, units="A",
                        desc="Constraint on charging rate")

        self.add_output('ConDs', 0.0, units="A",
                        desc="Constraint on discharging rate")

        self.add_output('ConS0', 0.0, units="unitless",
                        desc="Constraint on minimum state of charge")

        self.add_output('ConS1', 0.0, units="unitless",
                        desc="Constraint on maximum state of charge")

        self.KS_ch = KS.KSfunction()
        self.KS_ds = KS.KSfunction()
        self.KS_s0 = KS.KSfunction()
        self.KS_s1 = KS.KSfunction()

    def solve_nonlinear(self, inputs, outputs, resids):
        """ Calculate output. """

        I_bat = inputs['I_bat']
        SOC = inputs['SOC']

        outputs['ConCh'] = self.KS_ch.compute(I_bat - self.Imax, self.rho)
        outputs['ConDs'] = self.KS_ds.compute(self.Imin - I_bat, self.rho)
        outputs['ConS0'] = self.KS_s0.compute(self.SOC0 - SOC, self.rho)
        outputs['ConS1'] = self.KS_s1.compute(SOC - self.SOC1, self.rho)

    def linearize(self, inputs, outputs, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        self.dCh_dg, self.dCh_drho = self.KS_ch.derivatives()
        self.dDs_dg, self.dDs_drho = self.KS_ds.derivatives()
        self.dS0_dg, self.dS0_drho = self.KS_s0.derivatives()
        self.dS1_dg, self.dS1_drho = self.KS_s1.derivatives()

    def apply_linear(self, inputs, outputs, dinputs, doutputs, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        if mode == 'fwd':
            if 'I_bat' in dinputs:
                if 'ConCh' in dresids:
                    dresids['ConCh'] += np.dot(self.dCh_dg, dinputs['I_bat'])
                if 'ConDs' in dresids:
                    dresids['ConDs'] -= np.dot(self.dDs_dg, dinputs['I_bat'])

            if 'SOC' in dinputs:
                if 'ConS0' in dresids:
                    dresids['ConS0'] -= np.dot(self.dS0_dg, dinputs['SOC'][0, :])
                if 'ConS1' in dresids:
                    dresids['ConS1'] += np.dot(self.dS1_dg, dinputs['SOC'][0, :])

        else:
            if 'I_bat' in dinputs:
                dI_bat = dinputs['I_bat']
                if 'ConCh' in dresids:
                    dI_bat += self.dCh_dg * dresids['ConCh']
                if 'ConDs' in dresids:
                    dI_bat -= self.dDs_dg * dresids['ConDs']
                dinputs['I_bat'] = dI_bat

            if 'SOC' in dinputs:
                dSOC = dinputs['SOC']
                if 'ConS0' in dresids:
                    dSOC -= self.dS0_dg * dresids['ConS0']
                if 'ConS1' in dresids:
                    dSOC += self.dS1_dg * dresids['ConS1']
                dinputs['SOC'] = dSOC
