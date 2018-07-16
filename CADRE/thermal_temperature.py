"""
Thermal discipline for CADRE
"""

from six.moves import range
import numpy as np

from CADRE.rk4 import RK4

# constants
m_f = 0.4
m_b = 2.0
cp_f = 0.6e3
cp_b = 2.0e3
A_T = 2.66e-3

alpha_c = 0.9
alpha_r = 0.2
eps_c = 0.87
eps_r = 0.88

q_sol = 1360.0
K = 5.67051e-8


class ThermalTemperature(RK4):
    """
    Calculates the temperature distribution on the solar panels.
    """
    def __init__(self, n_times, h):
        super(ThermalTemperature, self).__init__(n_times, h)

        self.n_times = n_times

    def setup(self):
        n_times = self.n_times

        # Inputs
        self.add_input('T0', 273.*np.ones((5, )), units='degK',
                       desc='Initial temperatures for the 4 fins and body')

        self.add_input('exposedArea', np.zeros((7, 12, n_times)), units='m**2',
                       desc='Exposed area to the sun for each solar cell over time')

        self.add_input('cellInstd', np.ones((7, 12)), units=None,
                       desc='Cell/Radiator indication')

        self.add_input('LOS', np.zeros((n_times, )), units=None,
                       desc='Satellite to Sun line of sight over time')

        self.add_input('P_comm', np.ones((n_times, )), units='W',
                       desc='Communication power over time')

        # Outputs
        self.add_output('temperature', np.zeros((5, n_times)), units='degK',
                        desc='Temperature for the 4 fins and body over time.',
                        lower=50, upper=400)

        self.options['state_var'] = 'temperature'
        self.options['init_state_var'] = 'T0'
        self.options['external_vars'] = ['exposedArea', 'LOS', 'P_comm']
        self.options['fixed_external_vars'] = ['cellInstd']

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """

        temperature = outputs['temperature']

        # implementation of fixTemps from Thermal_Temperature.f90
        for i in range(0, self.n_times):
            for k in range(0, 5):
                temperature[k, i] = inputs['T0'][k]
                if temperature[k, i] < 0:
                    temperature[k, i] = 0.

        super(ThermalTemperature, self).compute(inputs, outputs)

    def f_dot(self, external, state):

        # revised implementation from ThermalTemperature.f90
        exposedArea = external[:84].reshape(7, 12)
        LOS = external[84]
        P_comm = external[85]
        cellInstd = external[86:].reshape(7, 12)

        fact1 = q_sol*LOS
        fact2 = K*A_T*state**4

        alpha = alpha_c*cellInstd + alpha_r - alpha_r*cellInstd
        s_eps = np.sum(eps_c*cellInstd + eps_r - eps_r*cellInstd, 0)
        s_al_ea = np.sum(alpha*exposedArea, 0)*fact1

        # Panels
        f = np.zeros((5, ))
        for p in range(0, 12):

            # Body
            if p < 4:
                f_i = 4
                m = m_b
                cp = cp_b

            # Fin
            else:
                f_i = (p+1) % 4
                m = m_f
                cp = cp_f

            # Cells
            f[f_i] += (s_al_ea[p] - s_eps[p]*fact2[f_i])/(m*cp)

        f[4] += 4.0 * P_comm / m_b / cp_b

        return f

    def df_dy(self, external, state):
        # revised implementation from ThermalTemperature.f90
        cellInstd = external[86:].reshape(7, 12)
        sum_eps = 4.0*K*A_T*np.sum(eps_c*cellInstd + eps_r - eps_r*cellInstd)

        dfdy = np.zeros((5, 5))

        # Panels
        for p in range(0, 12):

            # Body
            if p < 4:
                f_i = 4
                m = m_b
                cp = cp_b

            # Fin
            else:
                f_i = (p+1) % 4
                m = m_f
                cp = cp_f

            # Cells
            fact = state[f_i]**3/(m*cp)
            dfdy[f_i, f_i] -= sum_eps * fact

        return dfdy

    def df_dx(self, external, state):
        # revised implementation from ThermalTemperature.f90
        exposedArea = external[:84].reshape(7, 12)
        LOS = external[84]
        cellInstd = external[86:].reshape(7, 12)

        dfdx = np.zeros((5, 170))
        alpha = alpha_c*cellInstd + alpha_r - alpha_r*cellInstd
        dalpha_dw = alpha_c - alpha_r
        deps_dw = eps_c - eps_r

        alpha_A_sum = np.sum(alpha*exposedArea, 0)

        # Panels
        for p in range(0, 12):

            # Body
            if p < 4:
                f_i = 4
                m = m_b
                cp = cp_b

            # Fin
            else:
                f_i = (p+1) % 4
                m = m_f
                cp = cp_f

            # Cells
            fact3 = q_sol/(m*cp)
            fact1 = fact3*LOS
            fact2 = K*A_T*state[f_i]**4/(m*cp)

            dfdx[f_i, p:p+84:12] += alpha[:, p] * fact1
            dfdx[f_i, p+86:p+170:12] += dalpha_dw * exposedArea[:, p] * fact1 - \
                deps_dw * fact2
            dfdx[f_i, 84] += alpha_A_sum[p] * fact3

        dfdx[4, 85] += 4.0 / m_b / cp_b

        return dfdx
