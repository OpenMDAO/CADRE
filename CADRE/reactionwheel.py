"""
Reaction wheel discipline for CADRE
"""

from six.moves import range
import numpy as np

from openmdao.api import ExplicitComponent

from CADRE import rk4


class ReactionWheel_Motor(ExplicitComponent):
    """
    Compute reaction wheel motor torque.
    """

    def __init__(self, n):
        super(ReactionWheel_Motor, self).__init__()
        self.n = n

        # Constant
        self.J_RW = 2.8e-5

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('T_RW', np.zeros((3, n)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        self.add_input('w_B', np.zeros((3, n)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        self.add_input('w_RW', np.zeros((3, n)), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        # Outputs
        self.add_output('T_m', np.ones((3, n)), units='N*m',
                        desc='Torque vector of motor over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        T_RW = inputs['T_RW']
        w_B = inputs['w_B']
        w_RW = inputs['w_RW']
        T_m = outputs['T_m']

        w_Bx = np.zeros((3, 3))
        h_RW = self.J_RW * w_RW[:]
        for i in range(0, self.n):
            w_Bx[0, :] = (0., -w_B[2, i], w_B[1, i])
            w_Bx[1, :] = (w_B[2, i], 0., -w_B[0, i])
            w_Bx[2, :] = (-w_B[1, i], w_B[0, i], 0.)

            T_m[:, i] = -T_RW[:, i] - np.dot(w_Bx, h_RW[:, i])

    def _compute_partials(self, inputs):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        # T_RW = inputs['T_RW']
        w_B = inputs['w_B']
        w_RW = inputs['w_RW']

        w_Bx = np.zeros((3, 3))
        self.dT_dTm = np.zeros((self.n, 3, 3))
        self.dT_dwb = np.zeros((self.n, 3, 3))
        self.dT_dh = np.zeros((self.n, 3, 3))

        dwx_dwb = np.zeros((3, 3, 3))
        h_RW = self.J_RW * w_RW[:]
        for i in range(0, self.n):
            w_Bx[0, :] = (0., -w_B[2, i], w_B[1, i])
            w_Bx[1, :] = (w_B[2, i], 0., -w_B[0, i])
            w_Bx[2, :] = (-w_B[1, i], w_B[0, i], 0.)

            dwx_dwb[0, :, 0] = (0., 0., 0.)
            dwx_dwb[1, :, 0] = (0., 0., -1.)
            dwx_dwb[2, :, 0] = (0., 1., 0.)

            dwx_dwb[0, :, 1] = (0., 0., 1.)
            dwx_dwb[1, :, 1] = (0., 0., 0.)
            dwx_dwb[2, :, 1] = (-1., 0., 0.)

            dwx_dwb[0, :, 2] = (0., -1., 0.)
            dwx_dwb[1, :, 2] = (1., 0., 0.)
            dwx_dwb[2, :, 2] = (0., 0., 0.)

            for k in range(0, 3):
                self.dT_dTm[i, k, k] = -1.
                self.dT_dwb[i, :, k] = -np.dot(dwx_dwb[:, :, k], h_RW[:, i])
            self.dT_dh[i, :, :] = -w_Bx

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        self._compute_partials(inputs)
        dT_m = d_outputs['T_m']

        if mode == 'fwd':
            for k in range(3):
                for j in range(3):
                    if 'T_RW' in d_inputs:
                        dT_m[k, :] += self.dT_dTm[:, k, j] * d_inputs['T_RW'][j, :]
                    if 'w_B' in d_inputs:
                        dT_m[k, :] += self.dT_dwb[:, k, j] * d_inputs['w_B'][j, :]
                    if 'w_RW' in d_inputs:
                        dT_m[k, :] += self.dT_dh[:, k, j] * d_inputs['w_RW'][j, :] * self.J_RW
        else:
            for k in range(3):
                for j in range(3):
                    if 'T_RW' in d_inputs:
                        d_inputs['T_RW'][j, :] += self.dT_dTm[:, k, j] * dT_m[k, :]

                    if 'w_B' in d_inputs:
                        d_inputs['w_B'][j, :] += self.dT_dwb[:, k, j] * dT_m[k, :]

                    if 'w_RW' in d_inputs:
                        d_inputs['w_RW'][j, :] += self.dT_dh[:, k, j] * dT_m[k, :] * self.J_RW


class ReactionWheel_Power(ExplicitComponent):
    """
    Compute reaction wheel power.
    """
    # constants
    V = 4.0
    a = 4.9e-4
    b = 4.5e2
    I0 = 0.017

    def __init__(self, n):
        super(ReactionWheel_Power, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('w_RW', np.zeros((3, n)), units='1/s',
                       desc='Angular velocity vector of reaction wheel over time')

        self.add_input('T_RW', np.zeros((3, n)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        # Outputs
        self.add_output('P_RW', np.ones((3, n)), units='W',
                        desc='Reaction wheel power over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        w_RW = inputs['w_RW']
        T_RW = inputs['T_RW']
        P_RW = outputs['P_RW']

        for i in range(self.n):
            for k in range(3):
                P_RW[k, i] = (self.V * (self.a * w_RW[k, i] +
                              self.b * T_RW[k, i])**2 +
                              self.V * self.I0)

    def _compute_partials(self, inputs):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        w_RW = inputs['w_RW']
        T_RW = inputs['T_RW']

        self.dP_dw = np.zeros((self.n, 3))
        self.dP_dT = np.zeros((self.n, 3))
        for i in range(self.n):
            for k in range(3):
                prod = 2 * self.V * (self.a * w_RW[k, i] + self.b * T_RW[k, i])
                self.dP_dw[i, k] = self.a * prod
                self.dP_dT[i, k] = self.b * prod

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        self._compute_partials(inputs)
        dP_RW = d_outputs['P_RW']

        if mode == 'fwd':
            for k in range(3):
                if 'w_RW' in d_inputs:
                    dP_RW[k, :] += self.dP_dw[:, k] * d_inputs['w_RW'][k, :]
                if 'T_RW' in d_inputs:
                    dP_RW[k, :] += self.dP_dT[:, k] * d_inputs['T_RW'][k, :]
        else:
            for k in range(3):
                if 'w_RW' in d_inputs:
                    d_inputs['w_RW'][k, :] += self.dP_dw[:, k] * dP_RW[k, :]
                if 'T_RW' in d_inputs:
                    d_inputs['T_RW'][k, :] += self.dP_dT[:, k] * dP_RW[k, :]


class ReactionWheel_Torque(ExplicitComponent):
    """
    Compute torque vector of reaction wheel.
    """

    def __init__(self, n):
        super(ReactionWheel_Torque, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('T_tot', np.zeros((3, n)), units='N*m',
                       desc='Total reaction wheel torque over time')

        # Outputs
        self.add_output('T_RW', np.zeros((3, n)), units='N*m',
                        desc='Torque vector of reaction wheel over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['T_RW'][:] = inputs['T_tot'][:]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'T_tot' in d_inputs:
                d_outputs['T_RW'][:] += d_inputs['T_tot'][:]
        else:
            if 'T_tot' in d_inputs:
                d_inputs['T_tot'] += d_outputs['T_RW'][:]


class ReactionWheel_Dynamics(rk4.RK4):
    """
    Compute the angular velocity vector of reaction wheel.
    """

    def __init__(self, n_times, h):
        super(ReactionWheel_Dynamics, self).__init__(n_times, h)

        self.n_times = n_times

    def setup(self):
        n_times = self.n_times

        # Inputs
        self.add_input('w_B', np.zeros((3, n_times)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        self.add_input('T_RW', np.zeros((3, n_times)), units='N*m',
                       desc='Torque vector of reaction wheel over time')

        self.add_input('w_RW0', np.zeros((3,)), units='1/s',
                       desc='Initial angular velocity vector of reaction wheel')

        # Outputs
        self.add_output('w_RW', np.zeros((3, n_times)), units='1/s',
                        desc='Angular velocity vector of reaction wheel over time')

        self.options['state_var'] = 'w_RW'
        self.options['init_state_var'] = 'w_RW0'
        self.options['external_vars'] = ['w_B', 'T_RW']

        self.jy = np.zeros((3, 3))

        self.djy_dx = np.zeros((3, 3, 3))
        self.djy_dx[:, :, 0] = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]
        self.djy_dx[:, :, 1] = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
        self.djy_dx[:, :, 2] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]

        # unit conversion of some kind
        self.J_RW = 2.8e-5

    def f_dot(self, external, state):

        self.jy[0, :] = [0., -external[2], external[1]]
        self.jy[1, :] = [external[2], 0., -external[0]]
        self.jy[2, :] = [-external[1], external[0], 0.]

        # TODO: sort out unit conversion here with T_RW
        return (-external[3:]/2.8e-5 - self.jy.dot(state))

    def df_dy(self, external, state):

        self.jy[0, :] = [0., -external[2], external[1]]
        self.jy[1, :] = [external[2], 0., -external[0]]
        self.jy[2, :] = [-external[1], external[0], 0.]
        return -self.jy

    def df_dx(self, external, state):
        self.jx = np.zeros((3, 6))

        for i in range(3):
            self.jx[i, 0:3] = -self.djy_dx[:, :, i].dot(state)
            self.jx[i, i+3] = -1.0 / self.J_RW
        return self.jx
