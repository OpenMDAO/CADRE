"""
Communications Discpline for CADRE
"""

import os
from six.moves import range

import numpy as np
import scipy.sparse
from MBI import MBI

from CADRE.explicit import ExplicitComponent

from CADRE.kinematics import fixangles, computepositionspherical, \
    computepositionsphericaljacobian, computepositionrotd,\
    computepositionrotdjacobian

from CADRE import rk4


class Comm_DataDownloaded(rk4.RK4):
    """
    Integrate the incoming data rate to compute the time history of data
    downloaded from the satelite.
    """

    def __init__(self, n_times, h):
        super(Comm_DataDownloaded, self).__init__(n_times, h)

        self.n_times = n_times

    def setup(self):
        n_times = self.n_times

        # Inputs
        self.add_input('Dr', np.zeros(n_times), units='Gibyte/s',
                       desc='Download rate over time')

        # Initial State
        self.add_input('Data0', np.zeros((1, )), units='Gibyte',
                       desc='Initial downloaded data state')

        # States
        self.add_output('Data', np.zeros((1, n_times)), units='Gibyte',
                        desc='Downloaded data state over time')

        self.options['state_var'] = 'Data'
        self.options['init_state_var'] = 'Data0'
        self.options['external_vars'] = ['Dr']

        self.dfdy = np.array([[0.]])
        self.dfdx = np.array([[1.]])

    def f_dot(self, external, state):
        return external[0]

    def df_dy(self, external, state):
        return self.dfdy

    def df_dx(self, external, state):
        return self.dfdx


class Comm_AntRotation(ExplicitComponent):
    """
    Fixed antenna angle to time history of the quaternion.
    """

    def __init__(self, n):
        super(Comm_AntRotation, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('antAngle', 0.0, units='rad')

        # Outputs
        self.add_output('q_A', np.zeros((4, n)), units=None,
                        desc='Quarternion matrix in antenna angle frame over time')

        self.dq_dt = np.zeros(4)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """

        antAngle = inputs['antAngle']
        q_A = outputs['q_A']

        rt2 = np.sqrt(2)
        q_A[0, :] = np.cos(antAngle/2.)
        q_A[1, :] = np.sin(antAngle/2.) / rt2
        q_A[2, :] = - np.sin(antAngle/2.) / rt2
        q_A[3, :] = 0.0

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """

        antAngle = inputs['antAngle']

        rt2 = np.sqrt(2)
        self.dq_dt[0] = - np.sin(antAngle / 2.) / 2.
        self.dq_dt[1] = np.cos(antAngle / 2.) / rt2 / 2.
        self.dq_dt[2] = - np.cos(antAngle / 2.) / rt2 / 2.
        self.dq_dt[3] = 0.0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'antAngle' in d_inputs:
                for k in range(4):
                    d_outputs['q_A'][k, :] += self.dq_dt[k] * d_inputs['antAngle']
        else:
            if 'antAngle' in d_inputs:
                for k in range(4):
                    d_inputs['antAngle'] += self.dq_dt[k] * np.sum(d_outputs['q_A'][k, :])


class Comm_AntRotationMtx(ExplicitComponent):
    """
    Translate antenna angle into the body frame.
    """

    def __init__(self, n):
        super(Comm_AntRotationMtx, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('q_A', np.zeros((4, self.n)),
                       desc='Quarternion matrix in antenna angle frame over time')

        # Outputs
        self.add_output('O_AB', np.zeros((3, 3, self.n)), units=None,
                        desc='Rotation matrix from antenna angle to body-fixed '
                             'frame over time')

        self.J = np.empty((self.n, 3, 3, 4))

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        q_A = inputs['q_A']
        O_AB = outputs['O_AB']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        for i in range(0, self.n):
            A[0, :] = ( q_A[0, i], -q_A[3, i],  q_A[2, i])  # noqa: E201
            A[1, :] = ( q_A[3, i],  q_A[0, i], -q_A[1, i])  # noqa: E201
            A[2, :] = (-q_A[2, i],  q_A[1, i],  q_A[0, i])  # noqa: E201
            A[3, :] = ( q_A[1, i],  q_A[2, i],  q_A[3, i])  # noqa: E201

            B[0, :] = ( q_A[0, i],  q_A[3, i], -q_A[2, i])  # noqa: E201
            B[1, :] = (-q_A[3, i],  q_A[0, i],  q_A[1, i])  # noqa: E201
            B[2, :] = ( q_A[2, i], -q_A[1, i],  q_A[0, i])  # noqa: E201
            B[3, :] = ( q_A[1, i],  q_A[2, i],  q_A[3, i])  # noqa: E201

            O_AB[:, :, i] = np.dot(A.T, B)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        q_A = inputs['q_A']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))
        dA_dq = np.zeros((4, 3, 4))
        dB_dq = np.zeros((4, 3, 4))

        # dA_dq
        dA_dq[0, :, 0] = (1, 0, 0)
        dA_dq[1, :, 0] = (0, 1, 0)
        dA_dq[2, :, 0] = (0, 0, 1)
        dA_dq[3, :, 0] = (0, 0, 0)

        dA_dq[0, :, 1] = (0, 0, 0)
        dA_dq[1, :, 1] = (0, 0, -1)
        dA_dq[2, :, 1] = (0, 1, 0)
        dA_dq[3, :, 1] = (1, 0, 0)

        dA_dq[0, :, 2] = (0, 0, 1)
        dA_dq[1, :, 2] = (0, 0, 0)
        dA_dq[2, :, 2] = (-1, 0, 0)
        dA_dq[3, :, 2] = (0, 1, 0)

        dA_dq[0, :, 3] = (0, -1, 0)
        dA_dq[1, :, 3] = (1, 0, 0)
        dA_dq[2, :, 3] = (0, 0, 0)
        dA_dq[3, :, 3] = (0, 0, 1)

        # dB_dq
        dB_dq[0, :, 0] = (1, 0, 0)
        dB_dq[1, :, 0] = (0, 1, 0)
        dB_dq[2, :, 0] = (0, 0, 1)
        dB_dq[3, :, 0] = (0, 0, 0)

        dB_dq[0, :, 1] = (0, 0, 0)
        dB_dq[1, :, 1] = (0, 0, 1)
        dB_dq[2, :, 1] = (0, -1, 0)
        dB_dq[3, :, 1] = (1, 0, 0)

        dB_dq[0, :, 2] = (0, 0, -1)
        dB_dq[1, :, 2] = (0, 0, 0)
        dB_dq[2, :, 2] = (1, 0, 0)
        dB_dq[3, :, 2] = (0, 1, 0)

        dB_dq[0, :, 3] = (0, 1, 0)
        dB_dq[1, :, 3] = (-1, 0, 0)
        dB_dq[2, :, 3] = (0, 0, 0)
        dB_dq[3, :, 3] = (0, 0, 1)

        for i in range(0, self.n):
            A[0, :] = ( q_A[0, i], -q_A[3, i],  q_A[2, i])  # noqa: E201
            A[1, :] = ( q_A[3, i],  q_A[0, i], -q_A[1, i])  # noqa: E201
            A[2, :] = (-q_A[2, i],  q_A[1, i],  q_A[0, i])  # noqa: E201
            A[3, :] = ( q_A[1, i],  q_A[2, i],  q_A[3, i])  # noqa: E201

            B[0, :] = ( q_A[0, i],  q_A[3, i], -q_A[2, i])  # noqa: E201
            B[1, :] = (-q_A[3, i],  q_A[0, i],  q_A[1, i])  # noqa: E201
            B[2, :] = ( q_A[2, i], -q_A[1, i],  q_A[0, i])  # noqa: E201
            B[3, :] = ( q_A[1, i],  q_A[2, i],  q_A[3, i])  # noqa: E201

            for k in range(0, 4):
                self.J[i, :, :, k] = np.dot(dA_dq[:, :, k].T, B) + \
                    np.dot(A.T, dB_dq[:, :, k])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if 'q_A' in d_inputs:
            dq_A = d_inputs['q_A']
            dO_AB = d_outputs['O_AB']

            if mode == 'fwd':
                for u in range(3):
                    for v in range(3):
                        for k in range(4):
                            dO_AB[u, v, :] += \
                                self.J[:, u, v, k] * dq_A[k, :]
            else:
                for u in range(3):
                    for v in range(3):
                        for k in range(4):
                            dq_A[k, :] += self.J[:, u, v, k] * \
                                dO_AB[u, v, :]


class Comm_BitRate(ExplicitComponent):
    """
    Compute the data rate the satellite receives.
    """

    # constants
    pi = 2 * np.arccos(0.)
    c = 299792458
    Gr = 10 ** (12.9 / 10.)
    Ll = 10 ** (-2.0 / 10.)
    f = 437e6
    k = 1.3806503e-23
    SNR = 10 ** (5.0 / 10.)
    T = 500.
    alpha = c ** 2 * Gr * Ll / 16.0 / pi ** 2 / f ** 2 / k / SNR / T / 1e6

    def __init__(self, n):
        super(Comm_BitRate, self).__init__()
        self.n = n

    def setup(self):
        # Inputs
        self.add_input('P_comm', np.zeros(self.n), units='W',
                       desc='Communication power over time')

        self.add_input('gain', np.zeros(self.n), units=None,
                       desc='Transmitter gain over time')

        self.add_input('GSdist', np.zeros(self.n), units='km',
                       desc='Distance from ground station to satellite over time')

        self.add_input('CommLOS', np.zeros(self.n), units=None,
                       desc='Satellite to ground station line of sight over time')

        # Outputs
        self.add_output('Dr', np.zeros(self.n), units='Gibyte/s',
                        desc='Download rate over time')

    def compute(self, inputs, outputs):
        """
         Calculate outputs.
        """
        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']
        Dr = outputs['Dr']

        for i in range(0, self.n):
            if np.abs(GSdist[i]) > 1e-10:
                S2 = GSdist[i] * 1e3
            else:
                S2 = 1e-10
            Dr[i] = self.alpha * P_comm[i] * gain[i] * \
                CommLOS[i] / S2 ** 2

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']

        S2 = 0.0
        self.dD_dP = np.zeros(self.n)
        self.dD_dGt = np.zeros(self.n)
        self.dD_dS = np.zeros(self.n)
        self.dD_dLOS = np.zeros(self.n)

        for i in range(0, self.n):

            if np.abs(GSdist[i]) > 1e-10:
                S2 = GSdist[i] * 1e3
            else:
                S2 = 1e-10

            self.dD_dP[i] = self.alpha * gain[i] * \
                CommLOS[i] / S2 ** 2
            self.dD_dGt[i] = self.alpha * P_comm[i] * \
                CommLOS[i] / S2 ** 2
            self.dD_dS[i] = -2.0 * 1e3 * self.alpha * P_comm[i] * \
                gain[i] * CommLOS[i] / S2 ** 3
            self.dD_dLOS[i] = self.alpha * \
                P_comm[i] * gain[i] / S2 ** 2

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dDr = d_outputs['Dr']

        if mode == 'fwd':
            if 'P_comm' in d_inputs:
                dDr += self.dD_dP * d_inputs['P_comm']
            if 'gain' in d_inputs:
                dDr += self.dD_dGt * d_inputs['gain']
            if 'GSdist' in d_inputs:
                dDr += self.dD_dS * d_inputs['GSdist']
            if 'CommLOS' in d_inputs:
                dDr += self.dD_dLOS * d_inputs['CommLOS']
        else:
            if 'P_comm' in d_inputs:
                d_inputs['P_comm'] += self.dD_dP.T * dDr
            if 'gain' in d_inputs:
                d_inputs['gain'] += self.dD_dGt.T * dDr
            if 'GSdist' in d_inputs:
                d_inputs['GSdist'] += self.dD_dS.T * dDr
            if 'CommLOS' in d_inputs:
                d_inputs['CommLOS'] += self.dD_dLOS.T * dDr


class Comm_Distance(ExplicitComponent):
    """
    Calculates distance from ground station to satellite.
    """

    def __init__(self, n):
        super(Comm_Distance, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('r_b2g_A', np.zeros((3, self.n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        # Outputs
        self.add_output('GSdist', np.zeros(self.n), units='km',
                        desc='Distance from ground station to satellite over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_b2g_A = inputs['r_b2g_A']
        GSdist = outputs['GSdist']

        for i in range(0, self.n):
            GSdist[i] = np.dot(r_b2g_A[:, i], r_b2g_A[:, i])**0.5

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_b2g_A = inputs['r_b2g_A']

        self.J = np.zeros((self.n, 3))
        for i in range(0, self.n):
            norm = np.dot(r_b2g_A[:, i], r_b2g_A[:, i])**0.5
            if norm > 1e-10:
                self.J[i, :] = r_b2g_A[:, i] / norm
            else:
                self.J[i, :] = 0.

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'r_b2g_A' in d_inputs:
                for k in range(3):
                    d_outputs['GSdist'] += self.J[:, k] * d_inputs['r_b2g_A'][k, :]
        else:
            if 'r_b2g_A' in d_inputs:
                for k in range(3):
                    d_inputs['r_b2g_A'][k, :] += self.J[:, k] * d_outputs['GSdist']


class Comm_EarthsSpin(ExplicitComponent):
    """
    Returns the Earth quaternion as a function of time.
    """

    def __init__(self, n):
        super(Comm_EarthsSpin, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('t', np.zeros(self.n), units='s',
                       desc='Time')

        # Outputs
        self.add_output('q_E', np.zeros((4, self.n)), units=None,
                        desc='Quarternion matrix in Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        t = inputs['t']
        q_E = outputs['q_E']

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        q_E[0, :] = np.cos(theta)
        q_E[3, :] = -np.sin(theta)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        t = inputs['t']

        ntime = self.n
        self.dq_dt = np.zeros((ntime, 4))

        fact = np.pi / 3600.0 / 24.0
        theta = fact * t

        self.dq_dt[:, 0] = -np.sin(theta) * fact
        self.dq_dt[:, 3] = -np.cos(theta) * fact

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 't' in d_inputs:
                for k in range(4):
                    d_outputs['q_E'][k, :] += self.dq_dt[:, k] * d_inputs['t']
        else:
            if 't' in d_inputs:
                for k in range(4):
                    d_inputs['t'] += self.dq_dt[:, k] * d_outputs['q_E'][k, :]


class Comm_EarthsSpinMtx(ExplicitComponent):
    """
    Quaternion to rotation matrix for the earth spin.
    """

    def __init__(self, n):
        super(Comm_EarthsSpinMtx, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('q_E', np.zeros((4, self.n)), units=None,
                       desc='Quarternion matrix in Earth-fixed frame over time')

        # Outputs
        self.add_output('O_IE', np.zeros((3, 3, self.n)), units=None,
                        desc='Rotation matrix from Earth-centered inertial frame to '
                             'Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        q_E = inputs['q_E']
        O_IE = outputs['O_IE']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        for i in range(0, self.n):
            A[0, :] = ( q_E[0, i], -q_E[3, i],  q_E[2, i])  # noqa: E201
            A[1, :] = ( q_E[3, i],  q_E[0, i], -q_E[1, i])  # noqa: E201
            A[2, :] = (-q_E[2, i],  q_E[1, i],  q_E[0, i])  # noqa: E201
            A[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            B[0, :] = ( q_E[0, i],  q_E[3, i], -q_E[2, i])  # noqa: E201
            B[1, :] = (-q_E[3, i],  q_E[0, i],  q_E[1, i])  # noqa: E201
            B[2, :] = ( q_E[2, i], -q_E[1, i],  q_E[0, i])  # noqa: E201
            B[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            O_IE[:, :, i] = np.dot(A.T, B)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        q_E = inputs['q_E']

        A = np.zeros((4, 3))
        B = np.zeros((4, 3))

        dA_dq = np.zeros((4, 3, 4))
        dB_dq = np.zeros((4, 3, 4))

        self.J = np.zeros((self.n, 3, 3, 4))

        # dA_dq
        dA_dq[0, :, 0] = (1, 0, 0)
        dA_dq[1, :, 0] = (0, 1, 0)
        dA_dq[2, :, 0] = (0, 0, 1)
        dA_dq[3, :, 0] = (0, 0, 0)

        dA_dq[0, :, 1] = (0, 0, 0)
        dA_dq[1, :, 1] = (0, 0, -1)
        dA_dq[2, :, 1] = (0, 1, 0)
        dA_dq[3, :, 1] = (1, 0, 0)

        dA_dq[0, :, 2] = (0, 0, 1)
        dA_dq[1, :, 2] = (0, 0, 0)
        dA_dq[2, :, 2] = (-1, 0, 0)
        dA_dq[3, :, 2] = (0, 1, 0)

        dA_dq[0, :, 3] = (0, -1, 0)
        dA_dq[1, :, 3] = (1, 0, 0)
        dA_dq[2, :, 3] = (0, 0, 0)
        dA_dq[3, :, 3] = (0, 0, 1)

        # dB_dq
        dB_dq[0, :, 0] = (1, 0, 0)
        dB_dq[1, :, 0] = (0, 1, 0)
        dB_dq[2, :, 0] = (0, 0, 1)
        dB_dq[3, :, 0] = (0, 0, 0)

        dB_dq[0, :, 1] = (0, 0, 0)
        dB_dq[1, :, 1] = (0, 0, 1)
        dB_dq[2, :, 1] = (0, -1, 0)
        dB_dq[3, :, 1] = (1, 0, 0)

        dB_dq[0, :, 2] = (0, 0, -1)
        dB_dq[1, :, 2] = (0, 0, 0)
        dB_dq[2, :, 2] = (1, 0, 0)
        dB_dq[3, :, 2] = (0, 1, 0)

        dB_dq[0, :, 3] = (0, 1, 0)
        dB_dq[1, :, 3] = (-1, 0, 0)
        dB_dq[2, :, 3] = (0, 0, 0)
        dB_dq[3, :, 3] = (0, 0, 1)

        for i in range(0, self.n):
            A[0, :] = ( q_E[0, i], -q_E[3, i],  q_E[2, i])  # noqa: E201
            A[1, :] = ( q_E[3, i],  q_E[0, i], -q_E[1, i])  # noqa: E201
            A[2, :] = (-q_E[2, i],  q_E[1, i],  q_E[0, i])  # noqa: E201
            A[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            B[0, :] = ( q_E[0, i],  q_E[3, i], -q_E[2, i])  # noqa: E201
            B[1, :] = (-q_E[3, i],  q_E[0, i],  q_E[1, i])  # noqa: E201
            B[2, :] = ( q_E[2, i], -q_E[1, i],  q_E[0, i])  # noqa: E201
            B[3, :] = ( q_E[1, i],  q_E[2, i],  q_E[3, i])  # noqa: E201

            for k in range(0, 4):
                self.J[i, :, :, k] = np.dot(dA_dq[:, :, k].T, B) + \
                    np.dot(A.T, dB_dq[:, :, k])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if 'q_E' in d_inputs:
            dO_IE = d_outputs['O_IE']
            dq_E = d_inputs['q_E']

            if mode == 'fwd':
                for u in range(3):
                    for v in range(3):
                        for k in range(4):
                            dO_IE[u, v, :] += self.J[:, u, v, k] * \
                                dq_E[k, :]
            else:
                for u in range(3):
                    for v in range(3):
                        for k in range(4):
                            dq_E[k, :] += self.J[:, u, v, k] * \
                                dO_IE[u, v, :]


class Comm_GainPattern(ExplicitComponent):
    """
    Determines transmitter gain based on an external az-el map.
    """

    def __init__(self, n, rawG_file=None):
        super(Comm_GainPattern, self).__init__()

        self.n = n

        if not rawG_file:
            fpath = os.path.dirname(os.path.realpath(__file__))
            rawG_file = fpath + '/data/Comm/Gain.txt'

        rawGdata = np.genfromtxt(rawG_file)
        rawG = (10 ** (rawGdata / 10.0)).reshape((361, 361), order='F')

        pi = np.pi
        az = np.linspace(0, 2 * pi, 361)
        el = np.linspace(0, 2 * pi, 361)

        self.MBI = MBI(rawG, [az, el], [15, 15], [4, 4])
        self.x = np.zeros((self.n, 2), order='F')

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('azimuthGS', np.zeros(n), units='rad',
                       desc='Azimuth angle from satellite to ground station in '
                            'Earth-fixed frame over time')

        self.add_input('elevationGS', np.zeros(n), units='rad',
                       desc='Elevation angle from satellite to ground station '
                            'in Earth-fixed frame over time')

        # Outputs
        self.add_output('gain', np.zeros(n), units=None,
                        desc='Transmitter gain over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        result = fixangles(self.n, inputs['azimuthGS'], inputs['elevationGS'])
        self.x[:, 0] = result[0]
        self.x[:, 1] = result[1]
        outputs['gain'] = self.MBI.evaluate(self.x)[:, 0]

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.dg_daz = self.MBI.evaluate(self.x, 1)[:, 0]
        self.dg_del = self.MBI.evaluate(self.x, 2)[:, 0]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dgain = d_outputs['gain']

        if mode == 'fwd':
            if 'azimuthGS' in d_inputs:
                dgain += self.dg_daz * d_inputs['azimuthGS']
            if 'elevationGS' in d_inputs:
                dgain += self.dg_del * d_inputs['elevationGS']
        else:
            if 'azimuthGS' in d_inputs:
                d_inputs['azimuthGS'] += self.dg_daz * dgain
            if 'elevationGS' in d_inputs:
                d_inputs['elevationGS'] += self.dg_del * dgain


class Comm_GSposEarth(ExplicitComponent):
    """
    Returns position of the ground station in Earth frame.
    """

    # Constants
    Re = 6378.137
    d2r = np.pi / 180.

    def __init__(self, n):
        super(Comm_GSposEarth, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('lon', 0.0, units='rad',
                       desc='Longitude of ground station in Earth-fixed frame')
        self.add_input('lat', 0.0, units='rad',
                       desc='Latitude of ground station in Earth-fixed frame')
        self.add_input('alt', 0.0, units='km',
                       desc='Altitude of ground station in Earth-fixed frame')

        # Outputs
        self.add_output('r_e2g_E', np.zeros((3, self.n)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        lat = inputs['lat']
        lon = inputs['lon']
        alt = inputs['alt']
        r_e2g_E = outputs['r_e2g_E']

        cos_lat = np.cos(self.d2r * lat)
        r_GS = (self.Re + alt)

        r_e2g_E[0, :] = r_GS * cos_lat * np.cos(self.d2r*lon)
        r_e2g_E[1, :] = r_GS * cos_lat * np.sin(self.d2r*lon)
        r_e2g_E[2, :] = r_GS * np.sin(self.d2r*lat)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        lat = inputs['lat']
        lon = inputs['lon']
        alt = inputs['alt']

        self.dr_dlon = np.zeros(3)
        self.dr_dlat = np.zeros(3)
        self.dr_dalt = np.zeros(3)

        cos_lat = np.cos(self.d2r * lat)
        sin_lat = np.sin(self.d2r * lat)
        cos_lon = np.cos(self.d2r * lon)
        sin_lon = np.sin(self.d2r * lon)

        r_GS = (self.Re + alt)

        self.dr_dlon[0] = -self.d2r * r_GS * cos_lat * sin_lon
        self.dr_dlat[0] = -self.d2r * r_GS * sin_lat * cos_lon
        self.dr_dalt[0] = cos_lat * cos_lon

        self.dr_dlon[1] = self.d2r * r_GS * cos_lat * cos_lon
        self.dr_dlat[1] = -self.d2r * r_GS * sin_lat * sin_lon
        self.dr_dalt[1] = cos_lat * sin_lon

        self.dr_dlon[2] = 0.
        self.dr_dlat[2] = self.d2r * r_GS * cos_lat
        self.dr_dalt[2] = sin_lat

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_e2g_E = d_outputs['r_e2g_E']

        if mode == 'fwd':
            if 'lon' in d_inputs:
                for k in range(3):
                    dr_e2g_E[k, :] += self.dr_dlon[k] * d_inputs['lon']
            if 'lat' in d_inputs:
                for k in range(3):
                    dr_e2g_E[k, :] += self.dr_dlat[k] * d_inputs['lat']
            if 'alt' in d_inputs:
                for k in range(3):
                    dr_e2g_E[k, :] += self.dr_dalt[k] * d_inputs['alt']
        else:
            for k in range(3):
                if 'lon' in d_inputs:
                    d_inputs['lon'] += self.dr_dlon[k] * np.sum(dr_e2g_E[k, :])
                if 'lat' in d_inputs:
                    d_inputs['lat'] += self.dr_dlat[k] * np.sum(dr_e2g_E[k, :])
                if 'alt' in d_inputs:
                    d_inputs['alt'] += self.dr_dalt[k] * np.sum(dr_e2g_E[k, :])


class Comm_GSposECI(ExplicitComponent):
    """
    Convert time history of ground station position from earth frame
    to inertial frame.
    """
    def __init__(self, n):
        super(Comm_GSposECI, self).__init__()

        self.n = n

    def setup(self):
        # Inputs
        self.add_input('O_IE', np.zeros((3, 3, self.n)), units=None,
                       desc='Rotation matrix from Earth-centered inertial '
                            'frame to Earth-fixed frame over time')

        self.add_input('r_e2g_E', np.zeros((3, self.n)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-fixed frame over time')

        # Outputs
        self.add_output('r_e2g_I', np.zeros((3, self.n)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-centered inertial frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_IE = inputs['O_IE']
        r_e2g_E = inputs['r_e2g_E']
        r_e2g_I = outputs['r_e2g_I']

        for i in range(0, self.n):
            r_e2g_I[:, i] = np.dot(O_IE[:, :, i], r_e2g_E[:, i])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        O_IE = inputs['O_IE']
        r_e2g_E = inputs['r_e2g_E']

        self.J1 = np.zeros((self.n, 3, 3, 3))

        for k in range(0, 3):
            for v in range(0, 3):
                self.J1[:, k, k, v] = r_e2g_E[v, :]

        self.J2 = np.transpose(O_IE, (2, 0, 1))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_e2g_I = d_outputs['r_e2g_I']

        if mode == 'fwd':
            for k in range(3):
                for u in range(3):
                    if 'O_IE' in d_inputs:
                        for v in range(3):
                            dr_e2g_I[k, :] += self.J1[:, k, u, v] * \
                                d_inputs['O_IE'][u, v, :]

                    if 'r_e2g_E' in d_inputs:
                        dr_e2g_I[k, :] += self.J2[:, k, u] * \
                            d_inputs['r_e2g_E'][u, :]
        else:
            for k in range(3):
                if 'O_IE' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            d_inputs['O_IE'][u, v, :] += self.J1[:, k, u, v] * \
                                dr_e2g_I[k, :]
                if 'r_e2g_E' in d_inputs:
                    for j in range(3):
                        d_inputs['r_e2g_E'][j, :] += self.J2[:, k, j] * \
                            dr_e2g_I[k, :]


class Comm_LOS(ExplicitComponent):
    """
    Determines if the Satellite has line of sight with the ground stations.
    """

    # constants
    Re = 6378.137

    def __init__(self, n):
        super(Comm_LOS, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_I', np.zeros((3, n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('r_e2g_I', np.zeros((3, n)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('CommLOS', np.zeros(n), units=None,
                        desc='Satellite to ground station line of sight over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']
        CommLOS = outputs['CommLOS']

        Rb = 100.0
        for i in range(0, self.n):
            proj = np.dot(r_b2g_I[:, i], r_e2g_I[:, i]) / self.Re

            if proj > 0:
                CommLOS[i] = 0.
            elif proj < -Rb:
                CommLOS[i] = 1.
            else:
                x = (proj - 0) / (-Rb - 0)
                CommLOS[i] = 3 * x ** 2 - 2 * x ** 3

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_b2g_I = inputs['r_b2g_I']
        r_e2g_I = inputs['r_e2g_I']

        self.dLOS_drb = np.zeros((self.n, 3))
        self.dLOS_dre = np.zeros((self.n, 3))

        Rb = 10.0
        for i in range(0, self.n):

            proj = np.dot(r_b2g_I[:, i], r_e2g_I[:, i]) / self.Re

            if proj > 0:
                self.dLOS_drb[i, :] = 0.
                self.dLOS_dre[i, :] = 0.
            elif proj < -Rb:
                self.dLOS_drb[i, :] = 0.
                self.dLOS_dre[i, :] = 0.
            else:
                x = (proj - 0) / (-Rb - 0)
                dx_dproj = -1. / Rb
                dLOS_dx = 6 * x - 6 * x ** 2
                dproj_drb = r_e2g_I[:, i]
                dproj_dre = r_b2g_I[:, i]

                self.dLOS_drb[i, :] = dLOS_dx * dx_dproj * dproj_drb
                self.dLOS_dre[i, :] = dLOS_dx * dx_dproj * dproj_dre

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dCommLOS = d_outputs['CommLOS']

        if mode == 'fwd':
            for k in range(3):
                if 'r_b2g_I' in d_inputs:
                    dCommLOS += self.dLOS_drb[:, k] * d_inputs['r_b2g_I'][k, :]
                if 'r_e2g_I' in d_inputs:
                    dCommLOS += self.dLOS_dre[:, k] * d_inputs['r_e2g_I'][k, :]
        else:
            for k in range(3):
                if 'r_b2g_I' in d_inputs:
                    d_inputs['r_b2g_I'][k, :] += self.dLOS_drb[:, k] * dCommLOS
                if 'r_e2g_I' in d_inputs:
                    d_inputs['r_e2g_I'][k, :] += self.dLOS_dre[:, k] * dCommLOS


class Comm_VectorAnt(ExplicitComponent):
    """
    Transform from antenna to body frame
    """
    def __init__(self, n):
        super(Comm_VectorAnt, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_B', np.zeros((3, n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in body-fixed frame over time')

        self.add_input('O_AB', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from antenna angle to body-fixed '
                            'frame over time')

        # Outputs
        self.add_output('r_b2g_A', np.zeros((3, n)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in antenna angle frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_A'] = computepositionrotd(self.n, inputs['r_b2g_B'],
                                                 inputs['O_AB'])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.J1, self.J2 = computepositionrotdjacobian(self.n, inputs['r_b2g_B'],
                                                       inputs['O_AB'])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_b2g_A = d_outputs['r_b2g_A']

        if mode == 'fwd':
            for k in range(3):
                if 'O_AB' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            dr_b2g_A[k, :] += self.J1[:, k, u, v] * \
                                d_inputs['O_AB'][u, v, :]
                if 'r_b2g_B' in d_inputs:
                    for j in range(3):
                        dr_b2g_A[k, :] += self.J2[:, k, j] * \
                            d_inputs['r_b2g_B'][j, :]
        else:
            for k in range(3):
                if 'O_AB' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            d_inputs['O_AB'][u, v, :] += self.J1[:, k, u, v] * \
                                dr_b2g_A[k, :]
                if 'r_b2g_B' in d_inputs:
                    for j in range(3):
                        d_inputs['r_b2g_B'][j, :] += self.J2[:, k, j] * \
                            dr_b2g_A[k, :]


class Comm_VectorBody(ExplicitComponent):
    """
    Transform from body to inertial frame.
    """
    def __init__(self, n):
        super(Comm_VectorBody, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_I', np.zeros((3, n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in Earth-centered inertial frame over time')

        self.add_input('O_BI', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered'
                            'inertial frame over time')

        # Outputs
        self.add_output('r_b2g_B', np.zeros((3, n)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in body-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_b2g_I = inputs['r_b2g_I']
        O_BI = inputs['O_BI']
        r_b2g_B = outputs['r_b2g_B']

        for i in range(0, self.n):
            r_b2g_B[:, i] = np.dot(O_BI[:, :, i], r_b2g_I[:, i])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_b2g_I = inputs['r_b2g_I']
        O_BI = inputs['O_BI']

        self.J1 = np.zeros((self.n, 3, 3, 3))

        for k in range(0, 3):
            for v in range(0, 3):
                self.J1[:, k, k, v] = r_b2g_I[v, :]

        self.J2 = np.transpose(O_BI, (2, 0, 1))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_b2g_B = d_outputs['r_b2g_B']

        if mode == 'fwd':
            for k in range(3):
                if 'O_BI' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            dr_b2g_B[k, :] += self.J1[:, k, u, v] * \
                                d_inputs['O_BI'][u, v, :]
                if 'r_b2g_I' in d_inputs:
                    for j in range(3):
                        dr_b2g_B[k, :] += self.J2[:, k, j] * \
                            d_inputs['r_b2g_I'][j, :]
        else:
            for k in range(3):
                if 'O_BI' in d_inputs:
                    for u in range(3):
                        for v in range(3):
                            d_inputs['O_BI'][u, v, :] += self.J1[:, k, u, v] * \
                                dr_b2g_B[k, :]
                if 'r_b2g_I' in d_inputs:
                    for j in range(3):
                        d_inputs['r_b2g_I'][j, :] += self.J2[:, k, j] * \
                            dr_b2g_B[k, :]


class Comm_VectorECI(ExplicitComponent):
    """
    Determine vector between satellite and ground station.
    """
    def __init__(self, n):
        super(Comm_VectorECI, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2g_I', np.zeros((3, n)), units='km',
                       desc='Position vector from earth to ground station in '
                            'Earth-centered inertial frame over time')

        self.add_input('r_e2b_I', np.zeros((6, n)), units=None,
                       desc='Position and velocity vector from earth to satellite '
                            'in Earth-centered inertial frame over time')

        # Outputs
        self.add_output('r_b2g_I', np.zeros((3, n)), units='km',
                        desc='Position vector from satellite to ground station '
                             'in Earth-centered inertial frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_b2g_I'] = inputs['r_e2g_I'] - inputs['r_e2b_I'][:3, :]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_b2g_I = d_outputs['r_b2g_I']

        if mode == 'fwd':
            if 'r_e2g_I' in d_inputs:
                dr_b2g_I += d_inputs['r_e2g_I']
            if 'r_e2b_I' in d_inputs:
                dr_b2g_I += -d_inputs['r_e2b_I'][:3, :]

        else:
            if 'r_e2g_I' in d_inputs:
                d_inputs['r_e2g_I'] += dr_b2g_I
            if 'r_e2b_I' in d_inputs:
                dr_e2b_I = np.zeros(d_inputs['r_e2b_I'].shape)
                dr_e2b_I[:3, :] += -dr_b2g_I
                d_inputs['r_e2b_I'] += dr_e2b_I


class Comm_VectorSpherical(ExplicitComponent):
    """
    Convert satellite-ground vector into Az-El.
    """
    def __init__(self, n):
        super(Comm_VectorSpherical, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_b2g_A', np.zeros((3, n)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        # Outputs
        self.add_output('azimuthGS', np.zeros(n), units='rad',
                        desc='Azimuth angle from satellite to ground station in '
                             'Earth-fixed frame over time')

        self.add_output('elevationGS', np.zeros(n), units='rad',
                        desc='Elevation angle from satellite to ground station '
                             'in Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        azimuthGS, elevationGS = computepositionspherical(self.n, inputs['r_b2g_A'])
        outputs['azimuthGS'] = azimuthGS
        outputs['elevationGS'] = elevationGS

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.Ja1, self.Ji1, self.Jj1, self.Ja2, self.Ji2, self.Jj2 = \
            computepositionsphericaljacobian(self.n, 3 * self.n, inputs['r_b2g_A'])

        self.J1 = scipy.sparse.csc_matrix((self.Ja1, (self.Ji1, self.Jj1)),
                                          shape=(self.n, 3 * self.n))
        self.J2 = scipy.sparse.csc_matrix((self.Ja2, (self.Ji2, self.Jj2)),
                                          shape=(self.n, 3 * self.n))
        self.J1T = self.J1.transpose()
        self.J2T = self.J2.transpose()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'r_b2g_A' in d_inputs:
                r_b2g_A = d_inputs['r_b2g_A'].reshape((3 * self.n), order='F')
                if 'azimuthGS' in d_outputs:
                    d_outputs['azimuthGS'] += self.J1.dot(r_b2g_A)
                if 'elevationGS' in d_outputs:
                    d_outputs['elevationGS'] += self.J2.dot(r_b2g_A)
        else:
            if 'r_b2g_A' in d_inputs:
                if 'azimuthGS' in d_outputs:
                    az_GS = d_outputs['azimuthGS']
                    d_inputs['r_b2g_A'] += (self.J1T.dot(az_GS)).reshape((3, self.n), order='F')
                if 'elevationGS' in d_outputs:
                    el_GS = d_outputs['elevationGS']
                    d_inputs['r_b2g_A'] += (self.J2T.dot(el_GS)).reshape((3, self.n), order='F')
