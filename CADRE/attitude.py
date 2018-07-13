"""
Attitude discipline for CADRE.
"""

from six.moves import range
import numpy as np

from openmdao.api import ExplicitComponent

from CADRE.kinematics import computepositionrotd, computepositionrotdjacobian


class Attitude_Angular(ExplicitComponent):
    """
    Calculates angular velocity vector from the satellite's orientation
    matrix and its derivative.
    """
    def __init__(self, n=2):
        super(Attitude_Angular, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BI', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered '
                       'inertial frame over time')

        self.add_input('Odot_BI', np.zeros((3, 3, n)), units=None,
                       desc='First derivative of O_BI over time')

        # Outputs
        self.add_output('w_B', np.zeros((3, n)), units='1/s',
                        desc='Angular velocity vector in body-fixed frame over time')

        self.dw_dOdot = np.zeros((n, 3, 3, 3))
        self.dw_dO = np.zeros((n, 3, 3, 3))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_BI = inputs['O_BI']
        Odot_BI = inputs['Odot_BI']
        w_B = outputs['w_B']

        for i in range(0, self.n):
            w_B[0, i] = np.dot(Odot_BI[2, :, i], O_BI[1, :, i])
            w_B[1, i] = np.dot(Odot_BI[0, :, i], O_BI[2, :, i])
            w_B[2, i] = np.dot(Odot_BI[1, :, i], O_BI[0, :, i])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        O_BI = inputs['O_BI']
        Odot_BI = inputs['Odot_BI']

        for i in range(0, self.n):
            self.dw_dOdot[i, 0, 2, :] = O_BI[1, :, i]
            self.dw_dO[i, 0, 1, :] = Odot_BI[2, :, i]

            self.dw_dOdot[i, 1, 0, :] = O_BI[2, :, i]
            self.dw_dO[i, 1, 2, :] = Odot_BI[0, :, i]

            self.dw_dOdot[i, 2, 1, :] = O_BI[0, :, i]
            self.dw_dO[i, 2, 0, :] = Odot_BI[1, :, i]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dw_B = d_outputs['w_B']

        if mode == 'fwd':
            for k in range(3):
                for i in range(3):
                    for j in range(3):
                        if 'O_BI' in d_inputs:
                            dw_B[k, :] += self.dw_dO[:, k, i, j] * \
                                d_inputs['O_BI'][i, j, :]
                        if 'Odot_BI' in d_inputs:
                            dw_B[k, :] += self.dw_dOdot[:, k, i, j] * \
                                d_inputs['Odot_BI'][i, j, :]
        else:
            for k in range(3):
                for i in range(3):
                    for j in range(3):
                        if 'O_BI' in d_inputs:
                            d_inputs['O_BI'][i, j, :] += self.dw_dO[:, k, i, j] * \
                                dw_B[k, :]
                        if 'Odot_BI' in d_inputs:
                            d_inputs['Odot_BI'][i, j, :] += self.dw_dOdot[:, k, i, j] * \
                                dw_B[k, :]


class Attitude_AngularRates(ExplicitComponent):
    """
    Calculates time derivative of angular velocity vector.
    """

    def __init__(self, n=2, h=28.8):
        super(Attitude_AngularRates, self).__init__()

        self.n = n
        self.h = h

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('w_B', np.zeros((3, n)), units='1/s',
                       desc='Angular velocity vector in body-fixed frame over time')

        # Outputs
        self.add_output('wdot_B', np.zeros((3, n)), units='1/s**2',
                        desc='Time derivative of w_B over time')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        w_B = inputs['w_B']
        h = self.h
        wdot_B = outputs['wdot_B']

        wdot_B[:, 0] = w_B[:, 1] - w_B[:, 0]
        wdot_B[:, 1:-1] = (w_B[:, 2:] - w_B[:, :-2]) / 2.0
        wdot_B[:, -1] = w_B[:, -1] - w_B[:, -2]
        wdot_B *= 1.0/h

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        h = self.h

        dwdot_B = d_outputs['wdot_B']

        if mode == 'fwd':
            if 'w_B' in d_inputs:
                dwdot_B[:, 0] += d_inputs['w_B'][:, 1] / h
                dwdot_B[:, 0] -= d_inputs['w_B'][:, 0] / h
                dwdot_B[:, 1:-1] += d_inputs['w_B'][:, 2:] / 2.0 / h
                dwdot_B[:, 1:-1] -= d_inputs['w_B'][:, :-2] / 2.0 / h
                dwdot_B[:, -1] += d_inputs['w_B'][:, -1] / h
                dwdot_B[:, -1] -= d_inputs['w_B'][:, -2] / h
        else:
            if 'w_B' in d_inputs:
                w_B = np.zeros(d_inputs['w_B'].shape)
                w_B[:, 1] += dwdot_B[:, 0] / h
                w_B[:, 0] -= dwdot_B[:, 0] / h
                w_B[:, 2:] += dwdot_B[:, 1:-1] / 2.0 / h
                w_B[:, :-2] -= dwdot_B[:, 1:-1] / 2.0 / h
                w_B[:, -1] += dwdot_B[:, -1] / h
                w_B[:, -2] -= dwdot_B[:, -1] / h
                d_inputs['w_B'] += w_B


class Attitude_Attitude(ExplicitComponent):
    """
    Coordinate transformation from the interial plane to the rolled
    (forward facing) plane.
    """

    dvx_dv = np.zeros((3, 3, 3))
    dvx_dv[0, :, 0] = (0., 0., 0.)
    dvx_dv[1, :, 0] = (0., 0., -1.)
    dvx_dv[2, :, 0] = (0., 1., 0.)

    dvx_dv[0, :, 1] = (0., 0., 1.)
    dvx_dv[1, :, 1] = (0., 0., 0.)
    dvx_dv[2, :, 1] = (-1., 0., 0.)

    dvx_dv[0, :, 2] = (0., -1., 0.)
    dvx_dv[1, :, 2] = (1., 0., 0.)
    dvx_dv[2, :, 2] = (0., 0., 0.)

    def __init__(self, n=2):
        super(Attitude_Attitude, self).__init__()

        self.n = n

        self.dO_dr = np.zeros((n, 3, 3, 6))

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2b_I', np.zeros((6, n)), units=None,
                       desc='Position and velocity vector from earth to satellite in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('O_RI', np.zeros((3, 3, n)), units=None,
                        desc='Rotation matrix from rolled body-fixed frame to '
                             'Earth-centered inertial frame over time')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_e2b_I = inputs['r_e2b_I']
        O_RI = outputs['O_RI']

        O_RI[:] = np.zeros(O_RI.shape)
        for i in range(0, self.n):

            r = r_e2b_I[0:3, i]
            v = r_e2b_I[3:, i]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)
            jB = -np.dot(vx, iB)

            O_RI[0, :, i] = iB
            O_RI[1, :, i] = jB
            O_RI[2, :, i] = -v

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2b_I = inputs['r_e2b_I']

        diB_dv = np.zeros((3, 3))
        djB_dv = np.zeros((3, 3))

        for i in range(0, self.n):

            r = r_e2b_I[0:3, i]
            v = r_e2b_I[3:, i]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            dr_dr = np.zeros((3, 3))
            dv_dv = np.zeros((3, 3))

            for k in range(0, 3):
                dr_dr[k, k] += 1.0 / normr
                dv_dv[k, k] += 1.0 / normv
                dr_dr[:, k] -= r_e2b_I[
                    0:3, i] * r_e2b_I[k, i] / normr ** 3
                dv_dv[:, k] -= r_e2b_I[
                    3:, i] * r_e2b_I[3 + k, i] / normv ** 3

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)

            diB_dr = vx
            diB_dv[:, 0] = np.dot(self.dvx_dv[:, :, 0], r)
            diB_dv[:, 1] = np.dot(self.dvx_dv[:, :, 1], r)
            diB_dv[:, 2] = np.dot(self.dvx_dv[:, :, 2], r)

            djB_diB = -vx
            djB_dv[:, 0] = -np.dot(self.dvx_dv[:, :, 0], iB)
            djB_dv[:, 1] = -np.dot(self.dvx_dv[:, :, 1], iB)
            djB_dv[:, 2] = -np.dot(self.dvx_dv[:, :, 2], iB)

            self.dO_dr[i, 0, :, 0:3] = np.dot(diB_dr, dr_dr)
            self.dO_dr[i, 0, :, 3:] = np.dot(diB_dv, dv_dv)

            self.dO_dr[i, 1, :, 0:3] = np.dot(np.dot(djB_diB, diB_dr), dr_dr)
            self.dO_dr[i, 1, :, 3:] = np.dot(np.dot(djB_diB, diB_dv) + djB_dv,
                                             dv_dv)

            self.dO_dr[i, 2, :, 3:] = -dv_dv

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dO_RI = d_outputs['O_RI']

        if mode == 'fwd':
            if 'r_e2b_I' in d_inputs:
                for k in range(3):
                    for j in range(3):
                        for i in range(6):
                            dO_RI[k, j, :] += self.dO_dr[:, k, j, i] * \
                                d_inputs['r_e2b_I'][i, :]
        else:
            if 'r_e2b_I' in d_inputs:
                for k in range(3):
                    for j in range(3):
                        for i in range(6):
                            d_inputs['r_e2b_I'][i, :] += self.dO_dr[:, k, j, i] * \
                                dO_RI[k, j, :]


class Attitude_Roll(ExplicitComponent):
    """
    Calculates the body-fixed orientation matrix.
    """
    def __init__(self, n=2):
        super(Attitude_Roll, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('Gamma', np.zeros(n), units='rad',
                       desc='Satellite roll angle over time')

        # Outputs
        self.add_output('O_BR', np.zeros((3, 3, n)), units=None,
                        desc='Rotation matrix from body-fixed frame to rolled '
                        'body-fixed frame over time')

        self.dO_dg = np.zeros((n, 3, 3))

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        Gamma = inputs['Gamma']
        O_BR = outputs['O_BR']

        O_BR[:] = np.zeros((3, 3, self.n))
        O_BR[0, 0, :] = np.cos(Gamma)
        O_BR[0, 1, :] = np.sin(Gamma)
        O_BR[1, 0, :] = -O_BR[0, 1, :]
        O_BR[1, 1, :] = O_BR[0, 0, :]
        O_BR[2, 2, :] = np.ones(self.n)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        Gamma = inputs['Gamma']

        self.dO_dg = np.zeros((self.n, 3, 3))
        self.dO_dg[:, 0, 0] = -np.sin(Gamma)
        self.dO_dg[:, 0, 1] = np.cos(Gamma)
        self.dO_dg[:, 1, 0] = -self.dO_dg[:, 0, 1]
        self.dO_dg[:, 1, 1] = self.dO_dg[:, 0, 0]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dO_BR = d_outputs['O_BR']

        if mode == 'fwd':
            if 'Gamma' in d_inputs:
                for k in range(3):
                    for j in range(3):
                        dO_BR[k, j, :] += self.dO_dg[:, k, j] * \
                            d_inputs['Gamma']
        else:
            if 'Gamma' in d_inputs:
                for k in range(3):
                    for j in range(3):
                        d_inputs['Gamma'] += self.dO_dg[:, k, j] * \
                            dO_BR[k, j, :]


class Attitude_RotationMtx(ExplicitComponent):
    """
    Multiplies transformations to produce the orientation matrix of the
    body frame with respect to inertial.
    """

    def __init__(self, n=2):
        super(Attitude_RotationMtx, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BR', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from body-fixed frame to rolled '
                       'body-fixed frame over time')

        self.add_input('O_RI', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from rolled body-fixed '
                       'frame to Earth-centered inertial frame over time')

        # Outputs
        self.add_output('O_BI', np.zeros((3, 3, n)), units=None,
                        desc='Rotation matrix from body-fixed frame to '
                        'Earth-centered inertial frame over time')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_BR = inputs['O_BR']
        O_RI = inputs['O_RI']
        O_BI = outputs['O_BI']

        for i in range(0, self.n):
            O_BI[:, :, i] = np.dot(O_BR[:, :, i], O_RI[:, :, i])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dO_BI = d_outputs['O_BI']
        O_BR = inputs['O_BR']
        O_RI = inputs['O_RI']

        if mode == 'fwd':
            for u in range(3):
                for v in range(3):
                    for k in range(3):
                        if 'O_RI' in d_inputs:
                            dO_BI[u, v, :] += O_BR[u, k, :] * \
                                d_inputs['O_RI'][k, v, :]
                        if 'O_BR' in d_inputs:
                            dO_BI[u, v, :] += d_inputs['O_BR'][u, k, :] * \
                                O_RI[k, v, :]
        else:
            for u in range(3):
                for v in range(3):
                    for k in range(3):
                        if 'O_RI' in d_inputs:
                            d_inputs['O_RI'][k, v, :] += O_BR[u, k, :] * \
                                dO_BI[u, v, :]
                        if 'O_BR' in d_inputs:
                            d_inputs['O_BR'][u, k, :] += dO_BI[u, v, :] * \
                                O_RI[k, v, :]


class Attitude_RotationMtxRates(ExplicitComponent):
    """
    Calculates time derivative of body frame orientation matrix.
    """

    def __init__(self, n=2, h=28.2):
        super(Attitude_RotationMtxRates, self).__init__()

        self.n = n
        self.h = h

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BI', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from body-fixed frame to Earth-centered '
                            'inertial frame over time')

        # Outputs
        self.add_output('Odot_BI', np.zeros((3, 3, n)), units=None,
                        desc='First derivative of O_BI over time')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        O_BI = inputs['O_BI']
        h = self.h
        Odot_BI = outputs['Odot_BI']

        Odot_BI[:, :, 0] = O_BI[:, :, 1]
        Odot_BI[:, :, 0] -= O_BI[:, :, 0]
        Odot_BI[:, :, 1:-1] = O_BI[:, :, 2:] / 2.0
        Odot_BI[:, :, 1:-1] -= O_BI[:, :, :-2] / 2.0
        Odot_BI[:, :, -1] = O_BI[:, :, -1]
        Odot_BI[:, :, -1] -= O_BI[:, :, -2]
        Odot_BI *= 1.0/h

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dOdot_BI = d_outputs['Odot_BI']

        h = self.h

        if mode == 'fwd':
            if 'O_BI' in d_inputs:
                dO_BI = d_inputs['O_BI']
                dOdot_BI[:, :, 0] += dO_BI[:, :, 1] / h
                dOdot_BI[:, :, 0] -= dO_BI[:, :, 0] / h
                dOdot_BI[:, :, 1:-1] += dO_BI[:, :, 2:] / (2.0*h)
                dOdot_BI[:, :, 1:-1] -= dO_BI[:, :, :-2] / (2.0*h)
                dOdot_BI[:, :, -1] += dO_BI[:, :, -1] / h
                dOdot_BI[:, :, -1] -= dO_BI[:, :, -2] / h
        else:
            if 'O_BI' in d_inputs:
                dO_BI = np.zeros(d_inputs['O_BI'].shape)
                dO_BI[:, :, 1] += dOdot_BI[:, :, 0] / h
                dO_BI[:, :, 0] -= dOdot_BI[:, :, 0] / h
                dO_BI[:, :, 2:] += dOdot_BI[:, :, 1:-1] / (2.0*h)
                dO_BI[:, :, :-2] -= dOdot_BI[:, :, 1:-1] / (2.0*h)
                dO_BI[:, :, -1] += dOdot_BI[:, :, -1] / h
                dO_BI[:, :, -2] -= dOdot_BI[:, :, -1] / h
                d_inputs['O_BI'] += dO_BI


class Attitude_Sideslip(ExplicitComponent):
    """
    Determine velocity in the body frame.
    """

    def __init__(self, n=2):
        super(Attitude_Sideslip, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2b_I', np.zeros((6, n)), units=None,
                       desc='Position and velocity vector from earth to satellite '
                       'in Earth-centered inertial frame over time')

        self.add_input('O_BI', np.zeros((3, 3, n)), units=None,
                       desc='Rotation matrix from body-fixed frame to '
                       'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('v_e2b_B', np.zeros((3, n)), units='m/s',
                        desc='Velocity vector from earth to satellite'
                        'in body-fixed frame over time')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_e2b_I = inputs['r_e2b_I']
        O_BI = inputs['O_BI']
        v_e2b_B = outputs['v_e2b_B']

        v_e2b_B[:] = computepositionrotd(self.n, r_e2b_I[3:, :], O_BI)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2b_I = inputs['r_e2b_I']
        O_BI = inputs['O_BI']

        self.J1, self.J2 = computepositionrotdjacobian(self.n,
                                                       r_e2b_I[3:, :],
                                                       O_BI)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dv_e2b_B = d_outputs['v_e2b_B']

        if mode == 'fwd':
            for k in range(3):
                if 'O_BI' in inputs:
                    for u in range(3):
                        for v in range(3):
                            dv_e2b_B[k, :] += self.J1[:, k, u, v] * \
                                d_inputs['O_BI'][u, v, :]
                if 'r_e2b_I' in inputs:
                    for j in range(3):
                        dv_e2b_B[k, :] += self.J2[:, k, j] * \
                            d_inputs['r_e2b_I'][3+j, :]

        else:
            for k in range(3):
                if 'O_BI' in inputs:
                    for u in range(3):
                        for v in range(3):
                            d_inputs['O_BI'][u, v, :] += self.J1[:, k, u, v] * \
                                dv_e2b_B[k, :]
                if 'r_e2b_I' in inputs:
                    for j in range(3):
                        d_inputs['r_e2b_I'][3+j, :] += self.J2[:, k, j] * \
                            dv_e2b_B[k, :]


class Attitude_Torque(ExplicitComponent):
    """
    Compute the required reaction wheel tourque.
    """

    J = np.zeros((3, 3))
    J[0, :] = (0.018, 0., 0.)
    J[1, :] = (0., 0.018, 0.)
    J[2, :] = (0., 0., 0.006)

    def __init__(self, n=2):
        super(Attitude_Torque, self).__init__()

        self.n = n

        self.dT_dwdot = np.zeros((n, 3, 3))
        self.dwx_dw = np.zeros((3, 3, 3))

        self.dwx_dw[0, :, 0] = (0., 0., 0.)
        self.dwx_dw[1, :, 0] = (0., 0., -1.)
        self.dwx_dw[2, :, 0] = (0., 1., 0.)

        self.dwx_dw[0, :, 1] = (0., 0., 1.)
        self.dwx_dw[1, :, 1] = (0., 0., 0.)
        self.dwx_dw[2, :, 1] = (-1., 0, 0.)

        self.dwx_dw[0, :, 2] = (0., -1., 0)
        self.dwx_dw[1, :, 2] = (1., 0., 0.)
        self.dwx_dw[2, :, 2] = (0., 0., 0.)

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('w_B', np.zeros((3, n)), units='1/s',
                       desc='Angular velocity in body-fixed frame over time')

        self.add_input('wdot_B', np.zeros((3, n)), units='1/s**2',
                       desc='Time derivative of w_B over time')

        # Outputs
        self.add_output('T_tot', np.zeros((3, n)), units='N*m',
                        desc='Total reaction wheel torque over time')

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        w_B = inputs['w_B']
        wdot_B = inputs['wdot_B']
        T_tot = outputs['T_tot']

        wx = np.zeros((3, 3))
        for i in range(0, self.n):
            wx[0, :] = (0., -w_B[2, i], w_B[1, i])
            wx[1, :] = (w_B[2, i], 0., -w_B[0, i])
            wx[2, :] = (-w_B[1, i], w_B[0, i], 0.)
            T_tot[:, i] = np.dot(self.J, wdot_B[:, i]) + \
                np.dot(wx, np.dot(self.J, w_B[:, i]))

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        w_B = inputs['w_B']

        self.dT_dw = np.zeros((self.n, 3, 3))
        wx = np.zeros((3, 3))

        for i in range(0, self.n):
            wx[0, :] = (0., -w_B[2, i], w_B[1, i])
            wx[1, :] = (w_B[2, i], 0., -w_B[0, i])
            wx[2, :] = (-w_B[1, i], w_B[0, i], 0.)

            self.dT_dwdot[i, :, :] = self.J
            self.dT_dw[i, :, :] = np.dot(wx, self.J)

            for k in range(0, 3):
                self.dT_dw[i, :, k] += np.dot(self.dwx_dw[:, :, k],
                                              np.dot(self.J, w_B[:, i]))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dT_tot = d_outputs['T_tot']

        if mode == 'fwd':
            for k in range(3):
                for j in range(3):
                    if 'w_B' in d_inputs:
                        dT_tot[k, :] += self.dT_dw[:, k, j] * \
                            d_inputs['w_B'][j, :]
                    if 'wdot_B' in d_inputs:
                        dT_tot[k, :] += self.dT_dwdot[:, k, j] * \
                            d_inputs['wdot_B'][j, :]
        else:
            for k in range(3):
                for j in range(3):
                    if 'w_B' in d_inputs:
                        d_inputs['w_B'][j, :] += self.dT_dw[:, k, j] * \
                            dT_tot[k, :]
                    if 'wdot_B' in d_inputs:
                        d_inputs['wdot_B'][j, :] += self.dT_dwdot[:, k, j] * \
                            dT_tot[k, :]
