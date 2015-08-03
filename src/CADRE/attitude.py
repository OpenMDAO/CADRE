''' Attitude discipline for CADRE. '''

from six.moves import range
import numpy as np

from openmdao.core.component import Component

from CADRE.kinematics import computepositionrotd, computepositionrotdjacobian

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103


class Attitude_Angular(Component):
    """ Calculates angular velocity vector from the satellite's orientation
    matrix and its derivative.
    """

    def __init__(self, n=2):
        super(Attitude_Angular, self).__init__()

        self.n = n

        # Inputs
        self.add_param('O_BI', np.zeros((3, 3, n)), units="unitless",
                       desc="Rotation matrix from body-fixed frame to Earth-centered "
                       "inertial frame over time")

        self.add_param('Odot_BI', np.zeros((3, 3, n)), units="unitless",
                       desc="First derivative of O_BI over time")

        # Outputs
        self.add_output('w_B', np.zeros((3, n)), units="1/s",
                        desc="Angular velocity vector in body-fixed frame over time")

        self.dw_dOdot = np.zeros((n, 3, 3, 3))
        self.dw_dO = np.zeros((n, 3, 3, 3))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        O_BI = params['O_BI']
        Odot_BI = params['Odot_BI']
        w_B = unknowns['w_B']

        for i in range(0, self.n):
            w_B[0, i] = np.dot(Odot_BI[2, :, i], O_BI[1, :, i])
            w_B[1, i] = np.dot(Odot_BI[0, :, i], O_BI[2, :, i])
            w_B[2, i] = np.dot(Odot_BI[1, :, i], O_BI[0, :, i])

    def jacobian(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        O_BI = params['O_BI']
        Odot_BI = params['Odot_BI']

        for i in range(0, self.n):
            self.dw_dOdot[i, 0, 2, :] = O_BI[1, :, i]
            self.dw_dO[i, 0, 1, :] = Odot_BI[2, :, i]

            self.dw_dOdot[i, 1, 0, :] = O_BI[2, :, i]
            self.dw_dO[i, 1, 2, :] = Odot_BI[0, :, i]

            self.dw_dOdot[i, 2, 1, :] = O_BI[0, :, i]
            self.dw_dO[i, 2, 0, :] = Odot_BI[1, :, i]

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dw_B = dresids['w_B']

        if mode == 'fwd':
            for k in range(3):
                for i in range(3):
                    for j in range(3):
                        if 'O_BI' in dparams:
                            dw_B[k, :] += self.dw_dO[:, k, i, j] * \
                                dparams['O_BI'][i, j, :]
                        if 'Odot_BI' in dparams:
                            dw_B[k, :] += self.dw_dOdot[:, k, i, j] * \
                                dparams['Odot_BI'][i, j, :]

        else:
            for k in range(3):
                for i in range(3):
                    for j in range(3):

                        if 'O_BI' in dparams:
                            O_BI = dparams['O_BI']
                            O_BI[i, j, :] = self.dw_dO[:, k, i, j] * \
                                dw_B[k, :]
                            dparams['O_BI'] = O_BI

                        if 'Odot_BI' in dparams:
                            Odot_BI = dparams['Odot_BI']
                            Odot_BI[i, j, :] = self.dw_dOdot[:, k, i, j] * \
                                dw_B[k, :]
                            dparams['Odot_BI'] = Odot_BI


class Attitude_AngularRates(Component):
    """ Calculates time derivative of angular velocity vector.
    """

    def __init__(self, n=2, h=28.8):
        super(Attitude_AngularRates, self).__init__()

        self.n = n

        # Inputs
        self.add_param('w_B', np.zeros((3, n)), units="1/s",
                       desc="Angular velocity vector in body-fixed frame over time")

        # Outputs
        self.add_output('wdot_B', np.zeros((3, n)), units="1/s**2",
                        desc="Time derivative of w_B over time")

        self.h = h

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        w_B = params['w_B']
        h = self.h
        wdot_B = unknowns['wdot_B']

        wdot_B[:, 0] = w_B[:, 1] - w_B[:, 0]
        wdot_B[:, 1:-1] = (w_B[:, 2:] - w_B[:, :-2])/ 2.0
        wdot_B[:, -1] = w_B[:, -1] - w_B[:, -2]
        wdot_B *= 1.0/h

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        h = self.h
        dwdot_B = dresids['wdot_B']

        if mode == 'fwd':
            if 'w_B' in dparams:
                dwdot_B[:, 0] += dparams['w_B'][:, 1] / h
                dwdot_B[:, 0] -= dparams['w_B'][:, 0] / h
                dwdot_B[:, 1:-1] += dparams['w_B'][:, 2:] / 2.0 / h
                dwdot_B[:, 1:-1] -= dparams['w_B'][:, :-2] / 2.0 / h
                dwdot_B[:, -1] += dparams['w_B'][:, -1] / h
                dwdot_B[:, -1] -= dparams['w_B'][:, -2] / h

        else:
            if 'w_B' in dparams:
                w_B = dparams['w_B']
                w_B[:, 1] += dwdot_B[:, 0] / h
                w_B[:, 0] -= dwdot_B[:, 0] / h
                w_B[:, 2:] += dwdot_B[:, 1:-1] / 2.0 / h
                w_B[:, :-2] -= dwdot_B[:, 1:-1] / 2.0 / h
                w_B[:, -1] += dwdot_B[:, -1] / h
                w_B[:, -2] -= dwdot_B[:, -1] / h
                dparams['w_B'] = w_B


class Attitude_Attitude(Component):
    """ Coordinate transformation from the interial plane to the rolled
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

        # Inputs
        self.add_param('r_e2b_I', np.zeros((6, n)), units="unitless",
                       desc="Position and velocity vector from earth to satellite in "
                       "Earth-centered inertial frame over time")

        # Outputs
        self.add_output('O_RI', np.zeros((3, 3, n)), units="unitless",
                        desc="Rotation matrix from rolled body-fixed frame to "
                        "Earth-centered inertial frame over time")

        self.dO_dr = np.zeros((n, 3, 3, 6))

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        r_e2b_I = params['r_e2b_I']
        O_RI = unknowns['O_RI']

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

    def jacobian(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        r_e2b_I = params['r_e2b_I']

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

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dO_RI = dresids['O_RI']

        if mode == 'fwd':
            for k in range(3):
                for j in range(3):
                    for i in range(6):
                        dO_RI[k, j, :] += self.dO_dr[:, k, j, i] * \
                            dparams['r_e2b_I'][i, :]

        else:
            for k in range(3):
                for j in range(3):
                    for i in range(6):
                        r_e2b_I = dparams['r_e2b_I']
                        r_e2b_I[i, :] += self.dO_dr[:, k, j, i] * \
                            dO_RI[k, j, :]
                        dparams['r_e2b_I'] = r_e2b_I


class Attitude_Roll(Component):
    """ Calculates the body-fixed orientation matrix.
    """

    def __init__(self, n=2):
        super(Attitude_Roll, self).__init__()

        self.n = n

        # Inputs
        self.add_param('Gamma', np.zeros(n), units="rad",
                       desc="Satellite roll angle over time")

        # Outputs
        self.add_output('O_BR', np.zeros((3, 3, n)), units="unitless",
                        desc="Rotation matrix from body-fixed frame to rolled "
                        "body-fixed frame over time")

        self.dO_dg = np.zeros((n, 3, 3))


    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        Gamma = params['Gamma']
        O_BR = unknowns['O_BR']

        O_BR[:] = np.zeros((3, 3, self.n))
        O_BR[0, 0, :] = np.cos(Gamma)
        O_BR[0, 1, :] = np.sin(Gamma)
        O_BR[1, 0, :] = -O_BR[0, 1, :]
        O_BR[1, 1, :] = O_BR[0, 0, :]
        O_BR[2, 2, :] = np.ones(self.n)

    def jacobian(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        Gamma = params['Gamma']

        self.dO_dg = np.zeros((self.n, 3, 3))
        self.dO_dg[:, 0, 0] = -np.sin(Gamma)
        self.dO_dg[:, 0, 1] = np.cos(Gamma)
        self.dO_dg[:, 1, 0] = -self.dO_dg[:, 0, 1]
        self.dO_dg[:, 1, 1] = self.dO_dg[:, 0, 0]

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dO_BR = dresids['O_BR']

        if mode == 'fwd':
            for k in range(3):
                for j in range(3):
                    dO_BR[k, j, :] += self.dO_dg[:, k, j] * \
                        dparams['Gamma']

        else:
            for k in range(3):
                for j in range(3):
                    dparams['Gamma'] += self.dO_dg[:, k, j] * \
                        dO_BR[k, j, :]


class Attitude_RotationMtx(Component):
    """ Multiplies transformations to produce the orientation matrix of the
    body frame with respect to inertial.
    """

    def __init__(self, n=2):
        super(Attitude_RotationMtx, self).__init__()

        self.n = n

        # Inputs
        self.add_param('O_BR', np.zeros((3, 3, n)), units="unitless",
                       desc="Rotation matrix from body-fixed frame to rolled "
                       "body-fixed frame over time")

        self.add_param('O_RI', np.zeros((3, 3, n)), units="unitless",
                       desc="Rotation matrix from rolled body-fixed "
                       "frame to Earth-centered inertial frame over time")

        # Outputs
        self.add_output('O_BI', np.zeros((3, 3, n)), units="unitless",
                        desc="Rotation matrix from body-fixed frame to "
                        "Earth-centered inertial frame over time")

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        O_BR = params['O_BR']
        O_RI = params['O_RI']
        O_BI = unknowns['O_BI']

        for i in range(0, self.n):
            O_BI[:, :, i] = np.dot(O_BR[:, :, i], O_RI[:, :, i])

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dO_BI = dresids['O_BI']
        O_BR = params['O_BR']
        O_RI = params['O_RI']

        if mode == 'fwd':
            for u in range(3):
                for v in range(3):
                    for k in range(3):
                        if 'O_RI' in dparams:
                            dO_BI[u, v, :] += O_BR[u, k, :] * \
                                dparams['O_RI'][k, v, :]
                        if 'O_BR' in dparams:
                            dO_BI[u, v, :] += dparams['O_BR'][u, k, :] * \
                                O_RI[k, v, :]

        else:
            for u in range(3):
                for v in range(3):
                    for k in range(3):
                        if 'O_RI' in dparams:
                            dO_RI = dparams['O_RI']
                            dO_RI[k, v, :] += O_BR[u, k, :] * \
                                dO_BI[u, v, :]
                            dparams['O_RI'] = dO_RI
                        if 'O_BR' in dparams:
                            dO_BR = dparams['O_BR']
                            dO_BR[u, k, :] += dO_BI[u, v, :] * \
                                O_RI[k, v, :]
                            dparams['O_BR'] = dO_BR


class Attitude_RotationMtxRates(Component):
    """ Calculates time derivative of body frame orientation matrix.
    """

    def __init__(self, n=2, h=28.2):
        super(Attitude_RotationMtxRates, self).__init__()

        self.n = n
        self.h = h

        # Inputs
        self.add_param('O_BI', np.zeros((3, 3, n)), units="unitless",
                       desc="Rotation matrix from body-fixed frame to Earth-centered "
                       "inertial frame over time")

        # Outputs
        self.add_output('Odot_BI', np.zeros((3, 3, n)), units="unitless",
                        desc="First derivative of O_BI over time")


    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        O_BI = params['O_BI']
        h = self.h
        Odot_BI = unknowns['Odot_BI']

        Odot_BI[:, :, 0] = O_BI[:, :, 1]
        Odot_BI[:, :, 0] -= O_BI[:, :, 0]
        Odot_BI[:, :, 1:-1] = O_BI[:, :, 2:] / 2.0
        Odot_BI[:, :, 1:-1] -= O_BI[:, :, :-2] / 2.0
        Odot_BI[:, :, -1] = O_BI[:, :, -1]
        Odot_BI[:, :, -1] -= O_BI[:, :, -2]
        Odot_BI *= 1.0/h

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dOdot_BI = dresids['Odot_BI']
        h = self.h

        if mode == 'fwd':
            dO_BI = dparams['O_BI']
            dOdot_BI[:, :, 0] += dO_BI[:, :, 1] / h
            dOdot_BI[:, :, 0] -= dO_BI[:, :, 0] / h
            dOdot_BI[:, :, 1:-1] += dO_BI[:, :, 2:] / (2.0*h)
            dOdot_BI[:, :, 1:-1] -= dO_BI[:, :, :-2] / (2.0*h)
            dOdot_BI[:, :, -1] += dO_BI[:, :, -1] / h
            dOdot_BI[:, :, -1] -= dO_BI[:, :, -2] / h

        else:
            dO_BI = dparams['O_BI']
            dO_BI[:, :, 1] += dOdot_BI[:, :, 0] / h
            dO_BI[:, :, 0] -= dOdot_BI[:, :, 0] / h
            dO_BI[:, :, 2:] += dOdot_BI[:, :, 1:-1] / (2.0*h)
            dO_BI[:, :, :-2] -= dOdot_BI[:, :, 1:-1] / (2.0*h)
            dO_BI[:, :, -1] += dOdot_BI[:, :, -1] / h
            dO_BI[:, :, -2] -= dOdot_BI[:, :, -1] / h
            dparams['O_BI'] = dO_BI


class Attitude_Sideslip(Component):
    """ Determine velocity in the body frame."""

    def __init__(self, n=2):
        super(Attitude_Sideslip, self).__init__()

        self.n = n

        # Inputs
        self.add_param('r_e2b_I', np.zeros((6, n)), units="unitless",
                       desc="Position and velocity vector from earth to satellite "
                       "in Earth-centered inertial frame over time")

        self.add_param('O_BI', np.zeros((3, 3, n)), units="unitless",
                       desc="Rotation matrix from body-fixed frame to "
                       "Earth-centered inertial frame over time")

        # Outputs
        self.add_output('v_e2b_B', np.zeros((3, n)), units="m/s",
                        desc="Velocity vector from earth to satellite"
                        "in body-fixed frame over time")


    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        r_e2b_I = params['r_e2b_I']
        O_BI = params['O_BI']
        v_e2b_B = unknowns['v_e2b_B']

        v_e2b_B[:] = computepositionrotd(self.n, r_e2b_I[3:, :], O_BI)

    def jacobian(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        r_e2b_I = params['r_e2b_I']
        O_BI = params['O_BI']

        self.J1, self.J2 = computepositionrotdjacobian(self.n,
                                                       r_e2b_I[3:, :],
                                                       O_BI)

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dv_e2b_B = dresids['v_e2b_B']

        if mode == 'fwd':
            for k in range(3):
                if 'O_BI' in params:
                    for u in range(3):
                        for v in range(3):
                            dv_e2b_B[k, :] += self.J1[:, k, u, v] * \
                                dparams['O_BI'][u, v, :]
                if 'r_e2b_I' in params:
                    for j in range(3):
                        dv_e2b_B[k, :] += self.J2[:, k, j] * \
                            dparams['r_e2b_I'][3+j, :]

        else:
            for k in range(3):
                if 'O_BI' in params:
                    for u in range(3):
                        for v in range(3):
                            dO_BI = dparams['O_BI']
                            dO_BI[u, v, :] += self.J1[:, k, u, v] * \
                                dv_e2b_B[k, :]
                            dparams['O_BI'] = dO_BI
                if 'r_e2b_I' in params:
                    for j in range(3):
                        dr_e2b_I = dparams['r_e2b_I']
                        dr_e2b_I[3+j, :] += self.J2[:, k, j] * \
                            dv_e2b_B[k, :]
                        dparams['r_e2b_I'] = dr_e2b_I


class Attitude_Torque(Component):
    """ Compute the required reaction wheel tourque."""

    J = np.zeros((3, 3))
    J[0, :] = (0.018, 0., 0.)
    J[1, :] = (0., 0.018, 0.)
    J[2, :] = (0., 0., 0.006)

    def __init__(self, n=2):
        super(Attitude_Torque, self).__init__()

        self.n = n

        # Inputs
        self.add_param('w_B', np.zeros((3, n)), units="1/s",
                       desc="Angular velocity in body-fixed frame over time")

        self.add_param('wdot_B', np.zeros((3, n)), units="1/s**2",
                       desc="Time derivative of w_B over time")

        # Outputs
        self.add_output('T_tot', np.zeros((3, n)), units="N*m",
                        desc="Total reaction wheel torque over time")

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

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        w_B = params['w_B']
        wdot_B = params['wdot_B']
        T_tot = unknowns['T_tot']

        wx = np.zeros((3, 3))
        for i in range(0, self.n):
            wx[0, :] = (0., -w_B[2, i], w_B[1, i])
            wx[1, :] = (w_B[2, i], 0., -w_B[0, i])
            wx[2, :] = (-w_B[1, i], w_B[0, i], 0.)
            T_tot[:, i] = np.dot(self.J, wdot_B[:, i]) + \
                np.dot(wx, np.dot(self.J, w_B[:, i]))

    def jacobian(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        w_B = params['w_B']

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

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dT_tot = dresids['T_tot']

        if mode == 'fwd':
            for k in range(3):
                for j in range(3):
                    if 'w_B' in dparams:
                        dT_tot[k, :] += self.dT_dw[:, k, j] * \
                            dparams['w_B'][j, :]
                    if 'wdot_B' in dparams:
                        dT_tot[k, :] += self.dT_dwdot[:, k, j] * \
                            dparams['wdot_B'][j, :]

        else:
            for k in range(3):
                for j in range(3):
                    if 'w_B' in dparams:
                        dw_B = dparams['w_B']
                        dw_B[j, :] += self.dT_dw[:, k, j] * \
                            dT_tot[k, :]
                        dparams['w_B'] = dw_B
                    if 'wdot_B' in dparams:
                        dwdot_B = dparams['wdot_B']
                        dwdot_B[j, :] += self.dT_dwdot[:, k, j] * \
                            dT_tot[k, :]
                        dparams['wdot_B'] = dwdot_B
