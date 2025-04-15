"""
Sun discipline for CADRE
"""

from six.moves import range
import numpy as np
import scipy.sparse

from CADRE.explicit import ExplicitComponent

from CADRE.kinematics import computepositionrotd, computepositionrotdjacobian
from CADRE.kinematics import computepositionspherical, computepositionsphericaljacobian


class Sun_LOS(ExplicitComponent):
    """
    Compute the Satellite to sun line of sight.
    """

    def __init__(self, n=2):
        super(Sun_LOS, self).__init__()

        self.n = n

        # Earth's radius is 6378 km. 0.85 is the alpha in John Hwang's paper
        self.r1 = 6378.137 * 0.85
        self.r2 = 6378.137

    def setup(self):
        n = self.n

        self.add_input('r_e2b_I', np.zeros((6, n), order='F'), units=None,
                       desc='Position and velocity vectors from '
                            'Earth to satellite in Earth-centered '
                            'inertial frame over time.')

        self.add_input('r_e2s_I', np.zeros((3, n), order='F'), units='km',
                       desc='Position vector from Earth to sun in Earth-centered '
                            'inertial frame over time.')

        self.add_output('LOS', np.zeros((n, ), order='F'), units=None,
                        desc='Satellite to sun line of sight over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_e2b_I = inputs['r_e2b_I']
        r_e2s_I = inputs['r_e2s_I']
        LOS = outputs['LOS']

        for i in range(self.n):
            r_b = r_e2b_I[:3, i]
            r_s = r_e2s_I[:3, i]
            dot = np.dot(r_b, r_s)
            cross = np.cross(r_b, r_s)
            dist = np.sqrt(cross.dot(cross))

            if dot >= 0.0:
                LOS[i] = 1.0
            elif dist <= self.r1:
                LOS[i] = 0.0
            elif dist >= self.r2:
                LOS[i] = 1.0
            else:
                x = (dist - self.r1) / (self.r2 - self.r1)
                LOS[i] = 3*x**2 - 2*x**3

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2b_I = inputs['r_e2b_I']
        r_e2s_I = inputs['r_e2s_I']

        nj = 3*self.n

        Jab = np.zeros(shape=(nj, ), dtype=float)
        Jib = np.zeros(shape=(nj, ), dtype=int)
        Jjb = np.zeros(shape=(nj, ), dtype=int)
        Jas = np.zeros(shape=(nj, ), dtype=float)
        Jis = np.zeros(shape=(nj, ), dtype=int)
        Jjs = np.zeros(shape=(nj, ), dtype=int)

        r_b = np.zeros(shape=(3, ), dtype=int)
        r_s = np.zeros(shape=(3, ), dtype=int)
        Bx = np.zeros(shape=(3, 3, ), dtype=int)
        Sx = np.zeros(shape=(3, 3, ), dtype=int)
        cross = np.zeros(shape=(3, ), dtype=int)
        # ddist_cross = np.zeros(shape=(3, ), dtype=int)
        dcross_drb = np.zeros(shape=(3, 3, ), dtype=int)
        dcross_drs = np.zeros(shape=(3, 3, ), dtype=int)
        dLOS_dx = np.zeros(shape=(3, ), dtype=int)
        dLOS_drs = np.zeros(shape=(3, ), dtype=int)
        dLOS_drb = np.zeros(shape=(3, ), dtype=int)

        for i in range(self.n):
            r_b = r_e2b_I[:3, i]
            r_s = r_e2s_I[:3, i]
            Bx = crossMatrix(r_b)
            Sx = crossMatrix(-r_s)
            dot = np.dot(r_b, r_s)
            cross = np.cross(r_b, r_s)
            dist = np.sqrt(np.dot(cross, cross))

            if dot >= 0.0:
                dLOS_drb[:] = 0.0
                dLOS_drs[:] = 0.0
            elif dist <= self.r1:
                dLOS_drb[:] = 0.0
                dLOS_drs[:] = 0.0
            elif dist >= self.r2:
                dLOS_drb[:] = 0.0
                dLOS_drs[:] = 0.0
            else:
                x = (dist-self.r1)/(self.r2-self.r1)
                # LOS = 3*x**2 - 2*x**3
                ddist_dcross = cross/dist
                dcross_drb = Sx
                dcross_drs = Bx
                dx_ddist = 1.0/(self.r2-self.r1)
                dLOS_dx = 6*x - 6*x**2
                dLOS_drb = dLOS_dx*dx_ddist*np.dot(ddist_dcross, dcross_drb)
                dLOS_drs = dLOS_dx*dx_ddist*np.dot(ddist_dcross, dcross_drs)

            for k in range(3):
                iJ = i*3 + k
                Jab[iJ] = dLOS_drb[k]
                Jib[iJ] = i
                Jjb[iJ] = (i)*6 + k
                Jas[iJ] = dLOS_drs[k]
                Jis[iJ] = i
                Jjs[iJ] = (i)*3 + k

        self.Jb = scipy.sparse.csc_matrix((Jab, (Jib, Jjb)),
                                          shape=(self.n, 6*self.n))
        self.Js = scipy.sparse.csc_matrix((Jas, (Jis, Jjs)),
                                          shape=(self.n, 3*self.n))
        self.JbT = self.Jb.transpose()
        self.JsT = self.Js.transpose()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dLOS = d_outputs['LOS']

        if mode == 'fwd':
            if 'r_e2b_I' in d_inputs:
                r_e2b_I = d_inputs['r_e2b_I'][:].reshape((6*self.n), order='F')
                dLOS += self.Jb.dot(r_e2b_I)

            if 'r_e2s_I' in d_inputs:
                r_e2s_I = d_inputs['r_e2s_I'][:].reshape((3*self.n), order='F')
                dLOS += self.Js.dot(r_e2s_I)

        else:
            if 'r_e2b_I' in d_inputs:
                d_inputs['r_e2b_I'] += self.JbT.dot(dLOS).reshape((6, self.n), order='F')
            if 'r_e2s_I' in d_inputs:
                d_inputs['r_e2s_I'] += self.JsT.dot(dLOS).reshape((3, self.n), order='F')


def crossMatrix(v):
    # so m[1,0] is v[2], for example
    m = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return m


class Sun_PositionBody(ExplicitComponent):
    """
    Position vector from earth to sun in body-fixed frame.
    """

    def __init__(self, n=2):
        super(Sun_PositionBody, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('O_BI', np.zeros((3, 3, n), order='F'), units=None,
                       desc='Rotation matrix from the Earth-centered inertial frame '
                            'to the satellite frame.')

        self.add_input('r_e2s_I', np.zeros((3, n), order='F'), units='km',
                       desc='Position vector from Earth to Sun in Earth-centered '
                            'inertial frame over time.')

        # Outputs
        self.add_output('r_e2s_B', np.zeros((3, n, ), order='F'), units='km',
                        desc='Position vector from Earth to Sun in body-fixed '
                             'frame over time.')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['r_e2s_B'] = computepositionrotd(self.n, inputs['r_e2s_I'],
                                                 inputs['O_BI'])

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.J1, self.J2 = computepositionrotdjacobian(self.n, inputs['r_e2s_I'],
                                                       inputs['O_BI'])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_e2s_B = d_outputs['r_e2s_B']

        if mode == 'fwd':
            if 'O_BI' in d_inputs:
                for k in range(3):
                    for u in range(3):
                        for v in range(3):
                            dr_e2s_B[k, :] += self.J1[:, k, u, v] * d_inputs['O_BI'][u, v, :]
            if 'r_e2s_I' in d_inputs:
                for k in range(3):
                    for j in range(3):
                        dr_e2s_B[k, :] += self.J2[:, k, j] * d_inputs['r_e2s_I'][j, :]
        else:
            for k in range(3):
                if 'O_BI' in d_inputs:
                    dO_BI = d_inputs['O_BI']
                    for u in range(3):
                        for v in range(3):
                            dO_BI[u, v, :] += self.J1[:, k, u, v] * dr_e2s_B[k, :]
                if 'r_e2s_I' in d_inputs:
                    dr_e2s_I = d_inputs['r_e2s_I']
                    for j in range(3):
                        dr_e2s_I[j, :] += self.J2[:, k, j] * dr_e2s_B[k, :]


class Sun_PositionECI(ExplicitComponent):
    """
    Compute the position vector from Earth to Sun in Earth-centered inertial frame.
    """

    # constants
    d2r = np.pi/180.

    def __init__(self, n=2):
        super(Sun_PositionECI, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('LD', 0.0, units=None)

        self.add_input('t', np.zeros((n, ), order='F'), units='s', desc='Time')

        # Outputs
        self.add_output('r_e2s_I', np.zeros((3, n, ), order='F'), units='km',
                        desc='Position vector from Earth to Sun in Earth-centered '
                             'inertial frame over time.')

        self.Ja = np.zeros(3*n)
        self.Ji = np.zeros(3*n)
        self.Jj = np.zeros(3*n)

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_e2s_I = outputs['r_e2s_I']

        T = inputs['LD'] + inputs['t'][:]/3600./24.
        for i in range(0, self.n):
            L = self.d2r*280.460 + self.d2r*0.9856474*T[i]
            g = self.d2r*357.528 + self.d2r*0.9856003*T[i]
            Lambda = L + self.d2r*1.914666*np.sin(g) + self.d2r*0.01999464*np.sin(2*g)
            eps = self.d2r*23.439 - self.d2r*3.56e-7*T[i]
            r_e2s_I[0, i] = np.cos(Lambda)
            r_e2s_I[1, i] = np.sin(Lambda)*np.cos(eps)
            r_e2s_I[2, i] = np.sin(Lambda)*np.sin(eps)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        T = inputs['LD'] + inputs['t'][:]/3600./24.
        dr_dt = np.empty(3)
        for i in range(0, self.n):
            L = self.d2r*280.460 + self.d2r*0.9856474*T[i]
            g = self.d2r*357.528 + self.d2r*0.9856003*T[i]
            Lambda = L + self.d2r*1.914666*np.sin(g) + self.d2r*0.01999464*np.sin(2*g)
            eps = self.d2r*23.439 - self.d2r*3.56e-7*T[i]

            dL_dt = self.d2r*0.9856474
            dg_dt = self.d2r*0.9856003
            dlambda_dt = (dL_dt + self.d2r*1.914666*np.cos(g)*dg_dt +
                          self.d2r*0.01999464*np.cos(2*g)*2*dg_dt)
            deps_dt = -self.d2r*3.56e-7

            dr_dt[0] = -np.sin(Lambda)*dlambda_dt
            dr_dt[1] = np.cos(Lambda)*np.cos(eps)*dlambda_dt - np.sin(Lambda)*np.sin(eps)*deps_dt
            dr_dt[2] = np.cos(Lambda)*np.sin(eps)*dlambda_dt + np.sin(Lambda)*np.cos(eps)*deps_dt

            for k in range(0, 3):
                iJ = i*3 + k
                self.Ja[iJ] = dr_dt[k]
                self.Ji[iJ] = iJ
                self.Jj[iJ] = i

        self.J = scipy.sparse.csc_matrix((self.Ja, (self.Ji, self.Jj)),
                                         shape=(3*self.n, self.n))
        self.JT = self.J.transpose()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dr_e2s_I = d_outputs['r_e2s_I']

        if mode == 'fwd':
            if 'LD' in d_inputs and 't' in d_inputs:
                # TODO - Should split this up so we can hook one up but not the other.
                dr_e2s_I[:] += (self.J.dot(d_inputs['LD'] +
                                d_inputs['t']/3600./24.).reshape((3, self.n), order='F'))
        else:
            r_e2s_I = dr_e2s_I[:].reshape((3*self.n), order='F')
            if 'LD' in d_inputs:
                d_inputs['LD'] += sum(self.JT.dot(r_e2s_I))
            if 't' in d_inputs:
                d_inputs['t'] += self.JT.dot(r_e2s_I)/3600.0/24.0


class Sun_PositionSpherical(ExplicitComponent):
    """
    Compute the elevation angle of the Sun in the body-fixed frame.
    """

    def __init__(self, n=2):
        super(Sun_PositionSpherical, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('r_e2s_B', np.zeros((3, n)), units='km',
                       desc='Position vector from Earth to Sun in body-fixed '
                            'frame over time.')

        # Outputs
        self.add_output('azimuth', np.zeros((n,)), units='rad',
                        desc='Ezimuth angle of the Sun in the body-fixed frame '
                             'over time.')

        self.add_output('elevation', np.zeros((n, )), units='rad',
                        desc='Elevation angle of the Sun in the body-fixed frame '
                             'over time.')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        azimuth, elevation = computepositionspherical(self.n, inputs['r_e2s_B'])

        outputs['azimuth'] = azimuth
        outputs['elevation'] = elevation

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        self.Ja1, self.Ji1, self.Jj1, self.Ja2, self.Ji2, self.Jj2 = \
            computepositionsphericaljacobian(self.n, 3*self.n, inputs['r_e2s_B'])
        self.J1 = scipy.sparse.csc_matrix((self.Ja1, (self.Ji1, self.Jj1)),
                                          shape=(self.n, 3*self.n))
        self.J2 = scipy.sparse.csc_matrix((self.Ja2, (self.Ji2, self.Jj2)),
                                          shape=(self.n, 3*self.n))
        self.J1T = self.J1.transpose()
        self.J2T = self.J2.transpose()

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'r_e2s_B' in d_inputs:
                r_e2s_B = d_inputs['r_e2s_B'].reshape((3*self.n), order='F')
                if 'azimuth' in d_outputs:
                    d_outputs['azimuth'] += self.J1.dot(r_e2s_B)
                if 'elevation' in d_outputs:
                    d_outputs['elevation'] += self.J2.dot(r_e2s_B)
        else:
            if 'r_e2s_B' in d_inputs:
                if 'azimuth' in d_outputs:
                    azimuth = d_outputs['azimuth'][:]
                    d_inputs['r_e2s_B'] += self.J1T.dot(azimuth).reshape((3, self.n), order='F')
                if 'elevation' in d_outputs:
                    elevation = d_outputs['elevation'][:]
                    d_inputs['r_e2s_B'] += self.J2T.dot(elevation).reshape((3, self.n), order='F')
