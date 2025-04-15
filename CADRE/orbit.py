"""
Orbit discipline for CADRE
"""
from math import sqrt

from six.moves import range
import numpy as np

from CADRE.explicit import ExplicitComponent

from CADRE import rk4


# Constants
mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

C1 = -mu
C2 = -1.5*mu*J2*Re**2
C3 = -2.5*mu*J3*Re**3
C4 = 1.875*mu*J4*Re**4


class Orbit_Dynamics(rk4.RK4):
    """
    Computes the Earth to body position vector in Earth-centered intertial frame.
    """

    def __init__(self, n_times, h):
        super(Orbit_Dynamics, self).__init__(n_times, h)

        self.n_times = n_times

        self.options['state_var'] = 'r_e2b_I'
        self.options['init_state_var'] = 'r_e2b_I0'

    def setup(self):
        self.add_input('r_e2b_I0', np.zeros((6, )), units=None,  # fd_step=1e-2,
                       desc='Initial position and velocity vectors from earth to '
                            'satellite in Earth-centered inertial frame')

        self.add_output('r_e2b_I', 1000.0*np.ones((6, self.n_times)), units=None,
                        desc='Position and velocity vectors from earth to satellite '
                             'in Earth-centered inertial frame over time')

        self.dfdx = np.zeros((6, 1))

    def f_dot(self, external, state):

        x = state[0]
        y = state[1]
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z*z
        z3 = z2*z
        z4 = z3*z

        r = sqrt(x*x + y*y + z2)

        r2 = r*r
        r3 = r2*r
        r4 = r3*r
        r5 = r4*r
        r7 = r5*r*r

        T2 = 1 - 5*z2/r2
        T3 = 3*z - 7*z3/r2
        T4 = 1 - 14*z2/r2 + 21*z4/r4
        T3z = 3*z - 0.6*r2/z
        T4z = 4 - 28.0/3.0*z2/r2

        f_dot = np.zeros((6,))
        f_dot[0:3] = state[3:]
        f_dot[3:] = state[0:3]*(C1/r3 + C2/r5*T2 + C3/r7*T3 + C4/r7*T4)
        f_dot[5] += z*(2.0*C2/r5 + C3/r7*T3z + C4/r7*T4z)

        return f_dot

    def df_dy(self, external, state):

        x = state[0]
        y = state[1]
        z = state[2] if abs(state[2]) > 1e-15 else 1e-5

        z2 = z*z
        z3 = z2*z
        z4 = z3*z

        r = sqrt(x*x + y*y + z2)

        r2 = r*r
        r3 = r2*r
        r4 = r3*r
        r5 = r4*r
        r6 = r5*r
        r7 = r6*r
        r8 = r7*r

        dr = np.array([x, y, z])/r

        T2 = 1 - 5*z2/r2
        T3 = 3*z - 7*z3/r2
        T4 = 1 - 14*z2/r2 + 21*z4/r4
        T3z = 3*z - 0.6*r2/z
        T4z = 4 - 28.0/3.0*z2/r2

        dT2 = (10*z2)/(r3)*dr
        dT2[2] -= 10.*z/r2

        dT3 = 14*z3/r3*dr
        dT3[2] -= 21.*z2/r2 - 3

        dT4 = (28*z2/r3 - 84.*z4/r5)*dr
        dT4[2] -= 28*z/r2 - 84*z3/r4

        dT3z = -1.2*r/z*dr
        dT3z[2] += 0.6*r2/z2 + 3

        dT4z = 56.0/3.0*z2/r3*dr
        dT4z[2] -= 56.0/3.0*z/r2

        eye = np.identity(3)

        dfdy = np.zeros((6, 6))
        dfdy[0:3, 3:] += eye

        dfdy[3:, :3] += eye*(C1/r3 + C2/r5*T2 + C3/r7*T3 + C4/r7*T4)
        fact = (-3*C1/r4 - 5*C2/r6*T2 - 7*C3/r8*T3 - 7*C4/r8*T4)
        dfdy[3:, 0] += dr[0]*state[:3]*fact
        dfdy[3:, 1] += dr[1]*state[:3]*fact
        dfdy[3:, 2] += dr[2]*state[:3]*fact
        dfdy[3:, 0] += state[:3]*(C2/r5*dT2[0] + C3/r7*dT3[0] + C4/r7*dT4[0])
        dfdy[3:, 1] += state[:3]*(C2/r5*dT2[1] + C3/r7*dT3[1] + C4/r7*dT4[1])
        dfdy[3:, 2] += state[:3]*(C2/r5*dT2[2] + C3/r7*dT3[2] + C4/r7*dT4[2])
        dfdy[5, :3] += dr*z*(-5*C2/r6*2 - 7*C3/r8*T3z - 7*C4/r8*T4z)
        dfdy[5, :3] += z*(C3/r7*dT3z + C4/r7*dT4z)
        dfdy[5, 2] += (C2/r5*2 + C3/r7*T3z + C4/r7*T4z)

        return dfdy

    def df_dx(self, external, state):

        return self.dfdx


class Orbit_Initial(ExplicitComponent):
    """
    Computes initial position and velocity vectors of Earth to body position
    """

    def setup(self):
        # Inputs
        self.add_input('altPerigee', 500.)
        self.add_input('altApogee', 500.)
        self.add_input('RAAN', 66.279)
        self.add_input('Inc', 82.072)
        self.add_input('argPerigee', 0.0)
        self.add_input('trueAnomaly', 337.987)

        # Outputs
        self.add_output('r_e2b_I0', np.ones((6,)), units=None,
                        desc='Initial position and velocity vectors from Earth '
                             'to satellite in Earth-centered inertial frame')

        self.declare_partials('*', '*')

    def compute_rv(self, altPerigee, altApogee, RAAN, Inc, argPerigee, trueAnomaly):
        """
        Compute position and velocity from orbital elements
        """
        Re = 6378.137
        mu = 398600.44

        def S(v):
            S = np.zeros((3, 3), complex)
            S[0, :] = [0, -v[2], v[1]]
            S[1, :] = [v[2], 0, -v[0]]
            S[2, :] = [-v[1], v[0], 0]
            return S

        def getRotation(axis, angle):
            R = np.eye(3, dtype=complex) + S(axis)*np.sin(angle) + \
                (1 - np.cos(angle)) * (np.outer(axis, axis) - np.eye(3, dtype=complex))
            return R

        d2r = np.pi/180.0
        r_perigee = Re + altPerigee
        r_apogee = Re + altApogee
        e = (r_apogee-r_perigee)/(r_apogee+r_perigee)
        a = (r_perigee+r_apogee)/2
        p = a*(1-e**2)
        # h = np.sqrt(p*mu)

        rmag0 = p/(1+e*np.cos(d2r*trueAnomaly))

        r0_P = np.empty(3, dtype=complex)
        r0_P[0] = rmag0*np.cos(d2r*trueAnomaly)
        r0_P[1] = rmag0*np.sin(d2r*trueAnomaly)
        r0_P[2] = 0

        v0_P = np.empty(3, dtype=complex)
        v0_P[0] = -np.sqrt(mu/p)*np.sin(d2r*trueAnomaly)
        v0_P[1] = np.sqrt(mu/p)*(e+np.cos(d2r*trueAnomaly))
        v0_P[2] = 0

        O_IP = np.eye(3, dtype=complex)
        O_IP = np.dot(O_IP, getRotation(np.array([0, 0, 1]), RAAN*d2r))
        O_IP = np.dot(O_IP, getRotation(np.array([1, 0, 0]), Inc*d2r))
        O_IP = np.dot(O_IP, getRotation(np.array([0, 0, 1]), argPerigee*d2r))

        r0_ECI = np.dot(O_IP, r0_P)
        v0_ECI = np.dot(O_IP, v0_P)

        return r0_ECI, v0_ECI

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r0_ECI, v0_ECI = self.compute_rv(inputs['altPerigee'], inputs['altApogee'],
                                         inputs['RAAN'], inputs['Inc'], inputs['argPerigee'],
                                         inputs['trueAnomaly'])
        outputs['r_e2b_I0'][:3] = r0_ECI.real
        outputs['r_e2b_I0'][3:] = v0_ECI.real

    def compute_partials(self, inputs, J):
        """
        Calculate and save derivatives. (i.e., Jacobian).
        """
        h = 1e-16
        ih = complex(0, h)
        v = np.array([inputs['altPerigee'], inputs['altApogee'], inputs['RAAN'],
                      inputs['Inc'], inputs['argPerigee'], inputs['trueAnomaly']],
                      dtype=complex)
        jacs = np.zeros((6, 6))

        # Find derivatives by complex step.
        for i in range(6):
            v[i] += ih
            r0_ECI, v0_ECI = self.compute_rv(v[0], v[1], v[2], v[3], v[4], v[5])
            v[i] -= ih
            jacs[:3, i] = r0_ECI.imag/h
            jacs[3:, i] = v0_ECI.imag/h

        J['r_e2b_I0', 'altPerigee'] = jacs[:, 0]
        J['r_e2b_I0', 'altApogee'] = jacs[:, 1]
        J['r_e2b_I0', 'RAAN'] = jacs[:, 2]
        J['r_e2b_I0', 'Inc'] = jacs[:, 3]
        J['r_e2b_I0', 'argPerigee'] = jacs[:, 4]
        J['r_e2b_I0', 'trueAnomaly'] = jacs[:, 5]
