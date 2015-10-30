''' Power discipline for CADRE '''

import os
from six.moves import range

import numpy as np

from openmdao.core.component import Component

import MBI

# Allow non-standard variable names for scientific calc
# pylint: disable-msg=C0103


class Power_CellVoltage(Component):
    """Compute the output voltage of the solar panels.
    """

    def __init__(self, n, dat=None):
        super(Power_CellVoltage, self).__init__()

        self.n = n

        if dat is None:
            fpath = os.path.dirname(os.path.realpath(__file__))
            dat = np.genfromtxt(fpath + '/data/Power/curve.dat')

        # Inputs
        self.add_param('LOS', np.zeros((n)), units='unitless',
                       desc="Line of Sight over Time")

        self.add_param('temperature', np.zeros((5, n)), units="degK",
                       desc="Temperature of solar cells over time")

        self.add_param('exposedArea', np.zeros((7, 12, n)), units="m**2",
                       desc="Exposed area to sun for each solar cell over time")

        self.add_param('Isetpt', np.zeros((12, n)), units="A",
                       desc="Currents of the solar panels")

        # Outputs
        self.add_output('V_sol', np.zeros((12, n)), units="V",
                        desc="Output voltage of solar panel over time")

        nT, nA, nI = dat[:3]
        T = dat[3:3 + nT]
        A = dat[3 + nT:3 + nT + nA]
        I = dat[3 + nT + nA:3 + nT + nA + nI]
        V = dat[3 + nT + nA + nI:].reshape((nT, nA, nI), order='F')

        self.MBI = MBI.MBI(V, [T, A, I], [6, 6, 15], [3, 3, 3])

        self.x = np.zeros((84 * self.n, 3), order='F')
        self.xV = self.x.reshape((self.n, 7, 12, 3), order='F')
        self.dV_dL = np.zeros((self.n, 12), order='F')
        self.dV_dT = np.zeros((self.n, 12, 5), order='F')
        self.dV_dA = np.zeros((self.n, 7, 12), order='F')
        self.dV_dI = np.zeros((self.n, 12), order='F')

    def setx(self, params):

        temperature = params['temperature']
        LOS = params['LOS']
        exposedArea = params['exposedArea']
        Isetpt = params['Isetpt']

        for p in range(12):
            i = 4 if p < 4 else (p % 4)
            for c in range(7):
                self.xV[:, c, p, 0] = temperature[i, :]
                self.xV[:, c, p, 1] = LOS * exposedArea[c, p, :]
                self.xV[:, c, p, 2] = Isetpt[p, :]

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        self.setx(params)
        self.raw = self.MBI.evaluate(self.x)[:, 0].reshape((self.n, 7, 12),
                                                           order='F')
        unknowns['V_sol'] = np.zeros((12, self.n))
        for c in range(7):
            unknowns['V_sol'] += self.raw[:, c, :].T

    def linearize(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        exposedArea = params['exposedArea']
        LOS = params['LOS']

        self.raw1 = self.MBI.evaluate(self.x, 1)[:, 0].reshape((self.n, 7,
                                                                12), order='F')
        self.raw2 = self.MBI.evaluate(self.x, 2)[:, 0].reshape((self.n, 7,
                                                                12), order='F')
        self.raw3 = self.MBI.evaluate(self.x, 3)[:, 0].reshape((self.n, 7,
                                                                12), order='F')
        self.dV_dL[:] = 0.0
        self.dV_dT[:] = 0.0
        self.dV_dA[:] = 0.0
        self.dV_dI[:] = 0.0

        for p in range(12):
            i = 4 if p < 4 else (p % 4)
            for c in range(7):
                self.dV_dL[:, p] += self.raw2[:, c, p] * exposedArea[c, p,:]
                self.dV_dT[:, p, i] += self.raw1[:, c, p]
                self.dV_dA[:, c, p] += self.raw2[:, c, p] * LOS
                self.dV_dI[:, p] += self.raw3[:, c, p]

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dV_sol = dresids['V_sol']

        if mode == 'fwd':

            if 'LOS' in dparams:
                dV_sol += self.dV_dL.T * dparams['LOS']

            if 'temperature' in dparams:
                for p in range(12):
                    i = 4 if p < 4 else (p % 4)
                    dV_sol[p, :] += self.dV_dT[:, p, i] * dparams['temperature'][i,:]

            if 'Isetpt' in dparams:
                dV_sol += self.dV_dI.T * dparams['Isetpt']

            if 'exposedArea' in dparams:
                for p in range(12):
                    dV_sol[p, :] += \
                        np.sum(self.dV_dA[:,:, p] * dparams['exposedArea'][:, p,:].T, 1)

        else:
            for p in range(12):
                i = 4 if p < 4 else (p % 4)

                if 'LOS' in dparams:
                    dparams['LOS'] += self.dV_dL[:, p] * dV_sol[p,:]

                if 'temperature' in dparams:
                    dparams['temperature'][i,:] += self.dV_dT[:, p, i] * dV_sol[p,:]

                if 'Isetpt' in dparams:
                    dparams['Isetpt'][p,:] += self.dV_dI[:, p] * dV_sol[p,:]

                if 'exposedArea' in dparams:
                    dexposedArea = dparams['exposedArea']
                    for c in range(7):
                        dexposedArea[c, p, :] += self.dV_dA[:, c, p] * dV_sol[p,:]


class Power_SolarPower(Component):
    """ Compute the output power of the solar panels. """

    def __init__(self, n=2):
        super(Power_SolarPower, self).__init__()

        self.n = n

        # Inputs
        self.add_param('Isetpt', np.zeros((12, n)), units="A",
                       desc="Currents of the solar panels")

        self.add_param('V_sol', np.zeros((12, n)), units="V",
                       desc="Output voltage of solar panel over time")

        self.add_output('P_sol', np.zeros((n, )), units="W",
                        desc="Solar panels power over time")

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        V_sol = params['V_sol']
        Isetpt = params['Isetpt']
        P_sol = unknowns['P_sol']

        P_sol[:] = np.zeros((self.n, ))
        for p in range(12):
            P_sol += V_sol[p, :] * Isetpt[p, :]

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dP_sol = dresids['P_sol']

        if mode == 'fwd':
            if 'V_sol' in dparams:
                for p in range(12):
                    dP_sol += dparams['V_sol'][p, :] * params['Isetpt'][p, :]

            if 'Isetpt' in dparams:
                for p in range(12):
                    dP_sol += dparams['Isetpt'][p, :] * params['V_sol'][p, :]

        else:
            for p in range(12):
                if 'V_sol' in dparams:
                    dparams['V_sol'][p,:] += dP_sol* params['Isetpt'][p, :]

                if 'Isetpt' in dparams:
                    dparams['Isetpt'][p,:] += params['V_sol'][p, :] * dP_sol


class Power_Total(Component):
    """ Compute the battery power which is the sum of the loads. This
    includes a 2-Watt constant power usage that accounts for the scientific
    instruments on the satellite and small actuator inputs in response to
    disturbance torques.
    """

    def __init__(self, n=2):
        super(Power_Total, self).__init__()

        self.n = n

        # Inputs
        self.add_param('P_sol', np.zeros((n, ), order='F'), units='W',
                       desc='Solar panels power over time')

        self.add_param('P_comm', np.zeros((n, ), order='F'), units='W',
                       desc='Communication power over time')

        self.add_param('P_RW', np.zeros((3, n, ), order='F'), units='W',
                       desc='Power used by reaction wheel over time')

        # Outputs
        self.add_output('P_bat', np.zeros((n, ), order='F'), units='W',
                        desc='Battery power over time')

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        unknowns['P_bat'] = params['P_sol'] - 5.0*params['P_comm'] - \
            np.sum(params['P_RW'], 0) - 2.0

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        dP_bat = dresids['P_bat']

        if mode == 'fwd':
            if 'P_sol' in dparams:
                dP_bat += dparams['P_sol']

            if 'P_comm' in dparams:
                dP_bat -= 5.0 * dparams['P_comm']

            if 'P_RW' in dparams:
                for k in range(3):
                    dP_bat -= dparams['P_RW'][k,:]

        else:
            if 'P_sol' in dparams:
                dparams['P_sol'] += dP_bat[:]

            if 'P_comm' in dparams:
                dparams['P_comm'] -= 5.0 * dP_bat

            if 'P_RW' in dparams:
                for k in range(3):
                    dparams['P_RW'][k, :] -= dP_bat
