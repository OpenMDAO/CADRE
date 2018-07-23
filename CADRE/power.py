"""
Power discipline for CADRE
"""

import os
from six.moves import range

import numpy as np

from openmdao.api import ExplicitComponent

from MBI import MBI


class Power_CellVoltage(ExplicitComponent):
    """
    Compute the output voltage of the solar panels.
    """

    def __init__(self, n, filename=None):
        super(Power_CellVoltage, self).__init__()

        self.n = n

        if not filename:
            fpath = os.path.dirname(os.path.realpath(__file__))
            filename = fpath + '/data/Power/curve.dat'

        dat = np.genfromtxt(filename)

        nT, nA, nI = dat[:3]
        nT = int(nT)
        nA = int(nA)
        nI = int(nI)
        T = dat[3:3 + nT]
        A = dat[3 + nT:3 + nT + nA]
        I = dat[3 + nT + nA:3 + nT + nA + nI]  # noqa: E741
        V = dat[3 + nT + nA + nI:].reshape((nT, nA, nI), order='F')

        self.MBI = MBI(V, [T, A, I], [6, 6, 15], [3, 3, 3])

        self.x = np.zeros((84 * n, 3), order='F')
        self.xV = self.x.reshape((n, 7, 12, 3), order='F')
        self.dV_dL = np.zeros((n, 12), order='F')
        self.dV_dT = np.zeros((n, 12, 5), order='F')
        self.dV_dA = np.zeros((n, 7, 12), order='F')
        self.dV_dI = np.zeros((n, 12), order='F')

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('LOS', np.zeros((n)), units=None,
                       desc='Line of Sight over Time')

        self.add_input('temperature', np.zeros((5, n)), units='degK',
                       desc='Temperature of solar cells over time')

        self.add_input('exposedArea', np.zeros((7, 12, n)), units='m**2',
                       desc='Exposed area to sun for each solar cell over time')

        self.add_input('Isetpt', np.zeros((12, n)), units='A',
                       desc='Currents of the solar panels')

        # Outputs
        self.add_output('V_sol', np.zeros((12, n)), units='V',
                        desc='Output voltage of solar panel over time')

    def setx(self, inputs):
        temperature = inputs['temperature']
        LOS = inputs['LOS']
        exposedArea = inputs['exposedArea']
        Isetpt = inputs['Isetpt']

        for p in range(12):
            i = 4 if p < 4 else (p % 4)
            for c in range(7):
                self.xV[:, c, p, 0] = temperature[i, :]
                self.xV[:, c, p, 1] = LOS * exposedArea[c, p, :]
                self.xV[:, c, p, 2] = Isetpt[p, :]

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        self.setx(inputs)
        self.raw = self.MBI.evaluate(self.x)[:, 0].reshape((self.n, 7, 12),
                                                           order='F')
        outputs['V_sol'] = np.zeros((12, self.n))
        for c in range(7):
            outputs['V_sol'] += self.raw[:, c, :].T

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        exposedArea = inputs['exposedArea']
        LOS = inputs['LOS']

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
                self.dV_dL[:, p] += self.raw2[:, c, p] * exposedArea[c, p, :]
                self.dV_dT[:, p, i] += self.raw1[:, c, p]
                self.dV_dA[:, c, p] += self.raw2[:, c, p] * LOS
                self.dV_dI[:, p] += self.raw3[:, c, p]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dV_sol = d_outputs['V_sol']

        if mode == 'fwd':
            if 'LOS' in d_inputs:
                dV_sol += self.dV_dL.T * d_inputs['LOS']

            if 'temperature' in d_inputs:
                for p in range(12):
                    i = 4 if p < 4 else (p % 4)
                    dV_sol[p, :] += self.dV_dT[:, p, i] * d_inputs['temperature'][i, :]

            if 'Isetpt' in d_inputs:
                dV_sol += self.dV_dI.T * d_inputs['Isetpt']

            if 'exposedArea' in d_inputs:
                for p in range(12):
                    dV_sol[p, :] += \
                        np.sum(self.dV_dA[:, :, p] * d_inputs['exposedArea'][:, p, :].T, 1)
        else:
            for p in range(12):
                i = 4 if p < 4 else (p % 4)

                if 'LOS' in d_inputs:
                    d_inputs['LOS'] += self.dV_dL[:, p] * dV_sol[p, :]

                if 'temperature' in d_inputs:
                    d_inputs['temperature'][i, :] += self.dV_dT[:, p, i] * dV_sol[p, :]

                if 'Isetpt' in d_inputs:
                    d_inputs['Isetpt'][p, :] += self.dV_dI[:, p] * dV_sol[p, :]

                if 'exposedArea' in d_inputs:
                    dexposedArea = d_inputs['exposedArea']
                    for c in range(7):
                        dexposedArea[c, p, :] += self.dV_dA[:, c, p] * dV_sol[p, :]


class Power_SolarPower(ExplicitComponent):
    """
    Compute the output power of the solar panels.
    """

    def __init__(self, n=2):
        super(Power_SolarPower, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('Isetpt', np.zeros((12, n)), units='A',
                       desc='Currents of the solar panels')

        self.add_input('V_sol', np.zeros((12, n)), units='V',
                       desc='Output voltage of solar panel over time')

        self.add_output('P_sol', np.zeros((n, )), units='W',
                        desc='Solar panels power over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        V_sol = inputs['V_sol']
        Isetpt = inputs['Isetpt']
        P_sol = outputs['P_sol']

        P_sol[:] = np.zeros((self.n, ))
        for p in range(12):
            P_sol += V_sol[p, :] * Isetpt[p, :]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dP_sol = d_outputs['P_sol']

        if mode == 'fwd':
            if 'V_sol' in d_inputs:
                for p in range(12):
                    dP_sol += d_inputs['V_sol'][p, :] * inputs['Isetpt'][p, :]

            if 'Isetpt' in d_inputs:
                for p in range(12):
                    dP_sol += d_inputs['Isetpt'][p, :] * inputs['V_sol'][p, :]
        else:
            for p in range(12):
                if 'V_sol' in d_inputs:
                    d_inputs['V_sol'][p, :] += dP_sol * inputs['Isetpt'][p, :]

                if 'Isetpt' in d_inputs:
                    d_inputs['Isetpt'][p, :] += inputs['V_sol'][p, :] * dP_sol


class Power_Total(ExplicitComponent):
    """
    Compute the battery power which is the sum of the loads. This
    includes a 2-Watt constant power usage that accounts for the scientific
    instruments on the satellite and small actuator inputs in response to
    disturbance torques.
    """

    def __init__(self, n=2):
        super(Power_Total, self).__init__()

        self.n = n

    def setup(self):
        n = self.n

        # Inputs
        self.add_input('P_sol', np.zeros((n, ), order='F'), units='W',
                       desc='Solar panels power over time')

        self.add_input('P_comm', np.zeros((n, ), order='F'), units='W',
                       desc='Communication power over time')

        self.add_input('P_RW', np.zeros((3, n, ), order='F'), units='W',
                       desc='Power used by reaction wheel over time')

        # Outputs
        self.add_output('P_bat', np.zeros((n, ), order='F'), units='W',
                        desc='Battery power over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        outputs['P_bat'] = inputs['P_sol'] - 5.0*inputs['P_comm'] - \
            np.sum(inputs['P_RW'], 0) - 2.0

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dP_bat = d_outputs['P_bat']

        if mode == 'fwd':
            if 'P_sol' in d_inputs:
                dP_bat += d_inputs['P_sol']

            if 'P_comm' in d_inputs:
                dP_bat -= 5.0 * d_inputs['P_comm']

            if 'P_RW' in d_inputs:
                for k in range(3):
                    dP_bat -= d_inputs['P_RW'][k, :]

        else:
            if 'P_sol' in d_inputs:
                d_inputs['P_sol'] += dP_bat[:]

            if 'P_comm' in d_inputs:
                d_inputs['P_comm'] -= 5.0 * dP_bat

            if 'P_RW' in d_inputs:
                for k in range(3):
                    d_inputs['P_RW'][k, :] -= dP_bat
