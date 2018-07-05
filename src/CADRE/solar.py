''' Solar discipline for CADRE '''

import os
from six.moves import range
import numpy as np

from openmdao.core.component import Component

from CADRE.kinematics import fixangles
from MBI import MBI

# Allow non-standard variable names for scientific calc
# pylint: disable-msg=C0103


class Solar_ExposedArea(Component):
    """Exposed area calculation for a given solar cell

    p: panel ID [0,11]
    c: cell ID [0,6]
    a: fin angle [0,90]
    z: azimuth [0,360]
    e: elevation [0,180]
    LOS: line of sight with the sun [0,1]
    """


    def __init__(self, n, raw1=None, raw2=None):
        super(Solar_ExposedArea, self).__init__()

        if raw1 is None:
            fpath = os.path.dirname(os.path.realpath(__file__))
            raw1 = np.genfromtxt(fpath + '/data/Solar/Area10.txt')
        if raw2 is None:
            fpath = os.path.dirname(os.path.realpath(__file__))
            raw2 = np.loadtxt(fpath + "/data/Solar/Area_all.txt")

        self.n = n
        self.nc = 7
        self.np = 12

        # Inputs
        self.add_input('finAngle', 0.0, units="rad",
                       desc="Fin angle of solar panel")

        self.add_input('azimuth', np.zeros((n, )), units='rad',
                       desc="Azimuth angle of the sun in the body-fixed frame "
                       "over time")

        self.add_input('elevation', np.zeros((n, )), units='rad',
                       desc='Elevation angle of the sun in the body-fixed '
                       'frame over time')

        # Outputs
        self.add_output('exposedArea', np.zeros((self.nc, self.np, self.n)),
                        desc="Exposed area to sun for each solar cell over time",
                        units='m**2', lower=-5e-3, upper=1.834e-1)

        self.na = 10
        self.nz = 73
        self.ne = 37
        angle = np.zeros(self.na)
        azimuth = np.zeros(self.nz)
        elevation = np.zeros(self.ne)

        index = 0
        for i in range(self.na):
            angle[i] = raw1[index]
            index += 1
        for i in range(self.nz):
            azimuth[i] = raw1[index]
            index += 1

        index -= 1
        azimuth[self.nz - 1] = 2.0 * np.pi
        for i in range(self.ne):
            elevation[i] = raw1[index]
            index += 1

        angle[0] = 0.0
        angle[-1] = np.pi / 2.0
        azimuth[0] = 0.0
        azimuth[-1] = 2 * np.pi
        elevation[0] = 0.0
        elevation[-1] = np.pi

        counter = 0
        data = np.zeros((self.na, self.nz, self.ne, self.np * self.nc))
        flat_size = self.na * self.nz * self.ne
        for p in range(self.np):
            for c in range(self.nc):
                data[:, :, :, counter] = \
                    raw2[7 * p + c][119:119 + flat_size].reshape((self.na,
                                                                  self.nz,
                                                                  self.ne))
                counter += 1

        self.MBI = MBI(data, [angle, azimuth, elevation],
                             [4, 10, 8],
                             [4, 4, 4])

        self.x = np.zeros((self.n, 3))
        self.Jfin = None
        self.Jaz = None
        self.Jel = None

    def solve_nonlinear(self, inputs, outputs, resids):
        """ Calculate output. """

        self.setx(inputs)
        P = self.MBI.evaluate(self.x).T
        outputs['exposedArea'] = P.reshape(7, 12, self.n, order='F')

    def setx(self, inputs):
        """ Sets our state array"""

        result = fixangles(self.n, inputs['azimuth'], inputs['elevation'])
        self.x[:, 0] = inputs['finAngle']
        self.x[:, 1] = result[0]
        self.x[:, 2] = result[1]

    def linearize(self, inputs, outputs, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        self.Jfin = self.MBI.evaluate(self.x, 1).reshape(self.n, 7, 12,
                                                         order='F')
        self.Jaz = self.MBI.evaluate(self.x, 2).reshape(self.n, 7, 12,
                                                        order='F')
        self.Jel = self.MBI.evaluate(self.x, 3).reshape(self.n, 7, 12,
                                                        order='F')

    def apply_linear(self, inputs, outputs, dinputs, doutputs, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        deA = dresids['exposedArea']

        if mode == 'fwd':
            for c in range(7):

                if 'finAngle' in dinputs:
                    deA[c, :, :] += \
                        self.Jfin[:, c, :].T * dinputs['finAngle']

                if 'azimuth' in dinputs:
                    deA[c, :, :] += \
                        self.Jaz[:, c, :].T * dinputs['azimuth']

                if 'elevation' in dinputs:
                    deA[c, :, :] += \
                        self.Jel[:, c, :].T * dinputs['elevation']

        else:
            for c in range(7):

                # incoming arg is often sparse, so check it first
                if len(np.nonzero(dresids['exposedArea'][c, :, :])[0]) == 0:
                    continue

                if 'finAngle' in dinputs:
                    dinputs['finAngle'] += \
                        np.sum(
                            self.Jfin[:, c, :].T * deA[c, :, :])

                if 'azimuth' in dinputs:
                    dinputs['azimuth'] += \
                        np.sum(
                            self.Jaz[:, c, :].T * deA[c, :, :], 0)

                if 'elevation' in dinputs:
                    dinputs['elevation'] += \
                        np.sum(
                            self.Jel[:, c, :].T * deA[c, :, :], 0)
