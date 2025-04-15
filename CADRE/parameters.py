"""
Bspline module for CADRE
"""

from six.moves import range
import numpy as np

from CADRE.explicit import ExplicitComponent

from MBI import MBI


class BsplineParameters(ExplicitComponent):
    """
    Creates a Bspline interpolant for several CADRE variables
    so that their time histories can be shaped with m control points
    instead of n time points.
    """

    def __init__(self, n, m):
        super(BsplineParameters, self).__init__()

        self.n = n
        self.m = m

        self.deriv_cached = False

    def setup(self):
        m = self.m
        n = self.n

        # Inputs
        self.add_input('t1', 0., units='s', desc='Start time')

        self.add_input('t2', 43200., units='s', desc='End time')

        self.add_input('CP_P_comm', np.zeros((m, )), units='W',
                       desc='Communication power at the control points')

        self.add_input('CP_gamma', np.zeros((m, )), units='rad',
                       desc='Satellite roll angle at control points')

        self.add_input('CP_Isetpt', np.zeros((12, m)), units='A',
                       desc='Currents of the solar panels at the control points')

        # Outputs
        self.add_output('P_comm', np.ones((n, )), units='W',
                        desc='Communication power over time')

        self.add_output('Gamma', 0.1*np.ones((n,)), units='rad',
                        desc='Satellite roll angle over time')

        self.add_output('Isetpt', 0.2*np.ones((12, n)), units='A',
                        desc='Currents of the solar panels over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        # Only need to do this once.
        if self.deriv_cached is False:
            t1 = inputs['t1']
            t2 = inputs['t2']
            n = self.n

            self.B = MBI(np.zeros(n), [np.linspace(t1, t2, n)], [self.m], [4]).getJacobian(0, 0)
            self.Bdot = MBI(np.zeros(n), [np.linspace(t1, t2, n)], [self.m], [4]).getJacobian(1, 0)

            self.BT = self.B.transpose()
            self.BdotT = self.Bdot.transpose()

            self.deriv_cached = True

        outputs['P_comm'] = self.B.dot(inputs['CP_P_comm'])
        outputs['Gamma'] = self.B.dot(inputs['CP_gamma'])
        for k in range(12):
            outputs['Isetpt'][k, :] = self.B.dot(inputs['CP_Isetpt'][k, :])

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        if mode == 'fwd':
            if 'P_comm' in d_outputs and 'CP_P_comm' in d_inputs:
                d_outputs['P_comm'] += self.B.dot(d_inputs['CP_P_comm'])

            if 'Gamma' in d_outputs and 'CP_gamma' in d_inputs:
                d_outputs['Gamma'] += self.B.dot(d_inputs['CP_gamma'])

            if 'Isetpt' in d_outputs and 'CP_Isetpt' in d_inputs:
                for k in range(12):
                    d_outputs['Isetpt'][k, :] += self.B.dot(d_inputs['CP_Isetpt'][k, :])
        else:
            if 'P_comm' in d_outputs and 'CP_P_comm' in d_inputs:
                d_inputs['CP_P_comm'] += self.BT.dot(d_outputs['P_comm'])

            if 'Gamma' in d_outputs and 'CP_gamma' in d_inputs:
                d_inputs['CP_gamma'] += self.BT.dot(d_outputs['Gamma'])

            if 'Isetpt' in d_outputs and 'CP_Isetpt' in d_inputs:
                for k in range(12):
                    d_inputs['CP_Isetpt'][k, :] += self.BT.dot(d_outputs['Isetpt'][k, :])
