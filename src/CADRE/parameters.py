''' Bspline module for CADRE '''

from six.moves import range
import numpy as np

from openmdao.core.component import Component

from MBI.MBI import MBI


class BsplineParameters(Component):
    '''Creates a Bspline interpolant for several CADRE variables
       so that their time histories can be shaped with m control points
       instead of n time points.'''

    def __init__(self, n, m):
        super(BsplineParameters, self).__init__()

        self.n = n
        self.m = m

        # Inputs
        self.add_param('t1', 0., units='s', desc='Start time')

        self.add_param('t2', 43200., units='s', desc='End time')

        self.add_param('CP_P_comm', np.zeros((self.m, )), units='W',
                       desc='Communication power at the control points')

        self.add_param('CP_gamma', np.zeros((self.m, )), units='rad',
                       desc='Satellite roll angle at control points')

        self.add_param('CP_Isetpt', np.zeros((12,self.m)), units='A',
                       desc='Currents of the solar panels at the control points')

        # Outputs
        self.add_output('P_comm', np.ones((n, )), units='W',
                        desc='Communication power over time')

        self.add_output('Gamma', 0.1*np.ones((n,)), units='rad',
                        desc='Satellite roll ang le over time')

        self.add_output('Isetpt', 0.2*np.ones((12, n)), units="A",
                        desc="Currents of the solar panels over time")

        self.deriv_cached = False

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        # Only need to do this once.
        if self.deriv_cached == False:

            t1 = params['t1']
            t2 = params['t2']
            n = self.n

            self.B = MBI(np.zeros(n), [np.linspace(t1, t2, n)],
                         [self.m], [4]).getJacobian(0, 0)

            self.Bdot = MBI(np.zeros(n), [np.linspace(t1, t2, n)],
                            [self.m], [4]).getJacobian(1, 0)

            self.BT = self.B.transpose()
            self.BdotT = self.Bdot.transpose()

            self.deriv_cached = True

        unknowns['P_comm'] = self.B.dot(params['CP_P_comm'])
        unknowns['Gamma'] = self.B.dot(params['CP_gamma'])
        for k in range(12):
            unknowns['Isetpt'][k, :] = self.B.dot(params['CP_Isetpt'][k, :])

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        if mode == 'fwd':
            if 'P_comm' in dresids and 'CP_P_comm' in dparams:
                dresids['P_comm'] += self.B.dot(dparams['CP_P_comm'])

            if 'Gamma' in dresids and 'CP_gamma' in dparams:
                dresids['Gamma'] += self.B.dot(dparams['CP_gamma'])

            if 'Isetpt' in dresids and 'CP_Isetpt' in dparams:
                for k in range(12):
                    dresids['Isetpt'][k, :] += self.B.dot(dparams['CP_Isetpt'][k, :])

        else:
            if 'P_comm' in dresids and 'CP_P_comm' in dparams:
                dparams['CP_P_comm'] += self.BT.dot(dresids['P_comm'])

            if 'Gamma' in dresids and 'CP_gamma' in dparams:
                dparams['CP_gamma'] += self.BT.dot(dresids['Gamma'])

            if 'Isetpt' in dresids and 'CP_Isetpt' in dparams:
                for k in range(12):
                    dparams['CP_Isetpt'][k, :] += self.BT.dot(dresids['Isetpt'][k, :])
