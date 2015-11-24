""" KS Function Component. """

import numpy as np

from openmdao.core.component import Component
from openmdao.util.options import OptionsDictionary

# Allow non-standard variable names for scientific calc
# pylint: disable-msg=C0103


class KSfunction(object):
    """Helper class that can be used inside other components to aggregate
    constraint vectors with a KS function."""

    def compute(self, g, rho=50):
        """Gets the value of the KS function for the given array of
        constraints."""

        self.rho = rho
        self.g_max = np.max(g)
        self.g_diff = g-self.g_max
        self.exponents = np.exp(rho * self.g_diff)
        self.summation = np.sum(self.exponents)
        self.KS = self.g_max + 1.0/rho * np.log(self.summation)

        return self.KS

    def derivatives(self):
        """returns a row vector of [dKS_gd, dKS_drho]"""
        dsum_dg = self.rho*self.exponents
        dKS_dsum = 1.0/self.rho/self.summation
        self.dKS_dg = dKS_dsum * dsum_dg

        dsum_drho = np.sum(self.g_diff*self.exponents)
        self.dKS_drho = dKS_dsum * dsum_drho

        return self.dKS_dg, self.dKS_drho


class KSComp(Component):
    """Aggregates a number of functions to a single value via the
    Kreisselmeier-Steinhauser Function."""

    def __init__(self, n=2):
        super(KS, self).__init__()

        self.n = n

        # Inputs
        self.add_param('g', np.zeros((n, )),
                       desc="Array of function values to be aggregated")

        # Outputs
        self.add_output('KS', 0.0,
                        desc="Value of the aggregate KS function")

        self.options = OptionsDictionary()
        self.options.add_option(rho, 0.1,
                                desc="Hyperparameter for the KS function")

        self._ks = KSfunction()

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        unknowns['KS'] = self._ks.compute(params['g'], self.options['rho'])

    def linearize(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        #use g_max, exponsnte, summation from last executed point
        J = {}
        J['KS', 'g'] = np.hstack(self._ks.derivatives())
