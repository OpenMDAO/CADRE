""" RK4 time integration component """

from six.moves import range

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from openmdao.core.component import Component
from openmdao.util.options import OptionsDictionary

# Allow non-standard variable names for scientific calc
# pylint: disable-msg=C0103


class RK4(Component):
    """Inherit from this component to use.

    State variable dimension: (num_states, num_time_points)
    External input dimension: (input width, num_time_points)
    """


    def __init__(self, n=2, h=.01):
        super(RK4, self).__init__()

        self.h = h

        # Inputs
        # All inputs are defined in subclasses.

        # Options
        self.options = opt = OptionsDictionary()
        opt.add_option('state_var', '',
                       desc="Name of the variable to be used for time "
                       "integration")
        opt.add_option('init_state_var', '',
                       desc="Name of the variable to be used for initial "
                       "conditions")
        opt.add_option('external_vars', [],
                       desc="List of names of variables that are external "
                       "to the system but DO vary with time.")
        opt.add_option('fixed_external_vars', [],
                       desc="List of names of variables that are "
                       "external to the system but DO NOT "
                       "vary with time.")

    def initialize(self, params, unknowns):
        """Set up dimensions and other data structures."""

        state_var = self.options['state_var']
        init_state_var = self.options['init_state_var']
        external_vars = self.options['external_vars']
        fixed_external_vars = self.options['fixed_external_vars']

        self.y = unknowns[state_var]
        self.y0 = params[init_state_var]


        self.n_states, self.n = self.y.shape
        self.ny = self.n_states*self.n
        self.nJ = self.n_states*(self.n + self.n_states*(self.n-1))

        ext = []
        self.ext_index_map = {}
        for name in external_vars:
            var = params[name]
            self.ext_index_map[name] = len(ext)

            #TODO: Check that shape[-1]==self.n
            ext.extend(var.reshape(-1, self.n))


        for name in fixed_external_vars:
            var = params[name]
            self.ext_index_map[name] = len(ext)

            flat_var = var.flatten()
            #create n copies of the var
            ext.extend(np.tile(flat_var, (self.n, 1)).T)

        self.external = np.array(ext)

        #TODO
        #check that len(y0) = self.n_states

        self.n_external = len(ext)
        self.reverse_name_map = {
            state_var:'y',
            init_state_var:'y0'
        }
        e_vars = np.hstack((external_vars, fixed_external_vars))
        for i, var in enumerate(e_vars):
            self.reverse_name_map[var] = i

        self.name_map = dict([(v, k) for k, v in
                              self.reverse_name_map.items()])

        #TODO
        #  check that all ext arrays of of shape (self.n, )

        #TODO
        #check that length of state var and external
        # vars are the same length

    def f_dot(self, external, state):
        """Time rate of change of state variables. This must be overridden in
        derived classes.

        Args
        ----
        external: ndarray
            array of external variables for a single time step

        state: ndarray
            array of state variables for a single time step.
        """
        raise NotImplementedError

    def df_dy(self, external, state):
        """Derivatives of states with respect to states. This must be
        overridden in derived classes.

        Args
        ----
        external: ndarray
            array or external variables for a single time step

        state: ndarray
            array of state variables for a single time step.
        """

        raise NotImplementedError

    def df_dx(self, external, state):
        """derivatives of states with respect to external vars external:
        array or external variables for a single time step state: array of
        state variables for a single time step. This must be overridden in
        derived classes.

        Args
        ----
        external: ndarray
            array or external variables for a single time step

        state: ndarray
            array of state variables for a single time step.
        """
        raise NotImplementedError

    def solve_nonlinear(self, params, unknowns, resids):
        """ Calculate output. """

        self.initialize(params, unknowns)

        n_state = self.n_states
        n_time = self.n
        h = self.h

        # Copy initial state into state array for t=0
        self.y = self.y.reshape((self.ny, ), order='f')
        self.y[0:n_state] = self.y0

        # Cache f_dot for use in linearize()
        size = (n_state, self.n)
        self.a = np.zeros(size)
        self.b = np.zeros(size)
        self.c = np.zeros(size)
        self.d = np.zeros(size)

        for k in range(0, n_time-1):
            k1 = (k)*n_state
            k2 = (k+1)*n_state

            # Next state a function of current input
            ex = self.external[:, k] if self.external.shape[0] \
                                       else np.array([])

            # Next state a function of previous state
            y = self.y[k1:k2]

            self.a[:, k] = a = self.f_dot(ex, y)
            self.b[:, k] = b = self.f_dot(ex, y + h/2.*a)
            self.c[:, k] = c = self.f_dot(ex, y + h/2.*b)
            self.d[:, k] = d = self.f_dot(ex, y + h*c)

            self.y[n_state+k1:n_state+k2] = \
                y + h/6.*(a + 2*(b + c) + d)

        state_var_name = self.name_map['y']
        unknowns[state_var_name][:] = self.y.T.reshape((n_time, n_state)).T

    def jacobian(self, params, unknowns, resids):
        """ Calculate and save derivatives. (i.e., Jacobian) """

        n_state = self.n_states
        n_time = self.n
        h = self.h
        I = np.eye(n_state)

        # Full Jacobian with respect to states
        self.Jy = np.zeros((self.n, self.n_states, self.n_states))

        # Full Jacobian with respect to inputs
        self.Jx = np.zeros((self.n, self.n_external, self.n_states))

        for k in range(0, n_time-1):

            k1 = k*n_state
            k2 = k1 + n_state

            ex = self.external[:, k] if self.external.shape[0] \
                                        else np.array([])

            y = self.y[k1:k2]

            a = self.a[:, k]
            b = self.b[:, k]
            c = self.c[:, k]

            # State vars
            df_dy = self.df_dy(ex, y)
            dg_dy = self.df_dy(ex, y + h/2.*a)
            dh_dy = self.df_dy(ex, y + h/2.*b)
            di_dy = self.df_dy(ex, y + h*c)

            da_dy = df_dy
            db_dy = dg_dy + dg_dy.dot(h/2.*da_dy)
            dc_dy = dh_dy + dh_dy.dot(h/2.*db_dy)
            dd_dy = di_dy + di_dy.dot(h*dc_dy)

            dR_dy = -I - h/6.*(da_dy + 2*(db_dy + dc_dy) + dd_dy)
            self.Jy[k, :, :] = dR_dy

            # External vars (Inputs)
            df_dx = self.df_dx(ex, y)
            dg_dx = self.df_dx(ex, y + h/2.*a)
            dh_dx = self.df_dx(ex, y + h/2.*b)
            di_dx = self.df_dx(ex, y + h*c)

            da_dx = df_dx
            db_dx = dg_dx + dg_dy.dot(h/2*da_dx)
            dc_dx = dh_dx + dh_dy.dot(h/2*db_dx)
            dd_dx = di_dx + di_dy.dot(h*dc_dx)

            # Input-State Jacobian at each time point.
            # No Jacobian with respect to previous time points.
            self.Jx[k+1, :, :] = h/6*(da_dx + 2*(db_dx + dc_dx) + dd_dx).T

    def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
        """ Matrix-vector product with the Jacobian. """

        if mode == 'fwd':
            result_ext = self._applyJext(dparams, dresids)

            svar = self.options['state_var']
            dresids[svar] += result_ext

        else:
            r2 = self._applyJextT_limited(dparams, dresids)

            for k, v in r2.iteritems():
                dparams[k] += v

    def _applyJext(self, dparams, dresids):
        """Apply derivatives with respect to inputs"""

        #Jx --> (n_times, n_external, n_states)
        n_state = self.n_states
        n_time = self.n
        result = np.zeros((n_state, n_time))

        # Time-varying inputs
        for name in self.options['external_vars']:

            if name not in dparams:
                continue

            # Take advantage of fact that arg is often pretty sparse
            dvar = dparams[name]
            if len(np.nonzero(dvar)[0]) == 0:
                continue

            # Collapse incoming a*b*...*c*n down to (ab...c)*n
            shape = dvar.shape
            dvar = dvar.reshape((np.prod(shape[:-1]), shape[-1]))

            i_ext = self.ext_index_map[name]
            ext_length = np.prod(dvar[:, 0].shape)
            for j in range(n_time-1):
                Jsub = self.Jx[j+1, i_ext:i_ext+ext_length, :]
                J_arg = Jsub.T.dot(dvar[:, j])
                result[:, j+1:n_time] += np.tile(J_arg, (n_time-j-1, 1)).T

        # Time-invariant inputs
        for name in self.options['fixed_external_vars']:

            if name not in dparams:
                continue

            # Take advantage of fact that arg is often pretty sparse
            dvar = dparams[name]
            if len(np.nonzero(dvar)[0]) == 0:
                continue

            if len(dvar) > 1:
                dvar = dvar.flatten()
            i_ext = self.ext_index_map[name]
            ext_length = np.prod(dvar.shape)
            for j in range(n_time-1):
                Jsub = self.Jx[j+1, i_ext:i_ext+ext_length, :]
                J_arg = Jsub.T.dot(dvar)
                result[:, j+1:n_time] += np.tile(J_arg, (n_time-j-1, 1)).T

        # Initial State
        name = self.options['init_state_var']
        if name in dparams:

            # Take advantage of fact that arg is often pretty sparse
            dvar = dparams[name]
            if len(np.nonzero(dvar)[0]) > 0:
                fact = np.eye(self.n_states)
                result[:, 0] = dvar
                for j in range(1, n_time):
                    fact = fact.dot(-self.Jy[j-1, :, :])
                    result[:, j] += fact.dot(dvar)

        return result

    def _applyJextT_limited(self, dparams, dresids):
        """Apply derivatives with respect to inputs"""

        # Jx --> (n_times, n_external, n_states)
        n_time = self.n
        result = {}

        argsv = dresids[self.options['state_var']].T
        argsum = np.zeros(argsv.shape)

        # Calculate these once, and use for every output
        for k in range(n_time - 2, -1, -1):
            argsum[k, :] = argsum[k+1, :] + argsv[k+1, :]

        # argsum is often sparse, so save indices.
        nonzero_k = np.unique(argsum.nonzero()[0])

        # Time-varying inputs
        for name in self.options['external_vars']:

            if name not in dparams:
                continue

            dext_var = dparams[name]
            i_ext = self.ext_index_map[name]
            ext_length = np.prod(dext_var.shape) / n_time
            result[name] = np.zeros((ext_length, n_time))

            i_ext_end = i_ext + ext_length
            for k in nonzero_k:
                Jsub = self.Jx[k + 1, i_ext:i_ext_end, :]
                result[name][:, k] += Jsub.dot(argsum[k, :])

        # Time-invariant inputs
        for name in self.options['fixed_external_vars']:

            if name not in dparams:
                continue

            di_ext_var = dparams[name]
            i_ext = self.ext_index_map[name]
            ext_length = np.prod(di_ext_var.shape)
            result[name] = np.zeros((ext_length))

            i_ext_end = i_ext + ext_length
            for k in nonzero_k:
                Jsub = self.Jx[k + 1, i_ext:i_ext_end, :]
                result[name] += Jsub.dot(argsum[k, :])

        # Initial State
        name = self.options['init_state_var']
        if name in dparams:
            fact = -self.Jy[0, :, :].T
            result[name] = argsv[0, :] + fact.dot(argsv[1, :])
            for k in range(1, n_time-1):
                fact = fact.dot(-self.Jy[k, :, :].T)
                result[name] += fact.dot(argsv[k+1, :])

        for name, val in result.iteritems():
            dvar = dparams[name]
            result[name] = val.reshape(dvar.shape)

        return result
