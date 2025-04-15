
from openmdao.core.explicitcomponent import ExplicitComponent as om_ExplicitComponent

class ExplicitComponent(om_ExplicitComponent):
    # override _linearize to get the old OpenMDAO behavior where a component could have
    # matrix free and jacobian methods.
    def _linearize(self, jac=None, sub_do_ln=False):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.

        Parameters
        ----------
        jac : Jacobian or None
            Ignored.
        sub_do_ln : bool
            Flag indicating if the children should call linearize on their linear solvers.
        """
        save = self.matrix_free
        self.matrix_free = False
        try:
            super()._linearize(jac, sub_do_ln)
        finally:
            self.matrix_free = save
