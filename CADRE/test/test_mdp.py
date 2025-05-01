"""
Tests the values computed in one execution of the MDP problem and
compares them to saved values in a pickle. It also tests the derivatives.
This is set up so that it can be tested in parallel too.
"""
from __future__ import print_function

import sys
import os
import pickle
import unittest

from packaging.version import Version

import numpy as np

from openmdao import __version__ as om_version
from openmdao.api import Problem, PETScKrylov
from openmdao.utils.mpi import MPI
try:
    from openmdao.vectors.petsc_vector import PETScVector
except ImportError:
    PETScVector = None

from CADRE.CADRE_mdp import CADRE_MDP_Group


# set verbose to True for debugging
verbose = False

@unittest.skipUnless(MPI and PETScVector, "MPI and PETSc are required.")
class TestCADRE(unittest.TestCase):

    N_PROCS = 2

    def test_CADRE_MDP(self):
        # read pickles
        fpath = os.path.dirname(os.path.realpath(__file__))
        if sys.version_info.major == 2:
            file1 = fpath + '/mdp_execute_py2.pkl'
            file2 = fpath + '/mdp_derivs_py2.pkl'
        else:
            file1 = fpath + '/mdp_execute.pkl'
            file2 = fpath + '/mdp_derivs.pkl'

        with open(file1, 'rb') as f:
            data = pickle.load(f)

        with open(file2, 'rb') as f:
            Ja = pickle.load(f)

        n = 1500
        m = 300
        npts = 2

        # instantiate
        model = CADRE_MDP_Group(n=n, m=m, npts=npts)

        # add design variables and constraints to each CADRE instance.
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):
            model.add_design_var('%s.CP_Isetpt' % name, lower=0., upper=0.4)
            model.add_design_var('%s.CP_gamma' % name, lower=0, upper=np.pi/2.)
            model.add_design_var('%s.CP_P_comm' % name, lower=0., upper=25.)
            model.add_design_var('%s.iSOC' % name, indices=[0], lower=0.2, upper=1.)

            model.add_constraint('%s.ConCh' % name, upper=0.0)
            model.add_constraint('%s.ConDs' % name, upper=0.0)
            model.add_constraint('%s.ConS0' % name, upper=0.0)
            model.add_constraint('%s.ConS1' % name, upper=0.0)
            model.add_constraint('%s_con5.val' % name, equals=0.0)

        # add broadcast parameters
        model.add_design_var('bp.cellInstd', lower=0., upper=1.0)
        model.add_design_var('bp.finAngle', lower=0., upper=np.pi/2.)
        model.add_design_var('bp.antAngle', lower=-np.pi/4, upper=np.pi/4)

        # add objective
        model.add_objective('obj.val')

        # for parallel execution, use KSP
        if MPI:
            model.linear_solver = PETScKrylov()

        # create problem
        prob = Problem(model)
        prob.setup(check=verbose)
        prob.run_driver()

        # a change in OpenMDAO 3.38.1-dev adds a resolver in place of the prom2abs/abs2prom attributes
        try:
            resolver = model._resolver

            def get_abs_name(prom_name):
                print(f"prom_name: {prom_name} -> {resolver.prom2abs(prom_name, 'output')}")
                return resolver.prom2abs(prom_name, 'output')

            def get_prom_name(abs_name):
                print(f"abs_name: {abs_name} -> {resolver.abs2prom(abs_name, 'output')}")
                return resolver.prom2abs(abs_name, 'output')

        except AttributeError:
            def get_abs_name(prom_name):
                print(f"prom_name: {prom_name} -> {model._var_allprocs_prom2abs_list['output'][prom_name][0]}")
                return model._var_allprocs_prom2abs_list['output'][prom_name][0]

            def get_prom_name(abs_name):
                print(f"abs_name: {abs_name} -> {model._var_allprocs_abs2prom['output'][abs_name]}")
                return model._var_allprocs_abs2prom['output'][abs_name]

        # check output values
        checked = 0

        for var in data:
            # we changed constraint names
            xvar = var
            if '_con1' in xvar:
                xvar = xvar.replace('_con1.val', '.ConCh')
            if '_con2' in xvar:
                xvar = xvar.replace('_con2.val', '.ConDs')
            if '_con3' in xvar:
                xvar = xvar.replace('_con3.val', '.ConS0')
            if '_con4' in xvar:
                xvar = xvar.replace('_con4.val', '.ConS1')

            # make sure var is local before we try to look it up
            compname = get_abs_name(xvar).rsplit('.', 1)[0]
            comp = model._get_subsystem(compname)
            if comp and not (comp.comm is None or comp.comm == MPI.COMM_NULL):
                computed = prob[xvar]
                actual = data[var]
                if isinstance(computed, np.ndarray):
                    rel = np.linalg.norm(actual - computed) / np.linalg.norm(actual)
                else:
                    rel = np.abs(actual - computed) / np.abs(actual)

                if verbose:
                    print(xvar)
                    print(computed)
                    print(actual)

                if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                    assert rel <= 1e-3
                checked += 1

        # make sure we checked everything
        if MPI:
            # objective + con5 for both points + con1-4 on this proc
            self.assertEqual(checked, 7)
        else:
            # objective + 5 constraints for each point
            self.assertEqual(checked, 11)

        # check derivatives
        Jb = prob.compute_totals(debug_print=verbose)

        for key1, value in sorted(Ja.items()):
            for key2 in sorted(value.keys()):
                bkey1 = key1
                bkey2 = key2

                # we changed constraint names
                if '_con1' in bkey1:
                    bkey1 = bkey1.replace('_con1.val', '.ConCh')
                if '_con2' in bkey1:
                    bkey1 = bkey1.replace('_con2.val', '.ConDs')
                if '_con3' in bkey1:
                    bkey1 = bkey1.replace('_con3.val', '.ConS0')
                if '_con4' in bkey1:
                    bkey1 = bkey1.replace('_con4.val', '.ConS1')

                if bkey1.startswith('pt') and not bkey1.endswith('.val'):
                    # need full path of constraint vars in the parallel CADRE groups
                    parts = bkey1.split('.')
                    parts[0] = 'parallel.%s.BatteryConstraints' % parts[0]
                    bkey1 = '.'.join(parts)

                if bkey2.startswith('bp'):
                    # the 3 broadcast params were merged into a single IndepVarComp
                    parts = bkey2.split('.')
                    parts[0] = 'bp'
                    bkey2 = '.'.join(parts)
                elif bkey2.startswith('pt'):
                    # need full path of design vars in the parallel CADRE groups
                    parts = bkey2.split('.')
                    parts[0] = 'parallel.%s.design' % parts[0]
                    bkey2 = '.'.join(parts)

                actual = Ja[key1][key2]
                if Version(om_version) <= Version("3.30"):
                    computed = Jb[bkey1, bkey2]
                else:
                    # as of OpenMDAO 3.31.0, the keys in the jac are the 'user facing' names
                    # given to the design vars and responses, rather than the absolute names
                    # that were used previously
                    prom_bkey1 = get_prom_name(bkey1)
                    prom_bkey2 = get_prom_name(bkey2)
                    computed = Jb[prom_bkey1, prom_bkey2]

                if isinstance(computed, np.ndarray):
                    rel = np.linalg.norm(actual - computed)/np.linalg.norm(actual)
                else:
                    rel = np.abs(actual - computed)/np.abs(actual)

                if verbose:
                    print(bkey1, 'wrt', bkey2, '(', key1, 'wrt', key2, ')')
                    print(computed)
                    print(actual)

                if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                    assert rel <= 1e-3


if __name__ == '__main__':
    unittest.main()
