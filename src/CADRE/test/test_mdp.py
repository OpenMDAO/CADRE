"""
Tests the values computed in one execution of the MDP problem and
compares them to saved values in a pickle. It also tests the derivatives.
This is set up so that it can be tested in parallel too.
"""
from __future__ import print_function

import sys
import os
import time
import pickle
import unittest

import numpy as np

from openmdao.api import Problem, PETScKrylov
from openmdao.utils.mpi import MPI

from CADRE.CADRE_mdp import CADRE_MDP_Group


class TestCADRE(unittest.TestCase):

    N_PROCS = 2

    def test_CADRE_MDP(self):

        # Read pickles
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

        # Instantiate
        prob = Problem(CADRE_MDP_Group(n=n, m=m, npts=npts))
        model = prob.model

        # Add parameters and constraints to each CADRE instance.
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

        # Add Parameter groups
        model.add_design_var('bp.cellInstd', lower=0., upper=1.0)
        model.add_design_var('bp.finAngle', lower=0., upper=np.pi/2.)
        model.add_design_var('bp.antAngle', lower=-np.pi/4, upper=np.pi/4)

        # Add objective
        model.add_objective('obj.val')

        # For Parallel execution, we must use KSP
        if MPI:
            model.linear_solver = PETScKrylov()

        start_time = time.time()

        prob.setup(check=True)
        prob.run_driver()

        abs_u = model._var_allprocs_prom2abs_list['output']

        for var in data:
            # We changed constraint names
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
            compname = abs_u[xvar][0].rsplit('.', 1)[0]
            comp = model._get_subsystem(compname)
            if comp.is_active():
                computed = prob[xvar]
                actual = data[var]
                if isinstance(computed, np.ndarray):
                    rel = np.linalg.norm(actual - computed) / np.linalg.norm(actual)
                else:
                    rel = np.abs(actual - computed) / np.abs(actual)

                print(xvar)
                print(computed)
                print(actual)
                if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                    assert rel <= 1e-3

        # Now do derivatives
        Jb = prob.compute_totals(debug_print=True)

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

                if key2.startswith('bp'):
                    # the 3 broadcast params were merged into a single IndepVarComp
                    parts = bkey2.split('.')
                    parts[0] = 'bp'
                    bkey2 = '.'.join(parts)
                elif key2.startswith('pt'):
                    # need full path of design vars in the parallel CADRE groups
                    parts = bkey2.split('.')
                    parts[0] = 'parallel.%s.design' % parts[0]
                    bkey2 = '.'.join(parts)

                actual = Ja[key1][key2]
                computed = Jb[bkey1, bkey2]
                if isinstance(computed, np.ndarray):
                    rel = np.linalg.norm(actual - computed)/np.linalg.norm(actual)
                else:
                    rel = np.abs(actual - computed)/np.abs(actual)

                print(bkey1, 'wrt', key2)
                print(computed)
                print(actual)
                if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                    assert rel <= 1e-3

        print('\n\nElapsed time:', time.time()-start_time)


if __name__ == '__main__':
    unittest.main()
