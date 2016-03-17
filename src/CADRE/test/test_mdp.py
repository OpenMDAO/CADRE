""" Tests the values computed in one execution of the MDP problem and
compares them to saved values in a pickle. It also tests the derivatives.
This is set up so that it can be tested in parallel too."""

from __future__ import print_function

import sys
import os
import pickle
import unittest

import numpy as np

from openmdao.core.mpi_wrap import MPI
from openmdao.core.problem import Problem
from openmdao.solvers.petsc_ksp import PetscKSP
from openmdao.test.mpi_util import MPITestCase

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl

from CADRE.CADRE_mdp import CADRE_MDP_Group


class CADREMDPTests(MPITestCase):

    N_PROCS = 2

    def test_CADRE_MDP(self):

        # Read pickles
        fpath = os.path.dirname(os.path.realpath(__file__))
        if sys.version_info.major == 2:
            file1 = fpath + "/mdp_execute_py2.pkl"
            file2 = fpath + "/mdp_derivs_py2.pkl"
        else:
            file1 = fpath + "/mdp_execute.pkl"
            file2 = fpath + "/mdp_derivs.pkl"

        with open(file1, 'rb') as f:
            data = pickle.load(f)

        with open(file2, 'rb') as f:
            Ja = pickle.load(f)

        n = 1500
        m = 300
        npts = 2

        # Instantiate
        model = Problem(impl=impl)
        root = model.root = CADRE_MDP_Group(n=n, m=m, npts=npts)

        # Add parameters and constraints to each CADRE instance.
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):

            # add parameters to driver
            model.driver.add_desvar("%s.CP_Isetpt" % name, lower=0., upper=0.4)
            model.driver.add_desvar("%s.CP_gamma" % name, lower=0, upper=np.pi/2.)
            model.driver.add_desvar("%s.CP_P_comm" % name, lower=0., upper=25.)
            model.driver.add_desvar("%s.iSOC" % name, indices=[0], lower=0.2, upper=1.)

            model.driver.add_constraint('%s.ConCh'% name, upper=0.0)
            model.driver.add_constraint('%s.ConDs'% name, upper=0.0)
            model.driver.add_constraint('%s.ConS0'% name, upper=0.0)
            model.driver.add_constraint('%s.ConS1'% name, upper=0.0)
            model.driver.add_constraint('%s_con5.val'% name, equals=0.0)

        # Add Parameter groups
        model.driver.add_desvar("bp1.cellInstd", lower=0., upper=1.0)
        model.driver.add_desvar("bp2.finAngle", lower=0., upper=np.pi/2.)
        model.driver.add_desvar("bp3.antAngle", lower=-np.pi/4, upper=np.pi/4)

        # Add objective
        model.driver.add_objective('obj.val')

        # For Parallel exeuction, we must use KSP
        if MPI:
            model.root.ln_solver = PetscKSP()

        model.setup(check=False)
        model.run()

        abs_u = model.root._sysdata.to_abs_uname

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

            compname = abs_u[xvar].rsplit('.', 1)[0]
            comp = model.root._subsystem(compname)
            if comp.is_active():
                computed = model[xvar]
                actual = data[var]
                if isinstance(computed, np.ndarray):
                    rel = np.linalg.norm(actual - computed)/np.linalg.norm(actual)
                else:
                    rel = np.abs(actual - computed)/np.abs(actual)

                print(xvar)
                print(computed)
                print(actual)
                if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                    assert rel <= 1e-3

        # Now do derivatives
        params = list(model.driver.get_desvars().keys())
        unks = list(model.driver.get_objectives().keys()) + list(model.driver.get_constraints().keys())
        Jb = model.calc_gradient(params, unks, mode='rev', return_format='dict')

        for key1, value in sorted(Ja.items()):
            for key2 in sorted(value.keys()):
                # We changed constraint names
                xkey1 = key1
                if '_con1' in xkey1:
                    xkey1 = xkey1.replace('_con1.val', '.ConCh')
                if '_con2' in xkey1:
                    xkey1 = xkey1.replace('_con2.val', '.ConDs')
                if '_con3' in xkey1:
                    xkey1 = xkey1.replace('_con3.val', '.ConS0')
                if '_con4' in xkey1:
                    xkey1 = xkey1.replace('_con4.val', '.ConS1')

                computed = Jb[xkey1][key2]
                actual = Ja[key1][key2]
                if isinstance(computed, np.ndarray):
                    rel = np.linalg.norm(actual - computed)/np.linalg.norm(actual)
                else:
                    rel = np.abs(actual - computed)/np.abs(actual)

                print(xkey1, 'wrt', key2)
                print(computed)
                print(actual)
                if np.mean(actual) > 1e-3 or np.mean(computed) > 1e-3:
                    assert rel <= 1e-3

if __name__ == '__main__':
    from openmdao.test.mpi_util import mpirun_tests
    #mpirun_tests()
    unittest.main()
