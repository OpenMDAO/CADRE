"""
Optimization of the CADRE MDP.
"""
from __future__ import print_function

from six.moves import range

import resource

import numpy as np

from openmdao.api import Problem, pyOptSparseDriver, LinearBlockGS, SqliteRecorder
from openmdao.utils.mpi import MPI

from CADRE.CADRE_mdp import CADRE_MDP_Group

import pickle


import sys
argv = sys.argv[1:]

if 'paper' in argv:
    # These numbers are for the CADRE problem in the paper.
    n = 1500
    m = 300
    npts = 6
    print("Using parameters from paper:", n, m, npts)
else:
    # These numbers are for quick testing
    n = 150
    m = 6
    npts = 2
    print("Using parameters for quick test:", n, m, npts)


# Instantiate CADRE model
model = CADRE_MDP_Group(n=n, m=m, npts=npts)

# Add design variables and constraints to each CADRE instance.
names = ['pt%s' % i for i in range(npts)]
for i, name in enumerate(names):
    model.add_design_var('%s.CP_Isetpt' % name, lower=0., upper=0.4)
    model.add_design_var('%s.CP_gamma' % name, lower=0, upper=np.pi/2.)
    model.add_design_var('%s.CP_P_comm' % name, lower=0., upper=25.)
    model.add_design_var('%s.iSOC' % name, indices=[0], lower=0.2, upper=1.)

    model.add_constraint('%s.ConCh' % name, upper=0.0, parallel_deriv_color=i)
    model.add_constraint('%s.ConDs' % name, upper=0.0, parallel_deriv_color=i)
    model.add_constraint('%s.ConS0' % name, upper=0.0, parallel_deriv_color=i)
    model.add_constraint('%s.ConS1' % name, upper=0.0, parallel_deriv_color=i)
    model.add_constraint('%s_con5.val' % name, equals=0.0)

# Add broadcast parameters
model.add_design_var('bp.cellInstd', lower=0., upper=1.0)
model.add_design_var('bp.finAngle', lower=0., upper=np.pi/2.)
model.add_design_var('bp.antAngle', lower=-np.pi/4, upper=np.pi/4)

# Add objective
model.add_objective('obj.val')

# Instantiate Problem with driver (SNOPT) and recorder
prob = Problem(model)

prob.driver = pyOptSparseDriver(optimizer='SNOPT')
prob.driver.opt_settings = {
    'Major optimality tolerance': 1e-3,
    'Major feasibility tolerance': 1.0e-5,
    'Iterations limit': 500000000
}

if not MPI and 'record' in argv:
    prob.driver.add_recorder(SqliteRecorder('data.sql'))

prob.setup()

# For Parallel execution, we must use KSP or LinearGS
# prob.model.linear_solver = PETScKrylov()
model.linear_solver = LinearBlockGS()
model.parallel.linear_solver = LinearBlockGS()
for pt in range(npts):
    model._get_subsystem('parallel.pt%d' % pt).linear_solver = LinearBlockGS()

prob.set_solver_print(0)
prob.run_driver()

print('Memory Usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0, 'MB (on unix)')

# ----------------------------------------------------------------
# Below this line, code used for verifying and profiling.
# ----------------------------------------------------------------

if 'profile' in argv:
    import cProfile
    import pstats

    def zzz():
        for j in range(1):
            prob.run()

    cProfile.run('prob.compute_totals()', 'profout')
    # cProfile.run('zzz()', 'profout')
    p = pstats.Stats('profout')
    p.strip_dirs()
    p.sort_stats('cumulative', 'time')
    p.print_stats()
    print('\n\n---------------------\n\n')
    p.print_callers()
    print('\n\n---------------------\n\n')
    p.print_callees()
else:
    # prob.check_totals()
    Ja = prob.compute_totals()
    for key1, value in sorted(Ja.items()):
        for key2 in sorted(value.keys()):
            print(key1, key2)
            print(value[key2])
    print(Ja)
    Jf = prob.compute_totals()
    print(Jf)
    Jf = prob.compute_totals()
    print(Jf)
    pickle.dump(Ja, open('mdp_derivs.p', 'wb'))


# save result (objective and constraints) to a pickle
picklevars = ['obj.val']
cons = [
    'pt%d.ConCh', 'pt%d.ConDs', 'pt%d.ConS0', 'pt%d.ConS1', 'pt%d_con5.val'
]
for pt in range(npts):
    for con in cons:
        picklevars.append(con % pt)

data = {}
for var in picklevars:
    data[var] = prob[var]

if MPI:
    pickle.dump(data, open('mdp_parallel.p', 'wb'))
else:
    pickle.dump(data, open('mdp_serial.p', 'wb'))
