"""
Optimization of the CADRE MDP.
"""

from __future__ import print_function

import numpy as np

from openmdao.api import Problem, LinearBlockGS, PETScKrylov

from CADRE.CADRE_mdp import CADRE_MDP_Group

import cProfile
import pstats


# These numbers are for quick testing
n = 150
m = 6
npts = 2

# instantiate model
model = CADRE_MDP_Group(n=n, m=m, npts=npts)

# add design variables and constraints to each CADRE instance
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

# for parallel execution, we must use KSP
model.linear_solver = PETScKrylov()
# model.linear_solver = LinearBlockGS()
# model.parallel.linear_solver = LinearBlockGS()
# model.parallel.pt0.linear_solver = LinearBlockGS()
# model.parallel.pt1.linear_solver = LinearBlockGS()

# create problem
prob = Problem(model)
prob.setup()
prob.run_driver()


# ----------------------------------------------------------------
# Below this line, code I was using for verifying and profiling.
# ----------------------------------------------------------------


cProfile.run("prob.compute_totals()", 'profout')

p = pstats.Stats('profout')
p.strip_dirs()
p.sort_stats('cumulative', 'time')
p.print_stats()
print('\n\n---------------------\n\n')
p.print_callers()
print('\n\n---------------------\n\n')
p.print_callees()
