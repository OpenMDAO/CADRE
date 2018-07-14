"""
Optimization of the CADRE MDP.
"""

from __future__ import print_function
from six.moves import range

import resource

import numpy as np

from openmdao.api import Problem, pyOptSparseDriver, LinearBlockGS, SqliteRecorder

from CADRE.CADRE_mdp import CADRE_MDP_Group

# These numbers are for the CADRE problem in the paper.
n = 1500
m = 300
npts = 6
restart = False

# These numbers are for quick testing
# n = 150
# m = 6
# npts = 2


# Instantiate CADRE model
model = CADRE_MDP_Group(n=n, m=m, npts=npts)

# Add design variables and constraints to each CADRE instance.
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
prob.driver.add_recorder(SqliteRecorder('data.sql'))

prob.setup()

# For Parallel execution, we must use KSP or LinearGS
# prob.model.linear_solver = PETScKrylov()
model.linear_solver = LinearBlockGS()
model.parallel.linear_solver = LinearBlockGS()
model.parallel.pt0.linear_solver = LinearBlockGS()
model.parallel.pt1.linear_solver = LinearBlockGS()
model.parallel.pt2.linear_solver = LinearBlockGS()
model.parallel.pt3.linear_solver = LinearBlockGS()
model.parallel.pt4.linear_solver = LinearBlockGS()
model.parallel.pt5.linear_solver = LinearBlockGS()

prob.run_driver()

print('Memory Usage:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0, 'MB (on unix)')

# ----------------------------------------------------------------
# Below this line, code I was using for verifying and profiling.
# ----------------------------------------------------------------
# profile = False
# inputs = list(prob.driver.get_desvars().keys())
# unks = list(prob.driver.get_objectives().keys()) + list(prob.driver.get_constraints().keys())
# if profile is True:
#    import cProfile
#    import pstats
#    def zzz():
#        for j in range(1):
#            prob.run()
#    cProfile.run('prob.calc_gradient(inputs, unks, mode='rev', return_format='dict')', 'profout')
#    #cProfile.run('zzz()', 'profout')
#    p = pstats.Stats('profout')
#    p.strip_dirs()
#    p.sort_stats('cumulative', 'time')
#    p.print_stats()
#    print('\n\n---------------------\n\n')
#    p.print_callers()
#    print('\n\n---------------------\n\n')
#    p.print_callees()
# else:
#    #prob.check_total_derivatives()
#    Ja = prob.calc_gradient(inputs, unks, mode='rev', return_format='dict')
#    for key1, value in sorted(Ja.items()):
#        for key2 in sorted(value.keys()):
#            print(key1, key2)
#            print(value[key2])
#    #print(Ja)
#    #Jf = prob.calc_gradient(inputs, unks, mode='fwd', return_format='dict')
#    #print(Jf)
#    #Jf = prob.calc_gradient(inputs, unks, mode='fd', return_format='dict')
#    #print(Jf)
#    import pickle
#    pickle.dump(Ja, open( 'mdp_derivs.p', 'wb' ))

# import pickle
# data = {}
# varlist = []
# picklevars = ['obj.val',
#               'pt0_con1.val', 'pt0_con2.val', 'pt0_con3.val', 'pt0_con4.val', 'pt0_con5.val',
#               'pt1_con1.val', 'pt1_con2.val', 'pt1_con3.val', 'pt1_con4.val', 'pt1_con5.val',
#               ]
# for var in picklevars:
#     data[var] = prob[var]
# pickle.dump(data, open( 'mdp_execute.p', 'wb' ))
