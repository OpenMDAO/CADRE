""" Optimization of the CADRE MDP."""

from __future__ import print_function
from six.moves import range

import numpy as np

from openmdao.core.mpi_wrap import MPI
from openmdao.core.problem import Problem
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

try:
    from openmdao.core.petsc_impl import PetscImpl as impl
except ImportError:
    impl = None

from openmdao.solvers.ln_gauss_seidel import LinearGaussSeidel
from openmdao.solvers.petsc_ksp import PetscKSP

from CADRE.CADRE_mdp import CADRE_MDP_Group

# These numbers are for the CADRE problem in the paper.
n = 1500
m = 300
npts = 6
restart = False

# These numbers are for quick testing
#n = 150
#m = 6
#npts = 2


# Instantiate
model = Problem(impl=impl)
root = model.root = CADRE_MDP_Group(n=n, m=m, npts=npts)

# add SNOPT driver
model.driver = pyOptSparseDriver()
model.driver.options['optimizer'] = "SNOPT"
model.driver.opt_settings = {'Major optimality tolerance': 1e-3,
                             'Major feasibility tolerance': 1.0e-5,
                             'Iterations limit': 500000000,
                             #"New basis file": 10
                          }

# Restart File
if restart is True and os.path.exists("fort.10"):
    model.driver.opt_settings["Old basis file"] = 10

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

# For Parallel exeuction, we must use KSP or LinearGS
#model.root.ln_solver = PetscKSP()
model.root.ln_solver = LinearGaussSeidel()
model.root.parallel.ln_solver = LinearGaussSeidel()
model.root.parallel.pt0.ln_solver = LinearGaussSeidel()
model.root.parallel.pt1.ln_solver = LinearGaussSeidel()
model.root.parallel.pt2.ln_solver = LinearGaussSeidel()
model.root.parallel.pt3.ln_solver = LinearGaussSeidel()
model.root.parallel.pt4.ln_solver = LinearGaussSeidel()
model.root.parallel.pt5.ln_solver = LinearGaussSeidel()

# Parallel Derivative calculation
#for con_name in ['.ConCh','.ConDs','.ConS0','.ConS1','_con5.val']:
#    model.driver.parallel_derivs(['%s%s'%(n,con_name) for n in names])

# Recording
# Some constraints only exit on one process so cannot record everything
recording_includes_options = ['obj.val']
for j in range(npts):
    recording_includes_options.append('pt%s.ConCh' % str(j))
    recording_includes_options.append('pt%s.ConDs' % str(j))
    recording_includes_options.append('pt%s.ConS0' % str(j))
    recording_includes_options.append('pt%s.ConS1' % str(j))
    recording_includes_options.append('pt%s_con5.val' % str(j))

#from openmdao.recorders import DumpRecorder
#rec = DumpRecorder(out='data.dmp')
#model.driver.add_recorder(rec)
#rec.options['includes'] = recording_includes_options

from openmdao.recorders.sqlite_recorder import SqliteRecorder
rec = SqliteRecorder(out='data.sql')
model.driver.add_recorder(rec)
rec.options['includes'] = recording_includes_options

model.setup()
model.run()

import resource
print("Memory Usage:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000.0, "MB (on unix)")

#----------------------------------------------------------------
# Below this line, code I was using for verifying and profiling.
#----------------------------------------------------------------
#profile = False
#params = model.driver.get_desvars().keys()
#unks = model.driver.get_objectives().keys() + model.driver.get_constraints().keys()
#if profile is True:
#    import cProfile
#    import pstats
#    def zzz():
#        for j in range(1):
#            model.run()
#    cProfile.run("model.calc_gradient(params, unks, mode='rev', return_format='dict')", 'profout')
#    #cProfile.run("zzz()", 'profout')
#    p = pstats.Stats('profout')
#    p.strip_dirs()
#    p.sort_stats('cumulative', 'time')
#    p.print_stats()
#    print('\n\n---------------------\n\n')
#    p.print_callers()
#    print('\n\n---------------------\n\n')
#    p.print_callees()
#else:
#    #model.check_total_derivatives()
#    Ja = model.calc_gradient(params, unks, mode='rev', return_format='dict')
#    for key1, value in sorted(Ja.items()):
#        for key2 in sorted(value.keys()):
#            print(key1, key2)
#            print(value[key2])
#    #print(Ja)
#    #Jf = model.calc_gradient(params, unks, mode='fwd', return_format='dict')
#    #print(Jf)
#    #Jf = model.calc_gradient(params, unks, mode='fd', return_format='dict')
#    #print(Jf)
#    import pickle
#    pickle.dump(Ja, open( "mdp_derivs.p", "wb" ))
#
#import pickle
#data = {}
#varlist = []
#picklevars = ['obj.val',
              #'pt0_con1.val', 'pt0_con2.val', 'pt0_con3.val', 'pt0_con4.val', 'pt0_con5.val',
              #'pt1_con1.val', 'pt1_con2.val', 'pt1_con3.val', 'pt1_con4.val', 'pt1_con5.val',
              #]
#for var in picklevars:
    #data[var] = model[var]
#pickle.dump(data, open( "mdp_execute.p", "wb" ))
