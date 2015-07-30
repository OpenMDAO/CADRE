""" Optimization of the CADRE MDP."""

from openmdao.core.problem import Problem
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver
from CADRE.CADRE_mdp import CADRE_MDP_Group


n = 1500
m = 300
npts = 6
restart = False


# Instantiate
model = Problem()
root = model.root = CADRE_MDP_Group(n=n, m=m, npts=npts)

# add SNOPT driver
model.driver = pyOptSparseDriver()
model.driver['optimizer'] = "SNOPT"
model.driver.opt_settings = {'Major optimality tolerance': 1e-3,
                             'Iterations limit': 500000000,
                             "New basis file": 10}

# Restart File
if restart is True and os.path.exists("fort.10"):
    model.driver.opt_settings["Old basis file"] = 10

# Add parameters and constraints to each CADRE instance.
names = ['pt%s' % i for i in range(npts)]
for i, name in enumerate(names):

    # add parameters to driver
    model.driver.add_parameter("%s.CP_Isetpt" % name, low=0., high=0.4)
    model.driver.add_parameter("%s.CP_gamma" % name, low=0, high=np.pi/2.)
    model.driver.add_parameter("%s.CP_P_comm" % name, low=0., high=25.)
    model.driver.add_parameter("%s.iSOC[0]" % name, low=0.2, high=1.)

    # add constraints
    model.driver.add_constraint("%s.ConCh <= 0" % name)
    model.driver.add_constraint("%s.ConDs <= 0" % name)
    model.driver.add_constraint("%s.ConS0 <= 0" % name)
    model.driver.add_constraint("%s.ConS1 <= 0" % name)
    model.driver.add_constraint("%s.SOC[0][0] = %s.SOC[0][-1]" % (name, name))

# add parameter groups
cell_param = ["%s.cellInstd" % name for name in names]
self.driver.add_parameter(cell_param, low=0, high=1)

finangles = ["%s.finAngle" % name for name in names]
self.driver.add_parameter(finangles, low=0, high=np.pi / 2.)

antangles = ["%s.antAngle" % name for name in names]
self.driver.add_parameter(antangles, low=-np.pi / 4, high=np.pi / 4)

# add objective
obj = ''.join(["-%s.Data[0][-1]" % name for name in names])
self.driver.add_objective(obj)




# Load launch data
launch_data = np.loadtxt(fpath + '/Launch/launch1.dat')

# orbit position and velocity data for each design point
r_e2b_I0s = launch_data[1::2, 1:]

# number of days since launch for each design point
LDs = launch_data[1::2, 0] - 2451545

# Loop over CADRE instances to set these
comp['LD'] = LDs[i]
comp['r_e2b_I0'] = r_e2b_I0[i]