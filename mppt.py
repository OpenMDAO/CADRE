""" Optimization of the CADRE MDP."""

from six.moves import range

import numpy as np

from openmdao.api import IndepVarComp, Component, Group, Problem, ParallelGroup
from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

from CADRE.power import Power_SolarPower, Power_CellVoltage
from CADRE.parameters import BsplineParameters

try:
    from openmdao.core.petsc_impl import PetscImpl as impl
except ImportError:
    impl = None

import os
import pickle

class Perf(Component):

  def __init__(self, n):
    super(Perf, self).__init__()
    self.add_param('P_sol1', np.zeros((n, )), units="W",
                        desc="Solar panels power over time")
    self.add_param('P_sol2', np.zeros((n, )), units="W",
                        desc="Solar panels power over time")

    self.add_output("result", 0.0)

    self.J = -np.ones((1, n))

  def solve_nonlinear(self, p, u, r):
    u['result'] = -np.sum(p['P_sol1']) -np.sum(p['P_sol2'])

  def linearize(self, p, u, r):

    return {("result", "P_sol1") : self.J, 
            ("result", "P_sol2") : self.J}

class MPPT(Group):

    def __init__(self, LOS, temp, area, m, n):
      super(MPPT, self).__init__()

      params = (("LOS" ,LOS, {"units" : "unitless"}), 
                ("temperature" , temp, {"units" : "degK"}), 
                ("exposedArea" , area, {"units" : "m**2"}), 
                ("CP_Isetpt" , np.zeros((12, m)),  {"units" : "A"}))

      self.add("param", IndepVarComp(params))
      self.add("bspline", BsplineParameters(n, m))
      self.add("voltage", Power_CellVoltage(n))
      self.add("power", Power_SolarPower(n))
      #self.add("perf", Perf(n))

      self.connect("param.LOS", "voltage.LOS")
      self.connect("param.temperature", "voltage.temperature")
      self.connect("param.exposedArea", "voltage.exposedArea")
      
      self.connect("param.CP_Isetpt", "bspline.CP_Isetpt")
      self.connect("bspline.Isetpt", "voltage.Isetpt")
      
      self.connect("bspline.Isetpt", "power.Isetpt")
      self.connect("voltage.V_sol", "power.V_sol")

      #self.connect("power.P_sol", "perf.P_sol")


class MPPT_MDP(ParallelGroup):

  def __init__(self):
    super(MPPT_MDP, self).__init__()
    n = 1500
    m = 300
    fpath = os.path.dirname(os.path.realpath(__file__))
    data = pickle.load(open(fpath + "/src/CADRE/test/data1346.pkl", 'rb'))

    self.add("pt0", MPPT(data['0:LOS'], 
                         data['0:temperature'], 
                         data['0:exposedArea'], 
                         m, n))
    self.add("pt1", MPPT(data['1:LOS'], 
                         data['1:temperature'], 
                         data['1:exposedArea'], 
                         m, n))

    self.add("perf", Perf(1500))

    self.connect("pt0.power.P_sol", "perf.P_sol1")
    self.connect("pt1.power.P_sol", "perf.P_sol2")


if __name__ == "__main__":

  import pylab
  import time

  model = Problem(impl=impl)
  model.root = MPPT_MDP()

  # add SNOPT driver
  model.driver = pyOptSparseDriver()
  model.driver.options['optimizer'] = "SNOPT"
  model.driver.opt_settings = {'Major optimality tolerance': 1e-3,
                               'Major feasibility tolerance': 1.0e-5,
                               'Iterations limit': 500000000,
                               "New basis file": 10}

  model.driver.add_objective("perf.result")
  model.driver.add_desvar("pt0.param.CP_Isetpt", lower=0., upper=0.4)
  model.driver.add_desvar("pt1.param.CP_Isetpt", lower=0., upper=0.4)

  model.setup()

  pylab.figure()
  pylab.subplot(211)
  pylab.plot(model['pt0.param.CP_Isetpt'].T)
  pylab.plot(model['pt1.param.CP_Isetpt'].T)

  t = time.time()

  model.run()

  pylab.subplot(212)
  pylab.plot(model['pt0.param.CP_Isetpt'].T)
  pylab.plot(model['pt1.param.CP_Isetpt'].T)

  print time.time() - t
  pylab.show()

