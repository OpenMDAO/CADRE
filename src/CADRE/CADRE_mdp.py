""" CADRE MDP Problem."""

from six.moves import range

import os.path
import numpy as np

from openmdao.components import ConstraintComp, ExecComp, ParamComp
from openmdao.core import Group, ParallelGroup


from CADRE.CADRE_group import CADRE

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103


class CADRE_MDP_Group(Group):
    """ CADRE MDP Problem. Can be run in Parallel.

    Args
    ----
    n : int
        Number of time integration points.

    m : int
        Number of panels in fin discretization.

    npts : int
        Number of instances of CADRE in the problem.
    """

    def __init__(self, n=1500, m=300, npts=6):
        super(CADRE_MDP_Group, self).__init__()

        # Raw data to load
        fpath = os.path.dirname(os.path.realpath(__file__))
        fpath = os.path.join(fpath, 'data')
        solar_raw1 = np.genfromtxt(fpath + '/Solar/Area10.txt')
        solar_raw2 = np.loadtxt(fpath + '/Solar/Area_all.txt')
        comm_rawGdata = np.genfromtxt(fpath + '/Comm/Gain.txt')
        comm_raw = (10 ** (comm_rawGdata / 10.0)).reshape((361, 361), order='F')
        power_raw = np.genfromtxt(fpath + '/Power/curve.dat')

        # Load launch data
        launch_data = np.loadtxt(fpath + '/Launch/launch1.dat')

        # orbit position and velocity data for each design point
        r_e2b_I0s = launch_data[1::2, 1:]

        # number of days since launch for each design point
        LDs = launch_data[1::2, 0] - 2451545

        # Create ParmComps for broadcast parameters.
        self.add('bp1', ParamComp('cellInstd', np.ones((7, 12))))
        self.add('bp2', ParamComp('finAngle', 0.0))
        self.add('bp3', ParamComp('antAngle', 0.0))

        # CADRE instances go into a Parallel Group
        para = self.add('parallel', ParallelGroup(), promotes=['*'])

        # build design points
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):

            # Some initial values
            inits = {}
            inits['LD'] = float(LDs[i])
            inits['r_e2b_I0'] = r_e2b_I0s[i]

            comp = para.add(name, CADRE(n, m, solar_raw1, solar_raw2,
                                        comm_raw, power_raw,
                                        initial_params=inits))

            # Hook up broadcast params
            self.connect('bp1.cellInstd', "%s.cellInstd" % name)
            self.connect('bp2.finAngle', "%s.finAngle" % name)
            self.connect('bp3.antAngle', "%s.antAngle" % name)

            # Add single-point constraints
            self.add('%s_con1'% name, ConstraintComp("ConCh <= 0", out='val'))
            self.add('%s_con2'% name, ConstraintComp("ConDs <= 0", out='val'))
            self.add('%s_con3'% name, ConstraintComp("ConS0 <= 0", out='val'))
            self.add('%s_con4'% name, ConstraintComp("ConS1 <= 0", out='val'))
            #self.add('%s_con5'% name, ConstraintComp("%s.SOC[0][0] = %s.SOC[0][-1]" % (name, name),
            #                                         out='val'))

            self.connect("%s.ConCh" % name, '%s_con1.ConCh'% name)
            self.connect("%s.ConDs" % name, '%s_con2.ConDs'% name)
            self.connect("%s.ConS0" % name, '%s_con3.ConS0'% name)
            self.connect("%s.ConS1" % name, '%s_con4.ConS1'% name)

            #-----------------------------------------------------
            # Because of a bug, manually make SOC constraint comp
            #-----------------------------------------------------
            from openmdao.core.component import Component

            class SOCConstraint(Component):

                def __init__(self, n):
                    super(SOCConstraint, self).__init__()

                    # Inputs
                    self.add_param('SOC', np.zeros((1, n)), units="unitless",
                                   desc="Battery state of charge over time")

                    # Outputs
                    self.add_output('SOCi', 0.0, units="unitless",
                                    desc="Iniitial SOC")

                    self.add_output('SOCf', 0.0, units="unitless",
                                    desc="Final SOC")


                def solve_nonlinear(self, params, unknowns, resids):
                    """ Calculate output. """

                    unknowns['SOCi'] = params['SOC'][0][0]
                    unknowns['SOCf'] = params['SOC'][0][-1]

                def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
                    """ Matrix-vector product with the Jacobian. """

                    if mode == 'fwd':
                        dresids['SOCi'] += dparams['SOC'][0][0]
                        dresids['SOCf'] += dparams['SOC'][0][-1]

                    else:
                        dSOC = dparams['SOC']
                        dSOC[0][0] += dresids['SOCi']
                        dSOC[0][-1] += dresids['SOCf']
                        dparams['SOC'] = dSOC


            comp.add('SOC_Constraint', SOCConstraint(n), promotes=['*'])
            self.add('%s_con5'% name, ConstraintComp("SOCi = SOCf", out='val'))
            self.connect("%s.SOCi" % name, '%s_con5.SOCi'% name)
            self.connect("%s.SOCf" % name, '%s_con5.SOCf'% name)

            #-----------------------------------------------------
            # Same thing for the objective
            #-----------------------------------------------------

            class TotalData(Component):

                def __init__(self, n):
                    super(TotalData, self).__init__()

                    # Inputs
                    self.add_param('Data', np.zeros((1, n)), units="Gibyte",
                                    desc="Downloaded data state over time")

                    # Outputs
                    self.add_output('DataTot', 0.0, units="unitless",
                                    desc="Total Data")


                def solve_nonlinear(self, params, unknowns, resids):
                    """ Calculate output. """

                    unknowns['DataTot'] = params['Data'][0][-1]

                def apply_linear(self, params, unknowns, dparams, dunknowns, dresids, mode):
                    """ Matrix-vector product with the Jacobian. """

                    if mode == 'fwd':
                        dresids['DataTot'] += dparams['Data'][0][-1]

                    else:
                        dSOC = dparams['Data']
                        dSOC[0][-1] += dresids['DataTot']
                        dparams['Data'] = dSOC



            comp.add('TotalData', TotalData(n), promotes=['*'])

        obj = ''.join([" - %s_DataTot" % name for name in names])
        self.add('obj', ExecComp('val='+obj))
        for name in names:
            self.connect("%s.DataTot" % name, "obj.%s_DataTot" % name)

        # Set our groups to auto
        self.ln_solver.options['mode'] = 'auto'
        para.ln_solver.options['mode'] = 'auto'

