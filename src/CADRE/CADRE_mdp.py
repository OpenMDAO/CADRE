""" CADRE MDP Problem."""

from six.moves import range

import os.path
import numpy as np

from openmdao.components.exec_comp import ExecComp
from openmdao.components.indep_var_comp import IndepVarComp
from openmdao.core.group import Group
from openmdao.core.parallel_group import ParallelGroup


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
        self.add_subsystem('bp1', IndepVarComp('cellInstd', np.ones((7, 12))))
        self.add_subsystem('bp2', IndepVarComp('finAngle', np.pi/4.0))
        self.add_subsystem('bp3', IndepVarComp('antAngle', 0.0))

        # CADRE instances go into a Parallel Group
        para = self.add_subsystem('parallel', ParallelGroup(), promotes=['*'])

        # build design points
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):

            # Some initial values
            inits = {}
            inits['LD'] = float(LDs[i])
            inits['r_e2b_I0'] = r_e2b_I0s[i]

            comp = para.add_subsystem(name, CADRE(n, m, solar_raw1, solar_raw2,
                                        comm_raw, power_raw,
                                        initial_inputs=inits))

            # Hook up broadcast inputs
            self.connect('bp1.cellInstd', "%s.cellInstd" % name)
            self.connect('bp2.finAngle', "%s.finAngle" % name)
            self.connect('bp3.antAngle', "%s.antAngle" % name)

            self.add_subsystem('%s_con5'% name, ExecComp("val = SOCi - SOCf"))
            self.connect("%s.SOC" % name, '%s_con5.SOCi'% name, src_indices=[0])
            self.connect("%s.SOC" % name, '%s_con5.SOCf'% name, src_indices=[n-1])

        obj = ''.join([" - %s_DataTot" % name for name in names])
        self.add_subsystem('obj', ExecComp('val='+obj))
        for name in names:
            self.connect("%s.Data" % name, "obj.%s_DataTot" % name, src_indices=[n-1])

        # Set our groups to auto
        self.ln_solver.options['mode'] = 'auto'
        para.ln_solver.options['mode'] = 'auto'

