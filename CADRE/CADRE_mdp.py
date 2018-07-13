"""
CADRE MDP Group.
"""

from six.moves import range

import os.path
import numpy as np

from openmdao.api import Group, ParallelGroup, IndepVarComp, ExecComp

from CADRE.CADRE_group import CADRE


class CADRE_MDP_Group(Group):
    """
    CADRE MDP Group. Can be run in Parallel.

    Parameters
    ----------
    n : int
        Number of time integration points.

    m : int
        Number of panels in fin discretization.

    npts : int
        Number of instances of CADRE in the problem.
    """

    def __init__(self, n=1500, m=300, npts=6):
        super(CADRE_MDP_Group, self).__init__()

        self.n = n
        self.m = m
        self.npts = npts

    def setup(self):
        n = self.n
        m = self.m
        npts = self.npts

        # Raw data to load
        fpath = os.path.dirname(os.path.realpath(__file__))
        fpath = os.path.join(fpath, 'data')

        solar_raw1 = np.genfromtxt(fpath + '/Solar/Area10.txt')
        solar_raw2 = np.loadtxt(fpath + '/Solar/Area_all.txt')

        comm_rawGdata = np.genfromtxt(fpath + '/Comm/Gain.txt')
        comm_raw = (10 ** (comm_rawGdata / 10.0)).reshape((361, 361), order='F')

        power_raw = np.genfromtxt(fpath + '/Power/curve.dat')

        launch_data = np.loadtxt(fpath + '/Launch/launch1.dat')

        # orbit position and velocity data for each design point
        r_e2b_I0s = launch_data[1::2, 1:]

        # number of days since launch for each design point
        LDs = launch_data[1::2, 0] - 2451545

        # Create IndepVarComp for broadcast parameters.
        bp = self.add_subsystem('bp', IndepVarComp())
        bp.add_output('cellInstd', np.ones((7, 12)))
        bp.add_output('finAngle', np.pi/4.0, units='rad')
        bp.add_output('antAngle', 0.0, units='rad')

        # CADRE instances go into a Parallel Group
        para = self.add_subsystem('parallel', ParallelGroup(), promotes=['*'])

        # build design points
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):
            # Some initial values
            inits = {
                'LD': float(LDs[i]),
                'r_e2b_I0': r_e2b_I0s[i]
            }

            para.add_subsystem(name, CADRE(n, m, solar_raw1, solar_raw2,
                               comm_raw, power_raw, initial_inputs=inits))

            # Hook up broadcast inputs
            self.connect('bp.cellInstd', '%s.cellInstd' % name)
            self.connect('bp.finAngle', '%s.finAngle' % name)
            self.connect('bp.antAngle', '%s.antAngle' % name)

            self.add_subsystem('%s_con5' % name, ExecComp('val = SOCi - SOCf'))

            self.connect('%s.SOC' % name, '%s_con5.SOCi' % name,
                         src_indices=[0], flat_src_indices=True)
            self.connect('%s.SOC' % name, '%s_con5.SOCf' % name,
                         src_indices=[n-1], flat_src_indices=True)

        # objective: sum of data from all design points
        data_totals = ['%s_DataTot' % name for name in names]
        obj = ''.join([' - %s' % data_tot for data_tot in data_totals])

        meta_dicts = {}
        for dt in data_totals:
            meta_dicts[dt] = {'units': 'Gibyte'}

        self.add_subsystem('obj', ExecComp('val='+obj, **meta_dicts))
        for name in names:
            self.connect('%s.Data' % name, 'obj.%s_DataTot' % name,
                         src_indices=[n-1], flat_src_indices=True)
