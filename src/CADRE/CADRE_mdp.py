""" CADRE MDP Problem."""

from six.moves import range

import os.path
import numpy as np

from openmdao.core.parallelgroup import ParallelGroup

from CADRE.CADRE_group import CADRE

# Allow non-standard variable names for scientific calc
# pylint: disable=C0103


class CADRE_MDP_Group(ParallelGroup):

    def __init__(self, n=1500, m=300, npts=6):
        super(CADRE_MDP_Group, self).__init__()

        # Raw data to load
        fpath = os.path.dirname(os.path.realpath(__file__))
        fpath = os.path.join(fpath, 'data')
        solar_raw1 = np.genfromtxt(fpath + '/Solar/Area10.txt')
        solar_raw2 = np.loadtxt(fpath + '/Solar/Area_all.txt')
        comm_rawGdata = np.genfromtxt(fpath + '/Comm/Gain.txt')
        comm_raw = (10 ** (comm_rawGdata / 10.0)
                    ).reshape((361, 361), order='F')
        power_raw = np.genfromtxt(fpath + '/Power/curve.dat')

        # build design points
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):
            comp = self.add(name, CADRE(n, m, solar_raw1, solar_raw2,
                                        comm_raw, power_raw))

            comp['LD'] = LDs[i]
            comp['r_e2b_I0'] = r_e2b_I0[i]

