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

        h = 43200.0 / (n - 1)

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
        self.add('bp2', ParamComp('finAngle', np.pi/4.0))
        self.add('bp3', ParamComp('antAngle', 0.0))

        # CADRE instances go into a Parallel Group
        para = self.add('parallel', ParallelGroup(), promotes=['*'])

        # build design points
        names = ['pt%s' % i for i in range(npts)]
        for i, name in enumerate(names):

            # Some initial values
            initial_params = {}
            initial_params['LD'] = float(LDs[i])
            initial_params['r_e2b_I0'] = r_e2b_I0s[i]

            # User-defined initial parameters
            if initial_params is None:
                initial_params = {}
            if 't1' not in initial_params:
                initial_params['t1'] = 0.0
            if 't2' not in initial_params:
                initial_params['t2'] = 43200.0
            if 't' not in initial_params:
                initial_params['t'] = np.array(range(0, n))*h
            if 'CP_Isetpt' not in initial_params:
                initial_params['CP_Isetpt'] = 0.2 * np.ones((12, m))
            if 'CP_gamma' not in initial_params:
                initial_params['CP_gamma'] = np.pi/4 * np.ones((m, ))
            if 'CP_P_comm' not in initial_params:
                initial_params['CP_P_comm'] = 0.1 * np.ones((m, ))

            if 'iSOC' not in initial_params:
                initial_params['iSOC'] = np.array([0.5])

            # Fixed Station Parameters for the CADRE problem.
            # McMurdo station: -77.85, 166.666667
            # Ann Arbor: 42.2708, -83.7264
            if 'LD' not in initial_params:
                initial_params['LD'] = 5000.0
            if 'lat' not in initial_params:
                initial_params['lat'] = 42.2708
            if 'lon' not in initial_params:
                initial_params['lon'] = -83.7264
            if 'alt' not in initial_params:
                initial_params['alt'] = 0.256

            # Initial Orbital Elements
            if 'r_e2b_I0' not in initial_params:
                initial_params['r_e2b_I0'] = np.zeros((6, ))

            comp = para.add(name, CADRE(n, m, solar_raw1, solar_raw2,
                                        comm_raw, power_raw))

            # Some initial setup.
            self.add('%s_t1' % name, ParamComp('t1', initial_params['t1']))
            self.connect("%s_t1.t1" % name, '%s.t1'% name)
            self.add('%s_t2' % name, ParamComp('t2', initial_params['t2']))
            self.connect("%s_t2.t2" % name, '%s.t2'% name)
            self.add('%s_t' % name, ParamComp('t', initial_params['t']))
            self.connect("%s_t.t" % name, '%s.t'% name)

            # Design parameters
            self.add('%s_CP_Isetpt' % name, ParamComp('CP_Isetpt',
                                                        initial_params['CP_Isetpt']))
            self.connect("%s_CP_Isetpt.CP_Isetpt" % name,
                         '%s.CP_Isetpt'% name)
            self.add('%s_CP_gamma' % name, ParamComp('CP_gamma',
                                             initial_params['CP_gamma']))
            self.connect("%s_CP_gamma.CP_gamma" % name,
                         '%s.CP_gamma'% name)
            self.add('%s_CP_P_comm' % name, ParamComp('CP_P_comm',
                                              initial_params['CP_P_comm']))
            self.connect("%s_CP_P_comm.CP_P_comm" % name,
                         '%s.CP_P_comm'% name)
            self.add('%s_iSOC' % name, ParamComp('iSOC', initial_params['iSOC']))
            self.connect("%s_iSOC.iSOC" % name,
                         '%s.iSOC'% name)

            # These are broadcast params in the MDP.
            #self.add('p_cellInstd', ParamComp('cellInstd', np.ones((7, 12))),
            #         promotes=['*'])
            #self.add('p_finAngle', ParamComp('finAngle', np.pi / 4.), promotes=['*'])
            #self.add('p_antAngle', ParamComp('antAngle', 0.0), promotes=['*'])

            self.add('%s_param_LD' % name, ParamComp('LD', initial_params['LD']))
            self.connect("%s_param_LD.LD" % name,
                         '%s.LD'% name)
            self.add('%s_param_lat' % name, ParamComp('lat', initial_params['lat']))
            self.connect("%s_param_lat.lat" % name,
                         '%s.lat'% name)
            self.add('%s_param_lon' % name, ParamComp('lon', initial_params['lon']))
            self.connect("%s_param_lon.lon" % name,
                         '%s.lon'% name)
            self.add('%s_param_alt' % name, ParamComp('alt', initial_params['alt']))
            self.connect("%s_param_alt.alt" % name,
                         '%s.alt'% name)
            self.add('%s_param_r_e2b_I0' % name, ParamComp('r_e2b_I0',
                                                 initial_params['r_e2b_I0']))
            self.connect("%s_param_r_e2b_I0.r_e2b_I0" % name,
                         '%s.r_e2b_I0'% name)

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

            self.add('%s_con5'% name, ConstraintComp("SOCi = SOCf", out='val'))
            self.connect("%s.SOC" % name, '%s_con5.SOCi'% name, src_indices=[0])
            self.connect("%s.SOC" % name, '%s_con5.SOCf'% name, src_indices=[n-1])

        obj = ''.join([" - %s_DataTot" % name for name in names])
        self.add('obj', ExecComp('val='+obj))
        for name in names:
            self.connect("%s.Data" % name, "obj.%s_DataTot" % name, src_indices=[n-1])

        # Set our groups to auto
        self.ln_solver.options['mode'] = 'auto'
        para.ln_solver.options['mode'] = 'auto'

