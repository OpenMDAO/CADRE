""" Time CADRE's execution and derivative calculations."""

from __future__ import print_function

import os
import pickle
import unittest
import time

import numpy as np

from openmdao.core.problem import Problem

from CADRE.CADRE_group import CADRE


idx = '0'

setd = {}
fpath = os.path.dirname(os.path.realpath(__file__))
data = pickle.load(open(fpath + "/data1346.pkl", 'rb'))

for key in data.keys():
    if key[0] == idx or not key[0].isdigit():
        if not key[0].isdigit():
            shortkey = key
        else:
            shortkey = key[2:]
        # set floats correctly
        if data[key].shape == (1,) and shortkey != "iSOC":
            setd[shortkey] = data[key][0]
        else:
            setd[shortkey] = data[key]

n = setd['P_comm'].size
m = setd['CP_P_comm'].size

t0 = time.time()
assembly = Problem(root=CADRE(n, m))
print("Instantiation: ", time.time() - t0)

t0 = time.time()
assembly.setup(check=False)
print("Setup: ", time.time() - t0)

setd['r_e2b_I0'] = np.zeros(6)
setd['r_e2b_I0'][:3] = data[idx + ":r_e2b_I0"]
setd['r_e2b_I0'][3:] = data[idx + ":v_e2b_I0"]
setd['Gamma'] = data[idx + ":gamma"]

assembly['CP_P_comm'] = setd['CP_P_comm']
assembly['LD'] = setd['LD']
assembly['cellInstd'] = setd['cellInstd']
assembly['CP_gamma'] = setd['CP_gamma']
assembly['finAngle'] = setd['finAngle']
assembly['lon'] = setd['lon']
assembly['CP_Isetpt'] = setd['CP_Isetpt']
assembly['antAngle'] = setd['antAngle']
assembly['t'] = setd['t']
assembly['r_e2b_I0'] = setd['r_e2b_I0']
assembly['lat'] = setd['lat']
assembly['alt'] = setd['alt']
assembly['iSOC'] = setd['iSOC']

t0 = time.time()
assembly.run()
print("Execute: ", time.time() - t0)

inputs = ['CP_gamma']
outputs = ['Data']

t0 = time.time()
J1 = assembly.calc_gradient(inputs, outputs, mode='fwd')
print("Fwd Gradient: ", time.time() - t0)

t0 = time.time()
J2 = assembly.calc_gradient(inputs, outputs, mode='rev')
print("Rev Gradient: ", time.time() - t0)

t0 = time.time()
Jfd = assembly.calc_gradient(inputs, outputs, mode='fd')
print("FD Gradient: ", time.time() - t0)



