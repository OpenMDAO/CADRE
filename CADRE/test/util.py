"""
Utility functions for use in testing CADRE.
"""

import sys
import os
import pickle
import numpy as np


def load_validation_data(idx='0'):
    """
    load saved data from John's CMF implementation.
    """
    setd = {}
    fpath = os.path.dirname(os.path.realpath(__file__))
    if sys.version_info.major == 2:
        data = pickle.load(open(fpath + "/data1346_py2.pkl", 'rb'))
    else:
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
    h = 43200. / (n - 1)

    setd['r_e2b_I0'] = np.zeros(6)
    setd['r_e2b_I0'][:3] = data[idx + ":r_e2b_I0"]
    setd['r_e2b_I0'][3:] = data[idx + ":v_e2b_I0"]
    setd['Gamma'] = data[idx + ":gamma"]

    return n, m, h, setd
