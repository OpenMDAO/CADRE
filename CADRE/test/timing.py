"""
Time CADRE's execution and derivative calculations.
"""
from __future__ import print_function

import time

from openmdao.api import Problem

from CADRE.CADRE_group import CADRE
from CADRE.test.util import load_validation_data


n, m, h, setd = load_validation_data(idx='0')

# instantiate Problem with CADRE model
t0 = time.time()
model = CADRE(n, m)
prob = Problem(model)
print("Instantiation:  ", time.time() - t0, 's')

# do problem setup
t0 = time.time()
prob.setup(check=False)
print("Problem Setup:  ", time.time() - t0, 's')

# set initial values from validation data
input_data = [
    'CP_P_comm',
    'LD',
    'cellInstd',
    'CP_gamma',
    'finAngle',
    'lon',
    'CP_Isetpt',
    'antAngle',
    't',
    'r_e2b_I0',
    'lat',
    'alt',
    'iSOC'
]
for inp in input_data:
    prob[inp] = setd[inp]

# run model
t0 = time.time()
prob.run_model()
print("Execute Model:  ", time.time() - t0, 's')

inputs = ['CP_gamma']
outputs = ['Data']

# calculate total derivatives
t0 = time.time()
J1 = prob.compute_totals(of=outputs, wrt=inputs)
print("Compute Totals: ", time.time() - t0, 's')

# calculate total derivatives via finite difference
model.approx_totals(method='fd')

t0 = time.time()
J1 = prob.compute_totals(of=outputs, wrt=inputs)
print("Approx Totals:  ", time.time() - t0, 's')
