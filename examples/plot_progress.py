"""
Plots objective and constraint histories from the recorded data in 'data.sql'.
"""
from __future__ import print_function

from six.moves import range

import numpy as np
from matplotlib import pylab

from openmdao.api import CaseReader

# load cases from recording database
cases = CaseReader('data.sql').driver_cases
cases.load_cases()

num_cases = cases.num_cases
if num_cases == 0:
    print('No data yet...')
    quit()
else:
    print('# cases:', num_cases)

# determine the # of points (5 constraints per point)
constraints = cases.get_case(0).get_constraints().keys
n_point = len(constraints) // 5

# collect data into arrays for plotting
X = np.zeros(num_cases)       # obj.val
Y = np.zeros(num_cases)       # sum of constraints
Z = np.zeros((num_cases, 5))  # constraints

num_cases = cases.num_cases
for ic in range(num_cases):
    data = cases.get_case(ic).outputs
    X[ic] = -data['obj.val']

    c1 = c2 = c3 = c4 = c5 = 0
    for ip in range(n_point):
        c1 += data['pt%d.ConCh' % ip]
        c2 += data['pt%d.ConDs' % ip]
        c3 += data['pt%d.ConS0' % ip]
        c4 += data['pt%d.ConS1' % ip]
        c5 += data['pt%d_con5.val' % ip]

    feasible = [c1, c2, c3, c4, c5]

    Y[ic] = sum(feasible)
    Z[ic, :] = feasible

# generate plots
pylab.figure()

pylab.subplot(311)
pylab.title('total data')
pylab.plot(X, 'b')
pylab.plot([0, len(X)], [3e4, 3e4], 'k--', marker='o')

pylab.subplot(312)
pylab.title('Sum of Constraints')
pylab.plot([0, len(Y)], [0, 0], 'k--', marker='o')
pylab.plot(Y, 'k')

pylab.subplot(313)
pylab.title('Max of Constraints')
pylab.plot([0, len(Z)], [0, 0], 'k--')
pylab.plot(Z[:, 0], marker='o', label='ConCh')
pylab.plot(Z[:, 1], marker='o', label='ConDs')
pylab.plot(Z[:, 2], marker='o', label='ConS0')
pylab.plot(Z[:, 3], marker='o', label='ConS1')
pylab.plot(Z[:, 4], marker='o', label='c5')

pylab.legend(loc='best')

pylab.show()
