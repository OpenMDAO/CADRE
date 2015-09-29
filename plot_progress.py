""" Plots our objective and constraint histories from the current data in
data.dmp. You can do this while the model is running."""

from six.moves import range

import sqlitedict

import numpy as np
from matplotlib import pylab

# >>> import sqlitedict
# >>> db = sqlitedict.SqliteDict( 'data.sql', 'openmdao' )
# >>> data = db['SLSQP/1']
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "/Users/hschilli/anaconda/envs/clippy/lib/python2.7/site-packages/sqlitedict.py", line 212, in __getitem__
#     raise KeyError(key)
# KeyError: 'SLSQP/1'
# >>> data = db['SNOPT/1']
# >>> p = data['timestamp']
# >>> data.keys()
# ['Parameters', 'Unknowns', 'Residuals', 'timestamp']
# >>> data['Unknowns']
# {'pt0_con5.val': 13.753205729551595, 'pt1_con5.val': 18.54323848767929, 'obj.val': -195.90535395795752}

def extract_all_vars_sql(name):
    """ Reads in the file given in name and extracts all variables."""


    db = sqlitedict.SqliteDict( name, 'openmdao' )

    data = {}
    for iteration in range(len(db)):
        iteration_coordinate = 'SNOPT/{}'.format(iteration + 1 )

        record = db[iteration_coordinate]

        for key, value in record['Unknowns'].items():
            if key not in data:
                data[key] = []
            data[key].append(value)

    return data


def extract_all_vars(name):
    """ Reads in the file given in name and extracts all variables."""

    file = open(name)
    data = {}
    record_flag = False
    for line in file:

        if line.startswith('Unknowns'):
            record_flag = True
            continue
        elif line.startswith('Resids:'):
            record_flag = False
            continue

        if record_flag is True:
            key, value = line.split(':')
            key = key.lstrip()
            if key not in data:
                data[key] = []
            data[key].append(float(value.lstrip().replace('\n', '')))

    return data

#filename = 'data.dmp' # use this when plotting the results of a serial run
filename = 'data_0.dmp' # use this when plotting the results of a parallel run

filename_sql = 'data.sql'

#data = extract_all_vars(filename) # uncomment and use this if you want to plot a dump recorder file
data = extract_all_vars_sql(filename_sql)

if 'pt1.ConCh' in data:
    serial_run = True
else:
    serial_run = False

if serial_run:
    n_point = (len(data) - 1)/5
else:
    n_point = len(data) - 1

X = data['obj.val']
X = [-val for val in X]
ncase = len(X)


Y = []
Z = []
for ic in range(ncase):
    c1 = c2 = c3 = c4 = c5 = 0
    for ip in range(n_point):
        if serial_run:
            c1 += data['pt%d.ConCh' % ip][ic]
            c2 += data['pt%d.ConDs' % ip][ic]
            c3 += data['pt%d.ConS0' % ip][ic]
            c4 += data['pt%d.ConS1' % ip][ic]
        c5 += data['pt%d_con5.val' % ip][ic]

    if serial_run:
        feasible = [c1, c2, c3, c4, c5]
    else:
        feasible = [c5,]

    Y.append(sum(feasible))
    Z.append(feasible)

#for case in cases:
    #data = [ case['pt' + str(i) + '.Data'][0][1499] for i in xrange(6) ]
    #sumdata = sum([float(i) for i in data if i])

    #c1 = [get_constraint_value_from_case( cds, case, "pt" + str(i) + ".ConCh <= 0") for i in xrange(6)]
    #c2 = [get_constraint_value_from_case( cds, case, "pt" + str(i) + ".ConDs <= 0") for i in xrange(6)]
    #c3 = [get_constraint_value_from_case( cds, case, "pt" + str(i) + ".ConS0 <= 0") for i in xrange(6)]
    #c4 = [get_constraint_value_from_case( cds, case, "pt" + str(i) + ".ConS1 <= 0") for i in xrange(6)]
    #c5 = [get_constraint_value_from_case( cds, case, "pt" + str(i) + ".SOC[0][0] = pt" + str(i) + ".SOC[0][-1]") for i in xrange(6)]


    #c1_f = sum([float(i) for i in c1 if i])
    #c2_f = sum([float(i) for i in c2 if i])
    #c3_f = sum([float(i) for i in c3 if i])
    #c4_f = sum([float(i) for i in c4 if i])
    #c5_f = sum([float(i) for i in c5 if i])

    #feasible = [c1_f, c2_f,  c3_f, c4_f, c5_f]

    #X.append(sumdata), Y.append(sum(feasible)), Z.append(feasible)

    #pcom.append([float(case["pt5.CP_gamma"][i])for i in xrange(300)])

    ## print sumdata, sum(feasible), max(feasible) #,[ '%.1f' % i for i in
    ## feasible]
    #print sumdata

#pylab.figure()
#pylab.plot(pcom[-1])

Z = np.array(Z)
if not len(Z):
    print "no data yet..."
    quit()
pylab.figure()
pylab.subplot(311)
pylab.title("total data")
pylab.plot(X, 'b')
pylab.plot([0, len(X)], [3e4, 3e4], 'k--', marker="o")
pylab.subplot(312)
pylab.title("Sum of Constraints")
pylab.plot([0, len(Y)], [0, 0], 'k--', marker="o")
pylab.plot(Y, 'k')
pylab.subplot(313)
pylab.title("Max of Constraints")
pylab.plot([0, len(Z)], [0, 0], 'k--')
if serial_run:
    pylab.plot(Z[:, 0], marker="o", label="ConCh")
    pylab.plot(Z[:, 1], marker="o", label="ConDs")
    pylab.plot(Z[:, 2], marker="o", label="ConS0")
    pylab.plot(Z[:, 3], marker="o", label="ConS1")
    pylab.plot(Z[:, 4], marker="o", label="c5")
else:
    pylab.plot(Z[:, 0], marker="o", label="c5")

pylab.legend(loc="best")

pylab.show()
