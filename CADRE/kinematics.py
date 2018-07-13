"""
Some Kinematics methods.
"""

import numpy as np

from six.moves import range


def fixangles(n, azimuth0, elevation0):
    """
    Take an input azimuth and elevation angles in radians, and fix them to
    lie on [0, 2pi]. Also swap azimuth 180deg if elevation is over 90 deg.
    """
    azimuth, elevation = np.zeros(azimuth0.shape), np.zeros(elevation0.shape)

    for i in range(n):
        azimuth[i] = azimuth0[i] % (2*np.pi)
        elevation[i] = elevation0[i] % (2*np.pi)
        if elevation[i] > np.pi:
            elevation[i] = 2*np.pi - elevation[i]
            azimuth[i] = np.pi + azimuth[i]
            azimuth[i] = azimuth[i] % (2.0*np.pi)

    return azimuth, elevation


def computepositionrotd(n, vects, mat):
    """
    Perform matrix vector product between position vectors and rotation matrix.
    """
    result = np.empty(vects.shape)
    for i in range(n):
        result[:, i] = np.dot(mat[:, :, i], vects[:, i])
    return result


def computepositionrotdjacobian(n, v1, O_21):
    """
    Compute Jacobian for rotation matrix.
    """
    J1 = np.zeros((n, 3, 3, 3))

    for k in range(0, 3):
        for v in range(0, 3):
            J1[:, k, k, v] = v1[v, :]

    J2 = np.transpose(O_21, (2, 0, 1))

    return J1, J2


def computepositionspherical(n, v):
    """
    Convert (x,y,z) position in v to spherical coordinates.
    """
    azimuth, elevation = np.empty(n), np.empty(n)

    r = np.sqrt(np.sum(v*v, 0))
    for i in range(n):
        x = v[0, i]
        y = v[1, i]
        # z = v[2, i]
        if r[i] < 1e-15:
            r[i] = 1e-5
        azimuth[i] = arctan(x, y)

    elevation = np.arccos(v[2, :]/r)
    return azimuth, elevation


def arctan(x, y):
    """
    Specific way we want out arctans.
    """
    if x == 0:
        if y > 0:
            return np.pi/2.0
        elif y < 0:
            return 3*np.pi/2.0
        else:
            return 0.0
    elif y == 0.:
        if x < 0.:
            return np.pi
        else:
            return 0.0
    elif x < 0:
        return np.arctan(y/x) + np.pi
    elif y < 0:
        return np.arctan(y/x) + 2*np.pi
    elif y > 0.:
        return np.arctan(y/x)
    else:
        return 0.0


def computepositionsphericaljacobian(n, nJ, v):
    """
    Compute Jacobian across conversion to spherical coords.
    """
    Ja1 = np.empty(nJ)
    Ji1 = np.empty(nJ)
    Jj1 = np.empty(nJ)
    Ja2 = np.empty(nJ)
    Ji2 = np.empty(nJ)
    Jj2 = np.empty(nJ)

    for i in range(n):
        x = v[0, i]
        y = v[1, i]
        z = v[2, i]
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-15:
            r = 1e-5

        a = arctan(x, y)
        e = np.arccos(z/r)

        if e < 1e-15:
            e = 1e-5

        if e > (2*np.arccos(0.0) - 1e-15):
            e = 2*np.arccos(0.0) - 1e-5

        da_dr = 1.0/r * np.array([-np.sin(a)/np.sin(e), np.cos(a)/np.sin(e), 0.0])
        de_dr = 1.0/r * np.array([np.cos(a)*np.cos(e), np.sin(a)*np.cos(e), -np.sin(e)])

        for k in range(3):
            iJ = i*3 + k
            Ja1[iJ] = da_dr[k]
            Ji1[iJ] = i
            Jj1[iJ] = iJ
            Ja2[iJ] = de_dr[k]
            Ji2[iJ] = i
            Jj2[iJ] = iJ

    return Ja1, Ji1, Jj1, Ja2, Ji2, Jj2
