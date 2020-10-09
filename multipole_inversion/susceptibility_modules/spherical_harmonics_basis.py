# Susceptibility functions defined by expressing the multipole expansion in
# a spherical harmonic basis. The main theory is defined by Burnham and English
# in Ref. []
#
import numpy as np


def dipole_Bz_sus(dip_r, pos_r):
    """

    dip_r   :: N x 3 array OR 1 x 3 array
    pos_r   :: N x 3 array OR 1 x 3 array

    Returns

    N x 3 array

    Calculate magnetic flux Bz-susceptibility per dipole component generated
    by dipole
    located in position dip_r    (m)
    at position  pos_r           (m)
    unit of result is T / (Am2)
    """

    dr = pos_r - dip_r
    x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
    x2, y2, z2 = x ** 2, y ** 2, z ** 2

    r2 = np.sum(dr ** 2, axis=1)
    r = np.sqrt(r2)
    f = 1e-7 / (r2 * r2 * r)

    # Original ones: (WORKING)
    p3 = (3 * z2 - r2)
    p2 = (3 * y * z)
    p1 = (3 * x * z)

    #  Only return Bz
    return np.column_stack((f * p1, f * p2, f * p3))


# TO CHECK
def quadrupole_Bz_sus(dip_r, pos_r):  # see Overleaf
    """
    dip_r   :: N x 3 array OR 1 x 3 array
    pos_r   :: N x 3 array OR 1 x 3 array


    Returns

        N x 5  array with 5 quadrupole moments from the traceless 2-tensor
    """

    dr = pos_r - dip_r
    x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
    x2, y2, z2 = x ** 2, y ** 2, z ** 2

    r2 = np.sum(dr ** 2, axis=1)
    r = np.sqrt(r2)

    # Quad Field from the Cart version of Quad field SHs, by Stone et al
    p1 = np.sqrt(3 / 2) * z * (-3 * r2 + 5 * z2)
    p2 = -np.sqrt(2) * x * (r2 - 5 * z2)
    p3 = -np.sqrt(2) * y * (r2 - 5 * z2)
    p4 = (5 / np.sqrt(2)) * (x2 - y2) * z
    p5 = 5 * np.sqrt(2) * x * y * z
    g = 1e-7 / (r2 * r2 * r2 * r)

    # Only return Bz
    return np.column_stack((g * p1, g * p2, g * p3, g * p4, g * p5))


# TO CHECK
def octupole_Bz_sus(dip_r, pos_r):  # see Overleaf
    """
    dip_r   :: N x 3 array OR 1 x 3 array
    pos_r   :: N x 3 array OR 1 x 3 array

    Returns

        N x 7  array with 7 octupole moments from the traceless 3-tensor
    """

    octp = np.zeros((dip_r.shape[0], 7))

    dr = pos_r - dip_r
    x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
    x2, y2, z2 = x ** 2, y ** 2, z ** 2

    r2 = np.sum(dr ** 2, axis=1)
    r4 = r2 ** 2
    r = np.sqrt(r2)

    # Oct Field from the Cartesian version of Octupole field SHs, by Stone et al
    octp[:, 0] = (3 * (r2 ** 2) - 30 * r2 * z2 + 35 * (z2 * z2)) / np.sqrt(10)
    octp[:, 1] = np.sqrt(15) * x * z * (-3 * r2 + 7 * z2) / 2
    octp[:, 2] = np.sqrt(15) * y * z * (-3 * r2 + 7 * z2) / 2
    octp[:, 3] = -np.sqrt(1.5) * (x2 - y2) * (r2 - 7 * z2)
    octp[:, 4] = -np.sqrt(6) * x * y * (r2 - 7 * z2)
    octp[:, 5] = 7 * x * (x2 - 3 * y2) * z / 2
    octp[:, 6] = -7 * y * (-3 * x2 + y2) * z / 2

    g = 1e-7 / (r4 * r4 * r)

    # Only return Bz
    return octp * g[:, None]
