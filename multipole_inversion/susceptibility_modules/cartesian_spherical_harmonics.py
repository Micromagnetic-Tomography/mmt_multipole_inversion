import numpy as np


def dipole_Bz_sus(dip_r, pos_r):
    """

    dip_r   :: N x 3 array OR 1 x 3 array
    pos_r   :: N x 3 array OR 1 x 3 array

    Returns

    N x 3 array

    Calculate magnetic flux Bz-susceptibility per dipole component  generated
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

    # Based on Spherical Harmonics: (D / Dz) Psi
    p3 = np.sqrt(3 / 2) * (3 * z2 - r2)
    p2 = -np.sqrt(6) * (3 * y * z)
    p1 = np.sqrt(6) * (3 * x * z)

    # Original ones: (WORKING)
    # p3 = (3 * z2 - r2)
    # p2 = (3 * y * z)
    # p1 = (3 * x * z)

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

    p1 = -np.sqrt(5) * z * (3 * r2 - 5 * z2)
    p5 = np.sqrt(15) * y * (r2 - 5 * z2)
    p4 = 5 * np.sqrt(10) * z * (x2 - y2)
    p3 = -np.sqrt(15) * x * (r2 - 5 * z2)
    p2 = -10 * np.sqrt(10) * x * y * z
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

    octp[:, 0] = np.sqrt(7 / 10) * (3 * r4 - 30 * r2 * z2 + 35 * z2 * z2)
    octp[:, 1] = 5 * np.sqrt(42 / 19) * y * z * (3 * r2 - 7 * z2)
    octp[:, 2] = -np.sqrt(35) * (x2 - y2) * (r2 - 7 * z2)
    octp[:, 3] = 7 * np.sqrt(14) * y * (-3 * x2 + y2) * z
    octp[:, 4] = 5 * np.sqrt(42 / 19) * x * z * (-3 * r2 + 7 * z2)
    octp[:, 5] = 2 * np.sqrt(35) * x * y * (r2 - 7 * z2)
    octp[:, 6] = 7 * np.sqrt(14) * x * (x2 - 3 * y2) * z

    g = 1e-7 / (r4 * r4 * r)

    # Only return Bz
    return octp * g[:, None]
