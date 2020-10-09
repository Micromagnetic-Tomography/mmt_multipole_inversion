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

    r2 = np.sum(dr ** 2, axis=1)
    r = np.sqrt(r2)
    f = 3e-7 / (r2 * r2 * r)
    g = -1e-7 / (r2 * r)

    #  Only return Bz
    return np.column_stack((f * z * x, f * z * y, f * z * z + g))


# ORIGINAL:
# def quadrupole_Bz_sus(dip_r, pos_r):  # see Overleaf
#     r = [pos_r[k] - dip_r[k] for k in range(3)]
#     r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
#     r = np.sqrt(r2)
#     p1 = 5 * r[2] * (r[0] * r[0]-r[2] * r[2]) + 2 * r2 * r[2]
#     p2 = 5 * r[0] * r[1] * r[2]
#     p3 = 5 * r[0] * (r[2] * r[2] - r2)
#     p4 = 5 * r[2] * (r[1] * r[1]-r[2] * r[2]) + 2 * r2 * r[2]
#     p5 = 5 * r[1] * (r[2] * r[2] - r2)
#     g = 1e-7 / (r2 * r2 * r2 * r)
#
#     # Only return Bz
#     return([g * p1, g * p2, g * p3, g * p4, g * p5])

def quadrupole_Bz_sus(dip_r, pos_r):  # see Overleaf
    """
    dip_r   :: N x 3 array OR 1 x 3 array
    pos_r   :: N x 3 array OR 1 x 3 array

    Returns

        N x 5  array with 5 quadrupole moments from the traceless 2-tensor
    """

    dr = pos_r - dip_r
    x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]

    r2 = np.sum(dr ** 2, axis=1)
    r = np.sqrt(r2)

    p1 = 5. * z * (x * x - z * z) + 2 * r2 * z
    p2 = 10. * x * y * z
    p3 = 2. * x * (5 * z * z - r2)
    p4 = 5. * z * (y * y - z * z) + 2 * r2 * z
    p5 = 2. * y * (5 * z * z - r2)
    g = 1e-7 / (r2 * r2 * r2 * r)

    # Only return Bz
    return np.column_stack((g * p1, g * p2, g * p3, g * p4, g * p5))


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

    r2 = np.sum(dr ** 2, axis=1)
    r = np.sqrt(r2)

    octp[:, 0] = 5 * x * z * (7 * (x * x - 3 * z * z) + 6 * r2)   # w_xxx
    octp[:, 1] = 15 * y * z * (7 * (x * x - z * z) + 2 * r2)      # w_xxy
    octp[:, 2] = 5 * (7 * z * z * (3 * x * x - z * z)               # w_xxz
                      - 3 * r2 * (x * x - z * z))                 #
    octp[:, 3] = 30 * x * y * (7 * z * z - r2)                    # w_xyz
    octp[:, 4] = 15 * x * z * (7 * (y * y - z * z) + 2 * r2)      # w_yyx
    octp[:, 5] = 5 * y * z * (7 * (y * y - 3 * z * z) + 6 * r2)   # w_yyy
    octp[:, 6] = 5 * (7 * z * z * (3 * y * y - z * z)               # w_yyz
                      - 3 * r2 * (y * y - z * z))                 #
    g = 1e-7 / (r2 * r2 * r2 * r2 * r)

    # Only return Bz
    return octp * g[:, None]
