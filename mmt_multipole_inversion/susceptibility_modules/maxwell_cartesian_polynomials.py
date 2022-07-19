# Susceptibility functions obtained from the multipole expansion in Cartesian
# coordinates. These are also known as Maxwell-Cartesian harmonics, from the
# work of Applequist [ref]
import numpy as np
import numba


@numba.jit(nopython=True)
def dipole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    This function generates the dipolar Bz susceptibility field contributed
    from magnetic point sources over different positions of a scan grid. The
    method used here is populating the Q matrix which has size:

        (len(pos_r), len(dip_r) * n_col_stride)

    Parameters
    ----------
    dip_r
        N x 3 array OR 1 x 3 array
    pos_r
        M x 3 array OR 1 x 3 array

    Returns
    -------
    None
        None

    Calculate magnetic flux Bz-susceptibility per dipole component generated
    by dipoles located in position dip_r (m) at position  pos_r (m)
    Units of result is T / (A m2)
    """

    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)

        f = 1e-7 / (r2 * r2 * r)
        g = -1e-7 / (r2 * r)

        # Original ones: (WORKING)
        p1 = f * (3 * x * z)
        p2 = f * (3 * y * z)
        p3 = f * (3 * z * z) + g

        #  Only return Bz
        Q[i][::n_col_stride] = p1
        Q[i][1::n_col_stride] = p2
        Q[i][2::n_col_stride] = p3

    return None


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
@numba.jit(nopython=True)
def quadrupole_Bz_sus(dip_r, pos_r, Q, n_col_stride):  # see Overleaf
    """
    Parameters
    ----------
    dip_r
        N x 3 array OR 1 x 3 array
    pos_r
        M x 3 array OR 1 x 3 array

    Returns
    -------
    None
        None

    """

    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        # x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)

        # Quad Field using the M-C harmonics
        g = 1e-7 / (r2 * r2 * r2 * r)
        p1 = 5. * z * (x * x - z * z) + 2 * r2 * z
        p2 = 10. * x * y * z
        p3 = 2. * x * (5 * z * z - r2)
        p4 = 5. * z * (y * y - z * z) + 2 * r2 * z
        p5 = 2. * y * (5 * z * z - r2)

        # Fill the Q array in the corresponding entries for the quad moments
        Q[i][3::n_col_stride] = g * p1
        Q[i][4::n_col_stride] = g * p2
        Q[i][5::n_col_stride] = g * p3
        Q[i][6::n_col_stride] = g * p4
        Q[i][7::n_col_stride] = g * p5

    return None


@numba.jit(nopython=True)
def octupole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    Parameters
    ----------
    dip_r
        N x 3 array OR 1 x 3 array
    pos_r
        M x 3 array OR 1 x 3 array

    Returns
    -------
    None
        None

    """

    octp = np.zeros((dip_r.shape[0], 7))
    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)

        g = 1e-7 / (r2 * r2 * r2 * r2 * r)
        octp[:, 0] = 5 * x * z * (7 * (x * x - 3 * z * z) + 6 * r2)   # w_xxx
        octp[:, 1] = 15 * y * z * (7 * (x * x - z * z) + 2 * r2)      # w_xxy
        octp[:, 2] = 5 * (7 * z * z * (3 * x * x - z * z)             # w_xxz
                          - 3 * r2 * (x * x - z * z))                 #
        octp[:, 3] = 30 * x * y * (7 * z * z - r2)                    # w_xyz
        octp[:, 4] = 15 * x * z * (7 * (y * y - z * z) + 2 * r2)      # w_yyx
        octp[:, 5] = 5 * y * z * (7 * (y * y - 3 * z * z) + 6 * r2)   # w_yyy
        octp[:, 6] = 5 * (7 * z * z * (3 * y * y - z * z)             # w_yyz
                          - 3 * r2 * (y * y - z * z))                 #

        # Fill the Q array using n_col_stride = 8
        Q[i][8::n_col_stride] = g * octp[:, 0]
        Q[i][9::n_col_stride] = g * octp[:, 1]
        Q[i][10::n_col_stride] = g * octp[:, 2]
        Q[i][11::n_col_stride] = g * octp[:, 3]
        Q[i][12::n_col_stride] = g * octp[:, 4]
        Q[i][13::n_col_stride] = g * octp[:, 5]
        Q[i][14::n_col_stride] = g * octp[:, 6]

    return None
