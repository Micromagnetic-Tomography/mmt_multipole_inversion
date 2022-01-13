# Susceptibility functions defined by expressing the multipole expansion in
# a spherical harmonic basis. The main theory is defined by Burnham and English
# in Ref. []
#
import numpy as np
import numba


# TODO: Check size of Q array
@numba.jit(nopython=True)
def dipole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    This function generates the dipolar Bz susceptibility field contributed
    from magnetic point sources over different positions of a scan grid. The
    method used here is populating the Q matrix which has size::

        (len(pos_r), len(dip_r) * n_col_stride)

    Method

    For every position in the pos_r array, the function computes the susc Bz
    contribution from all the magnetic sources at positions in dip_r. These
    susc components are stored in the Q array. Every row i has the dipolar
    contributions from all the sources at pos_r[i]. The stride value
    n_col_stride is necessary to "jump" over array cells reserved to store
    quadrupolar or octupolar susc contributions.
    If::

        pos_r = [r0 r1 r2 ... rN]

    and if n_col_stride=5 (so we have space to store quadrupolar Bz sus)
    then Q is populated as::

        Q =  _                                                                                                      _
            | mx_0(r0)  my_0(r0)  mz_0(r0)  0  0  0  0  0  mx_1(r0)  my_1(r0)  mz_1(r0)  0  0 ... mz_M(r0) 0 0 0 0 0 |
            | mx_0(r1)  my_0(r1)  mz_0(r1)  0  0  0  0  0  mx_1(r0)  my_1(r0)  mz_1(r0)  0  0 ...                    |
            | mx_0(r2)  ...                                                                                          |
            |                                                                                                        |
            |   ...                                                                                                  |
            |_mx_0(rN)                                                                                              _|

    where mx_j(ri) is the x-component of the Bz susc contribution from the
    dipole located at dip_r[j]
    Notice that if Q also stores octupolar fields, then we would have strides
    of 12 zeros (5 quad + 7 oct moments), so n_col_stride=15

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

    Notes
    -----
    Calculate magnetic flux Bz-susceptibility per dipole component generated
    by dipoles located in position dip_r (m) at position  pos_r (m)
    Units of result is T / (A m2)

    """
    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)
        f = 1e-7 / (r2 * r2 * r)

        # Original ones: (WORKING)
        p3 = f * (3 * z2 - r2)
        p2 = f * (3 * y * z)
        p1 = f * (3 * x * z)

        #  Only return Bz
        Q[i][::n_col_stride] = p1
        Q[i][1::n_col_stride] = p2
        Q[i][2::n_col_stride] = p3

    return None


@numba.jit(nopython=True)
def quadrupole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    This function generates the quadrupolar Bz susceptibility field contributed
    from magnetic point sources over different positions of a scan grid. The
    method used here is populating the Q matrix which has size:

        (len(pos_r), len(dip_r) * n_col_stride)

    Methods

    See dipole_Bz_sus documentation for details of this function.
    The Q array is populated strating from the 3rd column since the 0-2 columns
    are reserved for the dipolar Bz susceptibility contributions

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
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)

        # Quad Field from the Cart version of Quad field SHs, by Stone et al
        g = 1e-7 / (r2 * r2 * r2 * r)
        p1 = g * np.sqrt(3 / 2) * z * (-3 * r2 + 5 * z2)
        p2 = g * -np.sqrt(2) * x * (r2 - 5 * z2)
        p3 = g * -np.sqrt(2) * y * (r2 - 5 * z2)
        p4 = g * (5 / np.sqrt(2)) * (x2 - y2) * z
        p5 = g * 5 * np.sqrt(2) * x * y * z

        # Fill the Q array in the corresponding entries
        Q[i][3::n_col_stride] = p1
        Q[i][4::n_col_stride] = p2
        Q[i][5::n_col_stride] = p3
        Q[i][6::n_col_stride] = p4
        Q[i][7::n_col_stride] = p5

    return None

    # Only return Bz
    # return np.column_stack((g * p1, g * p2, g * p3, g * p4, g * p5))


@numba.jit(nopython=True)
def octupole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    This function generates the octupolar Bz susceptibility field contributed
    from magnetic point sources over different positions of a scan grid. The
    method used here is populating the Q matrix which has size:

        (len(pos_r), len(dip_r) * n_col_stride)

    Methods

    See dipole_Bz_sus documentation for details of this function.
    The Q array is populated strating from the 8th column since the 0-2 columns
    are reserved for the dipolar Bz susceptibility contributions and
    columns 3-7 for the quadrupolar contributions.

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
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r4 = r2 ** 2
        r = np.sqrt(r2)

        # Oct Field from the Cartesian version of Octupole field SHs, by Stone et al
        g = 1e-7 / (r4 * r4 * r)
        octp[:, 0] = (3 * (r2 ** 2) - 30 * r2 * z2 + 35 * (z2 * z2)) / np.sqrt(10)
        octp[:, 1] = np.sqrt(15) * x * z * (-3 * r2 + 7 * z2) / 2
        octp[:, 2] = np.sqrt(15) * y * z * (-3 * r2 + 7 * z2) / 2
        octp[:, 3] = -np.sqrt(1.5) * (x2 - y2) * (r2 - 7 * z2)
        octp[:, 4] = -np.sqrt(6) * x * y * (r2 - 7 * z2)
        octp[:, 5] = 7 * x * (x2 - 3 * y2) * z / 2
        octp[:, 6] = -7 * y * (-3 * x2 + y2) * z / 2

        # Fill the Q array using n_col_stride = 8
        Q[i][8::n_col_stride] = g * octp[:, 0]
        Q[i][9::n_col_stride] = g * octp[:, 1]
        Q[i][10::n_col_stride] = g * octp[:, 2]
        Q[i][11::n_col_stride] = g * octp[:, 3]
        Q[i][12::n_col_stride] = g * octp[:, 4]
        Q[i][13::n_col_stride] = g * octp[:, 5]
        Q[i][14::n_col_stride] = g * octp[:, 6]

    return None
