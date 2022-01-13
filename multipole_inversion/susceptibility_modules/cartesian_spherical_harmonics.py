# Susceptibility functions defined by expressing the multipole expansion
# using real spherical harmonics and converting them to Cartesian coordinates.
# The difference with the Spherical Harmonic Basis file, is that these
# polynomials are taken directly from the SH basis used in Physics and
# normalisation might be different [UPDATE THIS DISCUSSION]
import numpy as np
import numba


@numba.jit(nopython=True)
def dipole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    This function generates the dipolar Bz susceptibility field contributed
    from magnetic point sources over different positions of a scan grid. The
    method used here is populating the Q matrix which has size::

        (len(pos_r), len(dip_r) * n_col_stride)

    Inputs ::

        dip_r   :: N x 3 array OR 1 x 3 array
        pos_r   :: M x 3 array OR 1 x 3 array

    Returns
    -------
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
        # x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)

        # Based on Real Spherical Harmonics Y(th, ph): (D / Dz) Y
        f = 1e-7 / (r2 * r2 * r)
        p1 = f * np.sqrt(6) * (3 * x * z)
        p2 = f * -np.sqrt(6) * (3 * y * z)
        p3 = f * np.sqrt(3 / 2) * (3 * z * z - r2)

        #  Only return Bz
        Q[i][::n_col_stride] = p1
        Q[i][1::n_col_stride] = p2
        Q[i][2::n_col_stride] = p3

    return None


@numba.jit(nopython=True)
def quadrupole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    dip_r   :: N x 3 array OR 1 x 3 array
    pos_r   :: M x 3 array OR 1 x 3 array

    Returns
    -------
    None

    """

    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)

        g = 1e-7 / (r2 * r2 * r2 * r)
        p1 = g * -np.sqrt(5) * z * (3 * r2 - 5 * z2)
        p5 = g * np.sqrt(15) * y * (r2 - 5 * z2)
        p4 = g * 5 * np.sqrt(10) * z * (x2 - y2)
        p3 = g * -np.sqrt(15) * x * (r2 - 5 * z2)
        p2 = g * -10 * np.sqrt(10) * x * y * z

        # Fill the Q array
        Q[i][3::n_col_stride] = p1
        Q[i][4::n_col_stride] = p2
        Q[i][5::n_col_stride] = p3
        Q[i][6::n_col_stride] = p4
        Q[i][7::n_col_stride] = p5

    return None


@numba.jit(nopython=True)
def octupole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
    """
    dip_r   :: N x 3 array OR 1 x 3 array
    pos_r   :: M x 3 array OR 1 x 3 array

    Returns
    -------
    ndarray
        N x 7  array with 7 octupole moments from the traceless 3-tensor

    """

    octp = np.zeros((dip_r.shape[0], 7))
    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r4 = r2 ** 2
        r = np.sqrt(r2)

        g = 1e-7 / (r4 * r4 * r)
        octp[:, 0] = np.sqrt(7 / 10) * (3 * r4 - 30 * r2 * z2 + 35 * z2 * z2)
        octp[:, 1] = 5 * np.sqrt(42 / 19) * y * z * (3 * r2 - 7 * z2)
        octp[:, 2] = -np.sqrt(35) * (x2 - y2) * (r2 - 7 * z2)
        octp[:, 3] = 7 * np.sqrt(14) * y * (-3 * x2 + y2) * z
        octp[:, 4] = 5 * np.sqrt(42 / 19) * x * z * (-3 * r2 + 7 * z2)
        octp[:, 5] = 2 * np.sqrt(35) * x * y * (r2 - 7 * z2)
        octp[:, 6] = 7 * np.sqrt(14) * x * (x2 - 3 * y2) * z

        # Fill the Q array using n_col_stride = 8
        Q[i][8::n_col_stride] = g * octp[:, 0]
        Q[i][9::n_col_stride] = g * octp[:, 1]
        Q[i][10::n_col_stride] = g * octp[:, 2]
        Q[i][11::n_col_stride] = g * octp[:, 3]
        Q[i][12::n_col_stride] = g * octp[:, 4]
        Q[i][13::n_col_stride] = g * octp[:, 5]
        Q[i][14::n_col_stride] = g * octp[:, 6]

    return None
