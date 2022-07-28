# Susceptibility functions defined by expressing the multipole expansion in
# a spherical harmonic basis. The main theory is defined by Burnham and English
# in Ref. []
#
import numpy as np
import numba


# TODO: Check size of Q array
@numba.jit(nopython=True)
def dipole_Bz_sus(dip_r, pos_r, Q, n_col_stride,
                  dx_sensor, dy_sensor, dz_sensor):
    """
    Sensor-Volume calculation of the dipole susceptibility contribution
    WARNING: Q should be zeroes
    """
    f = 1e-7
    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r

        for sz in [-1., 1.]:
            for sy in [-1., 1.]:
                for sx in [-1., 1.]:
                    sign = sx * sy * sz

                    x = dr[:, 0] + sx * dx_sensor
                    y = dr[:, 1] + sy * dy_sensor
                    z = dr[:, 2] + sz * dz_sensor

                    x2, y2, z2 = x ** 2, y ** 2, z ** 2
                    r = np.sqrt(x2 + y2 + z2)

                    Q[i][::n_col_stride] += sign * np.arctanh(y / r)
                    Q[i][1::n_col_stride] += sign * np.arctanh(x / r)
                    Q[i][2::n_col_stride] += sign * (-np.arctan2(x * y, r * z))

        # Multiply by f here or for every loop? This seems more optimal:
        Q[i][::n_col_stride] *= f
        Q[i][1::n_col_stride] *= f
        Q[i][2::n_col_stride] *= f

    return None


# @numba.jit(nopython=True)
# def quadrupole_Bz_sus(dip_r, pos_r, Q, n_col_stride):
#     """
#     This function generates the quadrupolar Bz susceptibility field contributed
#     from magnetic point sources over different positions of a scan grid. The
#     method used here is populating the Q matrix which has size:
# 
#         (len(pos_r), len(dip_r) * n_col_stride)
# 
#     Methods
# 
#     See dipole_Bz_sus documentation for details of this function.
#     The Q array is populated strating from the 3rd column since the 0-2 columns
#     are reserved for the dipolar Bz susceptibility contributions
# 
#     Parameters
#     ----------
#     dip_r
#         N x 3 array OR 1 x 3 array
#     pos_r
#         M x 3 array OR 1 x 3 array
# 
#     Returns
#     -------
#     None
#         None
# 
#     """
# 
#     for i, ref_pos in enumerate(pos_r):
# 
#         dr = ref_pos - dip_r
#         x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
#         x2, y2, z2 = x ** 2, y ** 2, z ** 2
# 
#         r2 = np.sum(dr ** 2, axis=1)
#         r = np.sqrt(r2)
# 
#         # Quad Field from the Cart version of Quad field SHs, by Stone et al
#         g = 1e-7 / (r2 * r2 * r2 * r)
#         p1 = g * np.sqrt(3 / 2) * z * (-3 * r2 + 5 * z2)
#         p2 = g * -np.sqrt(2) * x * (r2 - 5 * z2)
#         p3 = g * -np.sqrt(2) * y * (r2 - 5 * z2)
#         p4 = g * (5 / np.sqrt(2)) * (x2 - y2) * z
#         p5 = g * 5 * np.sqrt(2) * x * y * z
# 
#         # Fill the Q array in the corresponding entries
#         Q[i][3::n_col_stride] = p1
#         Q[i][4::n_col_stride] = p2
#         Q[i][5::n_col_stride] = p3
#         Q[i][6::n_col_stride] = p4
#         Q[i][7::n_col_stride] = p5
# 
#     return None
# 
#     # Only return Bz
#     # return np.column_stack((g * p1, g * p2, g * p3, g * p4, g * p5))
# 
# 
