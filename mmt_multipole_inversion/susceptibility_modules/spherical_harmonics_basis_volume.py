# Susceptibility functions defined by expressing the multipole expansion in
# a spherical harmonic basis. The main theory is defined by Burnham and English
# in Ref. []
#
import numpy as np
import numba


# TODO: Check size of Q array
@numba.jit(nopython=True)
def multipole_Bz_sus(dip_r, pos_r, Q, n_col_stride,
                     dx_sensor, dy_sensor, dz_sensor,
                     multipole_order):
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

                    # Dipole
                    if multipole_order > 0:
                        Q[i][::n_col_stride] += sign * np.arctanh(y / r)
                        Q[i][1::n_col_stride] += sign * np.arctanh(x / r)
                        Q[i][2::n_col_stride] += sign * (-np.arctan2(x * y, r * z))

                    # Quadrupole
                    if multipole_order > 1:
                        Q[i][3::n_col_stride] += sign * (-1 / np.sqrt(6.)) * (x * y * (r * r + z2)) / ((x2 + z2) * (y2 + z2) * r)
                        Q[i][4::n_col_stride] += sign * (np.sqrt(2.) / 3.) * y * z / (r * (x2 + z2))
                        Q[i][5::n_col_stride] += sign * (np.sqrt(2.) / 3.) * x * z / (r * (y2 + z2))
                        Q[i][6::n_col_stride] += sign * (-1. / (np.sqrt(2.) * 3.)) * x * y * (x2 - y2) / ((x2 + z2) * (y2 + z2) * r)
                        Q[i][7::n_col_stride] += sign * (-np.sqrt(2.) / 3.) / r

    # Multiply by f here or for every loop? This seems more optimal:
    Q *= f

    return None
