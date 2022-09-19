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
    Sensor-Area calculation of the multipole susceptibility contributions
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
                    r2 = r * r

                    # Dipole
                    if multipole_order > 0:
                        Q[i][::n_col_stride] += sign * (-y * z) / ((x2 + z2) * r)
                        Q[i][1::n_col_stride] += sign * (-x * z) / ((y2 + z2) * r)
                        Q[i][2::n_col_stride] += sign * (x * y * (r2 + z2)) / ((x2 + z2) * (y2 + z2) * r)

                    # Quadrupole
                    if multipole_order > 1:
                        x4, y4, z4 = x2 * x2, y2 * y2, z2 * z2
                        Q[i][3::n_col_stride] += sign * (1. / np.sqrt(6.)) * (x * y * z * (2 * x4 * x2 + 3 * x4 * y2 + 3 * x2 * y4 + 2 * y4 * y2 + (7 * x4 + 12 * x2 * y2 + 7 * y4) * z2 + 11 * (x2 + y2) * z4 + 6 * z4 * z2)) / ((x2 + z2)**2 * (y2 + z2)**2 * r2  r)
                        Q[i][4::n_col_stride] += sign * (np.sqrt(2) * y * (x2 * (x2 + y2) - (x2 + y2) * z2 - 2 * z4)) / (3. * (x2 + z2)**2 * r2 * r)
                        Q[i][5::n_col_stride] += sign * (np.sqrt(2) * x * (y2 * (x2 + y2) - (x2 + y2) * z2 - 2 * z4)) / (3. * (y2 + z2)**2 * r2 * r)

                        Q[i][6::n_col_stride] += sign * (-1. / (np.sqrt(2.) * 3.)) * x * y * (x2 - y2) / ((x2 + z2) * (y2 + z2) * r)
                        Q[i][7::n_col_stride] += sign * (-np.sqrt(2.) / 3.) / r

    # Multiply by f here or for every loop? This seems more optimal:
    Q *= f

    return None
