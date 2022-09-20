# Susceptibility functions defined by expressing the multipole expansion in
# a spherical harmonic basis. The main theory is defined by Burnham and English
# in Ref. []
#
import numpy as np
import numba


# TODO: Check size of Q array
@numba.jit(nopython=True)
def multipole_Bz_sus(dip_r, pos_r, Q, n_col_stride,
                     dx_sensor, dy_sensor,
                     multipole_order):
    r"""Populate the susceptibility matrix using 2D rectangular sensors

    The susceptibility matrix is computed by integrating the Bz field in the a
    rectangular area in the `xy`-plane, thus the flux is computed in units of
    Tesla m^2. In the main `MultipoleInversion` class the area flux might be
    scaled by the sensor area to obtain the average flux within the scan
    sensor. The area flux is computed as::

                       _        _  t(2)  1(2)     t(2)  2(2)          t(3)  1(3)       _
        Area_flux =   /  dx dy |  Θ     P     +  Θ     P     + ... + Θ     P     + ...  |
                    _/         |_  1     z        2     z             1     z          _|

                                   Dipole                            Quadrupole

    where Θ are the multipole moments (in tensor basis) and Pz are the
    z-component spherical harmonic polynomials which have to be integrated. For
    instance::

                         _   _      _
         2(2)        ∂  |  \/2  x z  |
        P      =  -  -- |  --------  |  ,  R = ( x^2 + y^2 + z^2 )^(1/2)
         z           ∂z |_    R^5   _|

    The susceptibility matrix, therefore, will contain these polynomial
    integrated in the xy-area of the sensor, which are multiplied by the
    components of the magnetic moments column vector. Every row of the
    susceptibility matrix will contain the integrals for all magnetic sources
    and for one sensor position, since
    `r = (x, y, z) = sensor_pos - magnetic_source_pos`.

    Parameters
    ----------
    dip_r
        `N x 3` array with the positions of the magnetic point sources
    pos_r
        `P x 3` array with the positions of `P` sensors that define the
        scanning surface
    Q
        Susceptibility / Forward matrix to be populated. Check that `Q` entries
        are zero before calling this function
    n_col_stride
        Number of column strides to populate the `Q` matrix. This is defined
        by the multipole order of the potential expansion
    dx_sensor, dy_sensor
        Half lengths of the sensor area
    multipole_order
        Expansion order of the magnetic potential

    Notes
    -----
    The integration of the polynomials were obtained mostly with W. Mathematica
    """
    f = 1e-7
    for i, ref_pos in enumerate(pos_r):

        dr = ref_pos - dip_r

        for sy in [-1., 1.]:
            for sx in [-1., 1.]:
                sign = sx * sy

                x = dr[:, 0] + sx * dx_sensor
                y = dr[:, 1] + sy * dy_sensor
                z = dr[:, 2]

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
                    x2_p_z2_sq = x4 + z4 + 2 * x2 * z2
                    y2_p_z2_sq = y4 + z4 + 2 * y2 * z2
                    r3 = r2 * r

                    Q[i][3::n_col_stride] += sign * (1. / np.sqrt(6.)) * (x * y * z * (2. * x4 * x2 + 3. * x4 * y2 + 3. * x2 * y4 + 2. * y4 * y2 + (7. * x4 + 12. * x2 * y2 + 7. * y4) * z2 + 11. * (x2 + y2) * z4 + 6. * z4 * z2)) / (x2_p_z2_sq * y2_p_z2_sq * r3)
                    Q[i][4::n_col_stride] += sign * np.sqrt(2.) * y * ((x2 - z2) * (x2 + y2) - 2. * z4) / (3. * x2_p_z2_sq * r3)
                    Q[i][5::n_col_stride] += sign * np.sqrt(2.) * x * ((y2 - z2) * (x2 + y2) - 2. * z4) / (3. * y2_p_z2_sq * r3)
                    Q[i][6::n_col_stride] += sign * (x * (x2 - y2) * y * z * (2. * x4 + 5. * x2 * y2 + 2. * y4 + 7. * (x2 + y2) * z2 + 5. * z4)) / (3. * np.sqrt(2) * x2_p_z2_sq * y2_p_z2_sq * r3)
                    # Q[i][6::n_col_stride] += sign * (x * (x2 - y2) * y * z * (r2 * r2 + x2 * y2 + 3 * z2 * r2) / (3. * np.sqrt(2) * x2_p_z2_sq * y2_p_z2_sq * r2 * r)
                    Q[i][7::n_col_stride] += sign * (np.sqrt(2.) * z) / (3. * r3)

    # Multiply by f here or for every loop? This seems more optimal:
    Q *= f

    return None
