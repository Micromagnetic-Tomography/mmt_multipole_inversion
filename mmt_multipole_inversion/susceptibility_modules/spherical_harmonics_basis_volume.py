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
    r"""Populate the susceptibility matrix using sensors with 3D-cuboid geometry

    The sensors from the scanning surface are modelled as cuboids with volume.
    The susceptibility matrix is computed by integrating the Bz field in this
    volume, thus the flux is computed in units of Tesla m^3. In the main
    `MultipoleInversion` class the volume flux might be scaled by the volume to
    obtain the average volume within the scan sensor. The volume flux is
    computed as::

                      _           _  t(1)  1(1)          t(1)  3(1)    t(2)  1(2)       _
        Vol_flux =   /  dx dy dz |  Θ     P     + ... + Θ     P     + Θ     P     + ...  |
                   _/            |_  1     z             3     z       1     z          _|

                                     Dipole                            Quadrupole

    where Θ are the multipole moments (in tensor basis) and Pz are the
    z-component spherical harmonic polynomials which have to be integrated. For
    instance::

                         _   _      _
         2(2)        ∂  |  \/2  x z  |
        P      =  -  -- |  --------  |  ,  R = ( x^2 + y^2 + z^2 )^(1/2)
         z           ∂z |_    R^5   _|

    Notice that the `dz` integration can be "cancelled" with the `∂z` field
    derivative. The susceptibility matrix, therefore, will contain these
    polynomial integrated in the cuboid volume of the sensor, which are
    multiplied by the components of the magnetic moments column vector. Every
    row of the susceptibility matrix will contain the integrals for all
    magnetic sources and for one sensor position, since
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
    dx_sensor, dy_sensor, dz_sensor
        Half lengths of the sensor volume
    multipole_order
        Expansion order of the magnetic potential

    Notes
    -----
    The integration of the polynomials were obtained mostly with W. Mathematica
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
