import numpy as np
import mmt_multipole_inversion.magnetic_sample as msp


def test_fw_model_and_bz_field():
    """
    Test for the creation of a FW model from the MagneticSample class
    The Bz field is tested at a specific point of the scan grid by comparing
    the simulation to the theoretical value
    """

    Hz = 1e-6       # Scan height in m
    Sx = 5e-6       # Scan area x - dimension in m
    Sy = 5e-6       # Scan area y - dimension in m
    Sdx = 1e-6      # Scan x - step in m
    Sdy = 1e-6      # Scan y - step in m
    Lx = Sx * 1.0   # Sample x - dimension in m
    Ly = Sy * 1.0   # Sample y - dimension in m
    Lz = 5e-6       # Sample thickness in m

    # Initialise the dipole class
    sample = msp.MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz,
                                sensor_origin=(0.5e-6, 0.5e-6),
                                bz_field_module='spherical_harmonics_basis'
                                )

    # Manually set the positions and magnetization of the two dipoles
    Ms = 1e5
    dipole_positions = np.array([[sample.Lx * 0.5, sample.Ly * 0.5, -sample.Lz * 0.5]])

    magnetization = Ms * (1 * 1e-18) * np.array([[1., 0., 0.]])
    volumes = np.array([1e-18])
    sample.generate_particles_from_array(dipole_positions, magnetization, volumes)

    # Test components of the dipole moment are passed correctly:
    assert abs(sample.dipole_moments[0][0] - 1e-13) < 1e-16
    assert abs(sample.dipole_moments[0][1] - 0.0) < 1e-16
    assert abs(sample.dipole_moments[0][2] - 0.0) < 1e-16

    print('Dipole moments:', sample.dipole_moments)

    # Generate the dipole field measured as the Bz field flux through the
    # measurement surface
    sample.generate_measurement_mesh()

    # Check that the scan grid has the right dimensions
    assert len(sample.Sx_range) == 5
    assert len(sample.Sy_range) == 5
    print(sample.Sx_range)

    # Analytic calculations ---------------------------------------------------
    # mu0 = 4 * np.pi * 1e-7
    # Theoretical field computed at the point in the scan grid:
    rx, ry, rz = 3.5e-6, 3.5e-6, 1e-6
    # This is the position of the dipole center:
    x, y, z = 2.5e-6, 2.5e-6, -2.5e-6
    # We need the difference
    dx, dy, dz = rx - x, ry - y, rz - z
    r2 = dx ** 2 + dy ** 2 + dz ** 2
    r = np.sqrt(r2)
    f = 1e-7 / (r2 * r2 * r)
    g = -1e-7 / (r2 * r)

    mx, my, mz = 1e-13, 0., 0.
    sp = mx * dx  # + my * dy + mz * dz
    f = 1e-7 * 3 * sp / (r2 * r2 * r)
    g = -1e-7 / (r2 * r)
    # Bz field
    bz_field = f * dz + g * mz
    # -------------------------------------------------------------------------

    # Now test the theoretical field with the one in the FW model simulation
    # print(np.array2string(sample.Bz_array, precision=5))
    # print(bz_field)
    # print('{:.30f}'.format(abs(bz_field - sample.Bz_array[3, 3])))
    assert abs(bz_field - sample.Bz_array[3, 3]) < 1e-20


# TODO:
# Test random generation of dipole sources
# Test generation of noise in sample


if __name__ == '__main__':
    test_fw_model_and_bz_field()
