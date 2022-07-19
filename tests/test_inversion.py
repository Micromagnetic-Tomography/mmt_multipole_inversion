import numpy as np
import mmt_multipole_inversion.magnetic_sample as msp
import mmt_multipole_inversion.multipole_inversion as minv
from pathlib import Path


TEST_SAVEDIR = Path('TEST_TMP')
TEST_SAVEDIR.mkdir(exist_ok=True)


def fw_model_fun():
    """
    """

    Hz = 1e-6       # Scan height in m
    Sx = 20e-6      # Scan area x - dimension in m
    Sy = 20e-6      # Scan area y - dimension in m
    Sdx = 1e-6      # Scan x - step in m
    Sdy = 1e-6      # Scan y - step in m
    Lx = Sx * 1.0   # Sample x - dimension in m
    Ly = Sy * 1.0   # Sample y - dimension in m
    Lz = 10e-6      # Sample thickness in m

    # Initialise the dipole class
    sample = msp.MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz,
                                scan_origin=(0e-6, 0e-6),
                                bz_field_module='spherical_harmonics_basis'
                                )

    # Manually set the positions and magnetization of the two dipoles
    dipole_positions = np.array([[sample.Lx * 0.5,
                                  sample.Ly * 0.5,
                                  -sample.Lz * 0.5]])

    Ms = 1e5
    orientation = np.array([1., 0., 1.])
    orientation /= np.linalg.norm(orientation)
    magnetization = Ms * (1 * 1e-18) * np.array([orientation])
    volumes = np.array([1e-18])
    sample.generate_particles_from_array(dipole_positions,
                                         magnetization,
                                         volumes)

    # Generate the dipole field measured as the Bz field flux through the
    # measurement surface
    sample.generate_measurement_mesh()

    sample.save_data(filename='fw_model_test_inversion', basedir=TEST_SAVEDIR)


def test_inversion_single_dipole(limit='dipole'):

    inv_model = minv.MultipoleInversion(
            TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
            TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
            expansion_limit=limit,
            sus_functions_module='spherical_harmonics_basis'
            )
    inv_model.generate_measurement_mesh()
    inv_model.compute_inversion()

    Ms = 1e5
    orientation = np.array([1., 0., 1.])
    orientation /= np.linalg.norm(orientation)
    expected_magnetization = Ms * (1 * 1e-18) * orientation

    # Compare the inverted dipole moments from the theoretical value by
    # analyzing the relative error
    for i in range(3):
        rel_diff = abs(inv_model.inv_multipole_moments[0][i] -
                       expected_magnetization[i])
        if expected_magnetization[i] > 0:
            rel_diff /= abs(expected_magnetization[i])
        # print(rel_diff)
        # print(inv_model.inv_multipole_moments[0][i])
        assert rel_diff < 1e-5


if __name__ == '__main__':
    # Generate arrays from the Forward model using a single dipole source
    fw_model_fun()

    test_inversion_single_dipole(limit='dipole')
    test_inversion_single_dipole(limit='quadrupole')
    test_inversion_single_dipole(limit='octupole')
