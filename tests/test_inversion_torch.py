import numpy as np
import mmt_multipole_inversion.magnetic_sample as msp
import mmt_multipole_inversion.multipole_inversion as minv
from pathlib import Path
import pytest
import matplotlib.pyplot as plt

LIMIT_params = ['dipole', 'quadrupole', 'octupole']


def fw_model_fun(sensor_dx=1e-6, sensor_dy=1e-6, overwrite=False,
                 SAVEDIR='TEST_TMP'):
    """
    """

    TEST_SAVEDIR = Path(SAVEDIR)
    TEST_SAVEDIR.mkdir(exist_ok=True)

    # Check if save dir is empty (wont check for specific npz and json names)
    if not overwrite and any(TEST_SAVEDIR.iterdir()):
        print(f'Save dir {TEST_SAVEDIR} not empty, skipping this function')
        return

    Hz = 1e-6         # Scan height in m
    Sx = 20e-6        # Scan area x - dimension in m
    Sy = 20e-6        # Scan area y - dimension in m
    Sdx = sensor_dx   # Scan x - step in m
    Sdy = sensor_dy   # Scan y - step in m
    Lx = Sx * 1.0     # Sample x - dimension in m
    Ly = Sy * 1.0     # Sample y - dimension in m
    Lz = 10e-6        # Sample thickness in m

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


@pytest.mark.parametrize("limit", LIMIT_params, ids=['dip', 'quad', 'oct'])
def test_inversion_single_dipole_torch(limit):

    TEST_SAVEDIR = Path('TEST_TMP')

    # Generate arrays from the Forward model using a single dipole source
    fw_model_fun()

    inv_model = minv.MultipoleInversion(
        TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
        TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
        expansion_limit=limit,
        sus_functions_module='spherical_harmonics_basis')
    inv_model.generate_measurement_mesh()
    inv_model.compute_inversion(method='torchmin')
    print(inv_model.Q)

    Ms = 1e5
    orientation = np.array([1., 0., 1.])
    orientation /= np.linalg.norm(orientation)
    expected_magnetization = Ms * (1 * 1e-18) * orientation

    # Compare the inverted dipole moments from the theoretical value by
    # analyzing the relative error
    print(f'{expected_magnetization=}')
    for i in range(3):
        rel_diff = abs(inv_model.inv_multipole_moments[0][i] - expected_magnetization[i])
        if expected_magnetization[i] > 0:
            rel_diff /= abs(expected_magnetization[i])

        print(f'{rel_diff=}')
        print(f'inv {i} = {inv_model.inv_multipole_moments[0][i]}')
        # assert rel_diff < 1e-5

    print(f'{rel_diff=}')
    print(f'{inv_model.inv_multipole_moments[0]=}')

    print(inv_model.inv_Bz_array.flatten()[:10])
    print(inv_model.Bz_array.flatten()[:10])
    f, axs = plt.subplots(ncols=2)
    axs[0].imshow(inv_model.inv_Bz_array, origin='lower', cmap='RdYlBu')
    axs[1].imshow(inv_model.Bz_array, origin='lower', cmap='RdYlBu')
    plt.show()


if __name__ == '__main__':
    fw_model_fun(overwrite=True)

    # test_inversion_single_dipole_torch(limit='dipole')
    test_inversion_single_dipole_torch(limit='quadrupole')
