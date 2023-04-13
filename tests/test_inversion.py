import numpy as np
import mmt_multipole_inversion.magnetic_sample as msp
import mmt_multipole_inversion.multipole_inversion as minv
from pathlib import Path
import pytest
try:
    from mmt_multipole_inversion.susceptibility_modules.cuda import cudalib as sus_cudalib
    HASCUDA = True
except ImportError:
    HASCUDA = False

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
                                sensor_origin=(0e-6, 0e-6),
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
def test_inversion_single_dipole_numba(limit):

    TEST_SAVEDIR = Path('TEST_TMP')

    # Generate arrays from the Forward model using a single dipole source
    fw_model_fun()

    inv_model = minv.MultipoleInversion(
        TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
        TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
        expansion_limit=limit,
        sus_functions_module='spherical_harmonics_basis')
    inv_model.generate_measurement_mesh()
    inv_model.compute_inversion()
    print(inv_model.Q)

    Ms = 1e5
    orientation = np.array([1., 0., 1.])
    orientation /= np.linalg.norm(orientation)
    expected_magnetization = Ms * (1 * 1e-18) * orientation

    # Compare the inverted dipole moments from the theoretical value by
    # analyzing the relative error
    print('POINT')
    for i in range(3):
        rel_diff = abs(inv_model.inv_multipole_moments[0][i] - expected_magnetization[i])
        if expected_magnetization[i] > 0:
            rel_diff /= abs(expected_magnetization[i])
        # print(rel_diff)
        print(inv_model.inv_multipole_moments[0][i])
        assert rel_diff < 1e-5


@pytest.mark.skipif(not HASCUDA, reason="CUDA not found")
@pytest.mark.parametrize("limit", LIMIT_params, ids=['dip', 'quad', 'oct'])
def test_compare_cuda_numba_populate_array(limit):
    """
    """
    TEST_SAVEDIR = Path('TEST_TMP')

    inv_model = minv.MultipoleInversion(
        TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
        TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
        expansion_limit=limit,
        sus_functions_module='spherical_harmonics_basis')
    inv_model.generate_measurement_mesh()
    inv_model.generate_forward_matrix(optimization='cuda')
    # inv_model.compute_inversion()
    Q_cuda = np.copy(inv_model.Q)

    inv_model = minv.MultipoleInversion(
        TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
        TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
        expansion_limit=limit,
        sus_functions_module='spherical_harmonics_basis')
    inv_model.generate_measurement_mesh()
    inv_model.generate_forward_matrix(optimization='numba')
    Q_numba = np.copy(inv_model.Q)

    # print(np.allclose(Q_cuda, Q_numba, rtol=1e-8, atol=1e-8))
    # print('Max', np.max(np.abs(Q_cuda - Q_numba)))

    # Absolute error using L-inf norm of the difference matrix
    # The differences increase for higher order mp susceptbs since the prefactor
    # makes large numerators: dipole ~ 1 / r^5, quadrupole ~ 1 / r^7, ...
    TOLS = {'dipole': 5e-7, 'quadrupole': 5e-2, 'octupole': 5e-2}
    inf_norm_abs_err = np.linalg.norm(Q_cuda - Q_numba, ord=np.inf)
    print('Infinity norm', inf_norm_abs_err)
    assert inf_norm_abs_err < TOLS[limit]

    # Notice that matrix values that are small, ~1e-8, differ significantly,
    # thus the relative error of these elements is super large
    # We need to check if this is a numerical error approx
    # idxs = np.where(np.abs(Q_cuda - Q_numba) > 1e-2)
    # print(idxs)
    # print(np.abs(Q_cuda - Q_numba)[109:112, :])
    # print(Q_cuda[idxs][:10])
    # print(Q_numba[idxs][:10])
    # print(np.abs(Q_cuda - Q_numba)[idxs][:10])


@pytest.mark.skipif(not HASCUDA, reason="CUDA not found")
@pytest.mark.parametrize("limit", LIMIT_params, ids=['dip', 'quad', 'oct'])
def test_inversion_single_dipole_cuda(limit):

    TEST_SAVEDIR = Path('TEST_TMP')

    # Generate arrays from the Forward model using a single dipole source
    fw_model_fun()

    inv_model = minv.MultipoleInversion(
        TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
        TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
        expansion_limit=limit,
        sus_functions_module='spherical_harmonics_basis')
    inv_model.generate_measurement_mesh()
    inv_model.generate_forward_matrix(optimization='cuda')
    inv_model.compute_inversion()

    Ms = 1e5
    orientation = np.array([1., 0., 1.])
    orientation /= np.linalg.norm(orientation)
    expected_magnetization = Ms * (1 * 1e-18) * orientation

    # Compare the inverted dipole moments from the theoretical value by
    # analyzing the relative error
    for i in range(3):
        rel_diff = abs(inv_model.inv_multipole_moments[0][i] - expected_magnetization[i])
        if expected_magnetization[i] > 0:
            rel_diff /= abs(expected_magnetization[i])
        # print(rel_diff)
        # print(inv_model.inv_multipole_moments[0][i])
        assert rel_diff < 1e-6


@pytest.mark.parametrize("limit", ['dipole', 'quadrupole'], ids=['dip', 'quad'])
def test_inversion_single_dipole_numba_sensor_3D(limit):

    TEST_SAVEDIR = Path('TEST_TMP')

    # Generate arrays from the Forward model using a single dipole source
    fw_model_fun()

    inv_model = minv.MultipoleInversion(
            TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
            TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
            expansion_limit=limit,
            sus_functions_module='spherical_harmonics_basis_volume'
            )
    inv_model.sensor_dims = (0.5e-6, 0.5e-6, 0.5e-6)
    inv_model.generate_measurement_mesh()
    inv_model.generate_forward_matrix(optimization='numba')
    print(inv_model.Q)
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
        print(inv_model.inv_multipole_moments[0][i])
        # assert rel_diff < 1e-5


@pytest.mark.parametrize("limit", ['dipole', 'quadrupole'], ids=['dip', 'quad'])
def test_inversion_single_dipole_numba_sensor_2D(limit):

    TEST_SAVEDIR = Path('TEST_TMP_2D')

    # Generate arrays from the Forward model using a single dipole source
    fw_model_fun(sensor_dx=0.1e-6, sensor_dy=0.1e-6, overwrite=True,
                 SAVEDIR=TEST_SAVEDIR)

    inv_model = minv.MultipoleInversion(
            TEST_SAVEDIR / 'MetaDict_fw_model_test_inversion.json',
            TEST_SAVEDIR / 'MagneticSample_fw_model_test_inversion.npz',
            expansion_limit=limit,
            sus_functions_module='spherical_harmonics_basis_area'
            )
    inv_model.sensor_dims = (0.1e-6, 0.1e-6)
    # inv_model.Sdx = 0.1e-6
    # inv_model.Sdy = 0.1e-6
    inv_model.generate_measurement_mesh()
    inv_model.generate_forward_matrix(optimization='numba')
    print(inv_model.Q)
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
        print(inv_model.inv_multipole_moments[0][i])
        # assert rel_diff < 1e-5


if __name__ == '__main__':
    fw_model_fun(overwrite=True)

    # test_inversion_single_dipole_numba(limit='dipole')
    # test_inversion_single_dipole_numba(limit='quadrupole')
    # test_inversion_single_dipole_numba(limit='octupole')

    # test_compare_cuda_numba_populate_array(limit='octupole')

    # test_inversion_single_dipole_cuda(limit='dipole')
    # test_inversion_single_dipole_cuda(limit='quadrupole')
    # test_inversion_single_dipole(limit='octupole')

    test_inversion_single_dipole_numba(limit='quadrupole')
    # test_inversion_single_dipole_numba_sensor_3D('quadrupole')
    test_inversion_single_dipole_numba_sensor_2D('quadrupole')
