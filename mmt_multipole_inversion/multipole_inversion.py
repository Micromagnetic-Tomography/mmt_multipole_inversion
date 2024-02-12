import numpy as np
import time
# import datetime
import json
# from scipy.special import sph_harm
import scipy.linalg as slin
# import warnings
# try:
#     import tensorflow as tf
# except ModuleNotFoundError:
#     warnings.warn('Could not import Tensorflow')
from pathlib import Path

# CUDA modules for populating the suscept matrix (if available)
try:
    from .susceptibility_modules.cuda import cudalib as sus_cudalib
    HASCUDA = True
except ImportError:
    HASCUDA = False

# Suscept modules using Numba:
from . import susceptibility_modules as sus_mods

from typing import Optional
from typing import Literal  # Working with Python >3.8
from typing import Union    # Working with Python >3.8
from . import plot_tools as pt

# -----------------------------------------------------------------------------

_SusOptions = Literal['spherical_harmonics_basis',
                      'maxwell_cartesian_polynomials',
                      'cartesian_spherical_harmonics',
                      'spherical_harmonics_basis_area',
                      'spherical_harmonics_basis_volume'
                      ]
_ExpOptions = Literal['dipole', 'quadrupole', 'octupole']
_MethodOptions = Literal['numba', 'cuda']
_InvMethodOps = Literal['np_pinv', 'sp_pinv', 'sp_pinv2']


# TODO: verbose should be more useful for debugging
class MultipoleInversion(object):
    """Class to perform multipole inversions

    Class to perform multipole inversions of a magnetic scan surface into
    multiple magnetic sources located within a sample. Specifications of
    the scan grid and the magnetic particles in the sample can be generated
    using the `MagneticSample` class. The sensors of the magnetic scan surface
    are modelled either as point sensors or sensor with cuboid shape (volume).
    """

    def __init__(self,
                 sample_config_file: Union[str, Path],
                 sample_arrays: Optional[Union[str, Path]],  # TODO: set to npz file
                 expansion_limit: _ExpOptions = 'quadrupole',
                 verbose: bool = True,
                 sus_functions_module: _SusOptions = 'spherical_harmonics_basis'
                 ) -> None:
        """
        Parameters
        ----------
        sample_config_file
            Path to a `json` file with the specifications of the scan grid and
            the magnetic particles. The following keys are mandatory::
                Scan height Hz
                Scan area x-dimension Sx
                Scan area y-dimension Sy
                Scan x-step Sdx
                Scan y-step Sdy
                Number of particles
                Time stamp
            The following are optional::
                Sensor origin x
                Sensor origin y
                Sensor dimensions
            Sensor dimensions are required if a 2D or 3D sensor is used. The
            sensor origins are the coordinates of the lower left sensor center,
            in the scanning surface. By default it is `(0.0, 0.0)`. The sensor
            grid in each dimension (`x` or `y`) is computed as multiples of
            `Scan area / Scan step`
        sample_arrays
            An `npz` file containing the scan signal Bz and the particle
            positions (magnetic sources). The file can contain other
            information as well but it is not read here. If the two arrays are
            not specified they can be set manually using the `Bz_array` and
            `particle_positions` class variables.
        expansion_limit
            Higher order multipole term to compute the field contribution from
            the potential of the magnetic particles. Options:
                `dipole`, `quadrupole`, `octupole`
        verbose
            Print extra information about the calculations.
        sus_functions_module
            Spherical harmonic basis for the susceptibility matrix used for the
            multipole inversion. The fully orthogonal and linearly independent
            basis is the `spherical_harmonics_basis`. Other options are not
            orthogonal but might be necessary for comparison. These modules
            populate the forward matrix with the assumption that sensors are a
            point, which can be seen as sensors with an area or volume where
            the magnetic flux from the sources is treated as constant.
            Alternatively, susceptibility modules ending with `_area` or
            `_volume` model the sensors using a geometry with a higher
            dimension, e.g. a rectangle, where the magnetic flux is integrated
            within it. For details see the comments in the libraries in the
            `sus_functions_module/` directory and the Notes.

        Notes
        -----
        Mathematical details for the multipole inversion can be found in::

            D. Cortés-Ortuño, K. Fabian, L. V. de Groot
            Single Particle Multipole Expansions From Micromagnetic Tomography
            G^3, 22(4), e2021GC009663 (2021)
            https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021GC009663

        """

        # Set the module from which to find the Bz susceptibility functions,
        # which is specified using the sus_functions_module argument
        self.sus_mod = getattr(sus_mods, sus_functions_module)

        self._expansion_limit = 'dipole'  # set default value
        self.expansion_limit = expansion_limit  # update def value

        # Turn log messages on/off. TODO: Wrap print into a function to avoid
        # multiple if statements
        self.verbose = verbose

        expected_arrays = ['Bz_array', 'particle_positions']
        self.Bz_array = np.empty(0)
        self.particle_positions = None

        # Any other array in the NPZ file will be loaded here
        if sample_arrays:
            data = np.load(sample_arrays)
            for key, value in data.items():
                setattr(self, key, value)

            for key in expected_arrays:
                if key not in data.keys():
                    if self.verbose:
                        print(f'*{key}* array required for calculations. '
                              ' Set manually.')

        # Optional sequence to set the origin of scan positions
        # self.scan_origin = (0.0, 0.0)

        # Load metadata
        with open(sample_config_file, 'r') as f:
            metadict = json.load(f)

        # Set the following keys from the json file as class parameters with
        # the 'value' as name, e.g. scan height is set in self.Hz
        multInVDict = {"Scan height Hz": 'Hz',
                       "Scan area x-dimension Sx": 'Sx',
                       "Scan area y-dimension Sy": 'Sy',
                       "Scan x-step Sdx": 'Sdx',
                       "Scan y-step Sdy": 'Sdy',
                       "Number of particles": 'N_particles',
                       "Time stamp": 'time_stamp',
                       # Set the variables sensor_origin_x / y  if
                       # found in the json file dict, otherwise set them to 0.0
                       "Sensor origin x": ('sensor_origin_x', 0.0),
                       "Sensor origin y": ('sensor_origin_y', 0.0),
                       # Sensor dimensions (empty tuple if not specified)
                       "Sensor dimensions": ('sensor_dims', ()),
                       }

        for k, v in multInVDict.items():
            if isinstance(v, tuple):
                setattr(self, v[0], metadict.get(k, v[1]))
            else:
                setattr(self, v, metadict.get(k))

            if self.verbose:
                if k not in metadict.keys():
                    print(f'Parameter {k} not found in json file')
                    if isinstance(v, tuple):
                        print(f'Setting {k} value to {v[1]}')

        for k in metadict.keys():
            if self.verbose:
                if k not in multInVDict.keys():
                    print(f'Not using parameter {k} in json file')

        self.generate_measurement_mesh()

        # Instantiate the forward matrix
        self.Q = np.empty(0)

    @property
    def expansion_limit(self):
        return self._expansion_limit

    @expansion_limit.setter
    def expansion_limit(self, string_value: str):
        # This will determine the number of columns in the forward calculation
        if string_value == 'dipole':
            self._N_cols = 3
        elif string_value == 'quadrupole':
            self._N_cols = 8
        elif string_value == 'octupole':
            self._N_cols = 15
        else:
            raise Exception('Specify a valid expansion limit for the multipole calculation')
        self._expansion_limit = string_value

        # Reset the Q matrix whose size depends on _N_cols
        self.Q = np.empty(0)

    def generate_measurement_mesh(self):
        """Generate coordinates for the measurement mesh

        The number of grid points in each direction (`xy`) are calculated by
        rounding the lateral size by the grid step size, e.g. `round(Sx / Sdx)`
        """

        # Generate measurement mesh
        self.Sx_range = self.sensor_origin_x + np.arange(round(self.Sx / self.Sdx)) * self.Sdx
        self.Sy_range = self.sensor_origin_y + np.arange(round(self.Sy / self.Sdy)) * self.Sdy
        self.Nx_surf = len(self.Sx_range)
        self.Ny_surf = len(self.Sy_range)

        if self.verbose:
            print(f'Scanning array size = {len(self.Sx_range)} x {len(self.Sy_range)}')
        self.N_sensors = self.Nx_surf * self.Ny_surf

    def generate_forward_matrix(self,
                                optimization: _MethodOptions = 'numba'):
        """
        Generate the forward matrix adding the field contribution from all
        the particles for every grid point at the scan surface. The field is
        computed from the scalar potential of the particles approximated with
        the multipole expansion up to the order specified by
        self.expansion_limit

        Parameters
        ----------
        optimization
            The method to optimize the calculation of the matrix elements:
            `numba` or `cuda`

        Notes
        -----
        In case of using one of the `_area` or `_volume` susceptibility
        modules, where the sensor is modelled in 2D or 3D, remember to specify
        the `self.sensor_dims` tuple with the dimensions of the sensor
        """
        # WARNING: Not checking wrong types here, we rely on Type hints

        # Moment vec m = [mx[0], my[0], mz[0], ... , mx[N-1], my[N-1], mz[N-1]]
        # Position vector  p = [(x[0], y[0]),  ... , x[
        # Generate  forward matrix
        # Q[i, j] =

        # The total flux array according to the specified expansion limit
        self.Q = np.zeros(shape=(self.N_sensors, self._N_cols * self.N_particles))

        # print('pos array:', particle_positions.shape)
        t0 = time.time()

        # Create all the positions of the scan grid
        scan_positions = np.ones((self.N_sensors, 3))
        X_pos, Y_pos = np.meshgrid(self.Sx_range, self.Sy_range)
        scan_positions[:, :2] = np.stack((X_pos, Y_pos), axis=2).reshape(-1, 2)
        scan_positions[:, 2] *= self.Hz
        mp_order = {'dipole': 1, 'quadrupole': 2, 'octupole': 3}

        if optimization == 'cuda':
            if len(self.sensor_dims) == 0:
                if HASCUDA is False:
                    raise RuntimeError('The cuda method is not available. Stopping calculation')

                sus_cudalib.SHB_populate_matrix(self.particle_positions,
                                                scan_positions,
                                                self.Q,
                                                self.N_particles, self.N_sensors,
                                                mp_order[self.expansion_limit],
                                                self.verbose)

        # For all the particles, whose positions are stored in the pos array
        # (N_particles x 3), compute the dipole (3 terms), quadrupole (5 terms)
        # or octupole (7 terms) contributions. Here we populate the Q array
        # using the numba-optimised susceptibility functions
        elif optimization == 'numba':
            if len(self.sensor_dims) == 0:
                self.sus_mod.dipole_Bz_sus(self.particle_positions, scan_positions,
                                           self.Q, self._N_cols)
                if self.expansion_limit in ['quadrupole', 'octupole']:
                    self.sus_mod.quadrupole_Bz_sus(self.particle_positions,
                                                   scan_positions,
                                                   self.Q, self._N_cols)
                if self.expansion_limit in ['octupole']:
                    self.sus_mod.octupole_Bz_sus(self.particle_positions,
                                                 scan_positions,
                                                 self.Q, self._N_cols)

            # AREA SENSOR
            elif len(self.sensor_dims) == 2:
                if self._expansion_limit == 'octupole':
                    self.Q = np.empty(0)
                    raise ValueError('Octupole expansion_limit for area sensors not implemented')
                self.sus_mod.multipole_Bz_sus(self.particle_positions, scan_positions,
                                              self.Q, self._N_cols,
                                              *self.sensor_dims,
                                              mp_order[self.expansion_limit]
                                              )
                aream = 1 / (4 * self.sensor_dims[0] * self.sensor_dims[1])
                # Convert area flux to average flux per sensor
                np.multiply(self.Q, aream, out=self.Q)

            # VOLUME SENSOR
            elif len(self.sensor_dims) == 3:
                if self._expansion_limit == 'octupole':
                    self.Q = np.empty(0)
                    raise ValueError('Octupole expansion_limit for volume sensors not implemented')
                self.sus_mod.multipole_Bz_sus(self.particle_positions, scan_positions,
                                              self.Q, self._N_cols,
                                              *self.sensor_dims,
                                              mp_order[self.expansion_limit]
                                              )
                volm = 1 / (8 * self.sensor_dims[0] * self.sensor_dims[1] * self.sensor_dims[2])
                # Convert volume flux to average flux per sensor
                np.multiply(self.Q, volm, out=self.Q)
            else:
                raise ValueError('Wrong sensor dimensions')
        else:
            raise ValueError(f'Optimization {optimization} not valid')

        t1 = time.time()
        if self.verbose:
            print(f'Generation of Q matrix took: {t1 - t0:.4f} s')
        # print('Q shape:', Q.shape)

    def compute_inversion(self,
                          method: _InvMethodOps = 'sp_pinv',
                          sigma_field_noise: Optional[float] = None,
                          **method_kwargs
                          ):
        """
        Computes the multipole inversion. Results are saved in the
        `inv_multipole_moments` and `inv_Bz_array` variables. This method
        requires the generation of the `Q` matrix, hence the
        `generate_forward_matrix` method using `numba` is called if `Q` has not
        been set. To optimize the calculation of `Q`, call the function before
        this method.

        Parameters
        ----------
        method
            The numerical method to perform the inversion. Options:
                np_pinv  -> Numpy's pinv
                sp_pinv  -> Scipy's pinv (not recommended -> memory issues)
                sp_pinv2 -> Scipy's pinv2 (this will call sp_pinv instead)
        sigma_field_noise
            If a `float` is specified, a covariance matrix is produced and
            stored in the `covariance_matrix` variable. This matrix uses
            the value of `sigma` as the standard deviation of uncorrelated
            noise in the magnetic flux field. Units are T m^2. In addition, the
            standard deviation of the magnetic moments are calculated and
            stored in the `inv_moments_std` 2D array where every row has the
            results per grain. For details, see
            [F. Out et al. Geochemistry, Geophysics, Geosystems 23(4). 2022]
        **method_kwargs
            Extra parameters passed to Numpy or Scipy functions. For Numpy, the
            tolerance can be set using `rcond` while for `Scipy` it is
            recommended to use `atol` and `rtol`. See their documentations for
            detailed information.
        """
        if self.Q.size == 0:
            if self.verbose:
                print('Generating forward matrix')
            self.generate_forward_matrix()

        if method == 'np_pinv':
            if self.verbose:
                print('Using numpy.pinv for inversion')
            self.IQ = np.linalg.pinv(self.Q, **method_kwargs)
        elif method == 'sp_pinv' or method == 'sp_pinv2':
            if self.verbose:
                print('Using scipy.linalg.pinv for inversion')
            self.IQ = slin.pinv(self.Q, **method_kwargs)
        # elif method == 'tf_pinv':
        #     if self.verbose:
        #         print('Using tf.linalg.pinv for inversion')
        #     self.IQ = tf.linalg.pinv(self.Q, rcond=rcond, **method_kwargs)
        else:
            raise ValueError(f'Method {method} not implemented')

        # print('Bz_data shape OLD:', Bz_array.shape)
        Bz_Data = np.reshape(self.Bz_array, self.N_sensors, order='C')
        # print('Bz_data shape:', Bz_Data.shape)
        # print('IQ shape:', IQ.shape)
        self.inv_multipole_moments = np.dot(self.IQ, Bz_Data)
        self.inv_multipole_moments.shape = (self.N_particles, self._N_cols)
        # print('mags:', mags.shape)

        # Forward field
        self.inv_Bz_array = np.matmul(self.Q,
                                      self.inv_multipole_moments.reshape(-1))
        self.inv_Bz_array.shape = (self.Ny_surf, -1)

        # Generate covariance matrix if sigma not none
        if isinstance(sigma_field_noise, float):
            self.covariance_matrix = (sigma_field_noise ** 2) * np.matmul(self.IQ, self.IQ.transpose())
            # Compute the std deviation in the mag moments solutions
            self.inv_moments_std = np.sqrt(np.diag(self.covariance_matrix))
            # Reshape into (N_particles, N_multipoles) matrix
            self.inv_moments_std.shape = (self.N_particles, -1)

    def save_multipole_moments(self,
                               save_name: str = 'TIME_STAMP',
                               basedir: Union[Path, str] = '.',
                               save_identifier: bool = False,
                               save_moments_std: bool = False) -> None:
        """Save the multipole values in `npz` files.

        Values need to be computed using the `compute_inversion` method.

        Parameters
        ----------
        save_name
            File name of the `npz` file
        basedir
            Base directory where results are saved. Will be created if it
            does not exist
        save_identifier
            Add a set identifier to the magnetic moments
        save_moments_std
            Add the standard deviation of the magnetic moments to the `npz`
            file in case the `sigma_field_noise` was specified in the inversion
        """
        BASEDIR = Path(basedir)
        BASEDIR.mkdir(exist_ok=True)

        if save_name == 'TIME_STAMP':
            fname = BASEDIR / f'InvMagQuad_{self.time_stamp}.npz'
        else:
            fname = BASEDIR / f'InvMagQuad_{save_name}.npz'

        data_dict = dict(inv_multipole_moments=self.inv_multipole_moments)
        if save_identifier and hasattr(self, 'identifier'):
            data_dict['identifier'] = self.identifier
        if save_moments_std:
            data_dict['moments_std'] = self.inv_moments_std

        np.savez(fname, **data_dict)


# -----------------------------------------------------------------------------
# DEPRECATED

def dipole_field(dip_r, dip_m, pos_r):
    """
    Calculate magnetic flux B generated by dipole
    with magnetic moment dip_m  (Am2)
    located in position dip_r    (m)
    at position  pos_r           (m)
    unit of result is T
    """
    r = [pos_r[k] - dip_r[k] for k in range(3)]
    rho2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
    rho = np.sqrt(rho2)
    sp = dip_m[0] * r[0] + dip_m[1] * r[1] + dip_m[2] * r[2]
    f = 3e-7 * sp / (rho2 * rho2 * rho)
    g = -1e-7 / (rho2 * rho)
    return([f * r[k] + g * dip_m[k] for k in range(3)])
