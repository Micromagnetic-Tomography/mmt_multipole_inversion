from enum import EnumCheck
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

# Suscept modules:
from . import susceptibility_modules as sus_mods

from typing import Optional
from typing import Literal  # Working with Python >3.8
from typing import Union    # Working with Python >3.8
from collections.abc import Callable
from . import plot_tools as pt

# For the mask using image:
import PIL
import scipy.interpolate as si

import logging
# Notice this will inherit the params of the root looger in __init__
LOGGER = logging.getLogger(__name__)

# Import pylops and linearoperator
import pylops
# pylinv = pylops.optimization.leastsquares.normal_equations_inversion
pylinv2 = pylops.optimization.leastsquares.regularized_inversion
from .susceptibility_modules.pylops.pylopsclass import GreensMatrix

import torch
import torchmin as tmin
from .susceptibility_modules.torchm.torch_functions import Bflux_residual_f
from .susceptibility_modules.scipym.spm_functions import Bflux_residual_f as spm_Bflux_residual_f
import sys

# -----------------------------------------------------------------------------

_SusOptions = Literal['spherical_harmonics_basis',
                      'maxwell_cartesian_polynomials',
                      'cartesian_spherical_harmonics',
                      'spherical_harmonics_basis_area',
                      'spherical_harmonics_basis_volume'
                      ]
_ExpOptions = Literal['dipole', 'quadrupole', 'octupole']
_MethodOptions = Literal['numba', 'cuda']
_InvMethodOps = Literal['np_pinv', 'sp_pinv', 'sp_pinv2', 'pylops', 'torchmin', 'minimize_bfgs']


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

        LOGGER.info('🧲 Initialized new MultipoleInversion simulation')

        # Set the module from which to find the Bz susceptibility functions,
        # which is specified using the sus_functions_module argument
        self.sus_mod = getattr(sus_mods, sus_functions_module)

        self._expansion_limit = 'dipole'  # set default value
        self.expansion_limit = expansion_limit  # update def value

        # Optional sequence to set the origin of scan positions
        # self.scan_origin = (0.0, 0.0)

        # TODO: Wrap all key dict analysis in function??
        # TODO: Add option to use dict instead of json
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
            if k in metadict.keys():
                # need this for keys with a default value like sensor dimensions:
                attr = v[0] if isinstance(v, tuple) else v
                setattr(self, attr, metadict[k])
            else:
                msg = (f'Parameter "{k}" not found in json file. ')
                # For dict keys in multInVDict that have a default value: inform use of def value
                if isinstance(v, tuple):
                    LOGGER.warning(msg + f'Setting class attribute {v[0]} to default {v[1]}')
                    setattr(self, v[0], v[1])
                else:
                    LOGGER.warning(msg + 'Set a value for it in the json file.')

        for k in metadict.keys():
            if k not in multInVDict.keys():
                LOGGER.info(f'Not using parameter {k} in json file')

        # Information/checking against sample-sensor distance sign
        if self.Hz < 0:
            LOGGER.warning('The calculations use a right-hand coordinate system, where z=0 '
                           'is expected as the sample surface. Check that the z-positions '
                           'of the grains are defined accordingly: below zero or the scan height.')

        self.generate_measurement_mesh()

        # We set the arrays from the NPZ array here, to check against the
        # dimensions of the generated measurement mesh
        expected_arrays = ['Bz_array', 'particle_positions']
        # self.Bz_array = np.empty(0)
        self.particle_positions = None

        # Any other array in the NPZ file will be loaded here
        if sample_arrays:
            data = np.load(sample_arrays)
            for key, value in data.items():
                setattr(self, key, value)

            for key in expected_arrays:
                if key not in data.keys():
                    LOGGER.info(f'*{key}* array required for calculations. Set manually.')

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


    @property
    def Bz_array(self):
        return self._Bz_array

    @Bz_array.setter
    def Bz_array(self, Bz_array: np.ndarray):
        """
        """
        self._Bz_array = Bz_array
        LOGGER.info(f'Bz data array size     : {self._Bz_array.shape[0]} x {self._Bz_array.shape[1]}')
        LOGGER.info('Bz data memory         : {:.4f} Mb'.format(self._Bz_array.nbytes / (1024 * 1024)))

        # Check against size of generated mesh
        if self._Bz_array.shape[1] != self.Nx_surf or self._Bz_array.shape[0] != self.Ny_surf:
            LOGGER.error('Bz array dimensions do not match with measurement mesh dimensions. Inversions will fail')

        # TODO: Problem here if Bz array was not specified::
        # Generate field mask array with True everywhere as default
        LOGGER.warning('Bz array set. Setting raw fieldMask array')
        self.fieldMask = np.ones_like(self._Bz_array).astype(bool)

    def generate_measurement_mesh(self):
        """Generate coordinates for the measurement mesh

        The number of grid points in each direction (`xy`) are calculated by
        rounding the lateral size by the grid step size, e.g. `round(Sx / Sdx)`
        """

        # TODO: Check that Sx/Sy correspond to the dimensions of the Bz array
        # Generate measurement mesh
        self.Sx_range = self.sensor_origin_x + np.arange(round(self.Sx / self.Sdx)) * self.Sdx
        self.Sy_range = self.sensor_origin_y + np.arange(round(self.Sy / self.Sdy)) * self.Sdy
        self.Nx_surf = len(self.Sx_range)
        self.Ny_surf = len(self.Sy_range)

        LOGGER.info('Scanning array sizes (row x col)')
        LOGGER.info(f'Computed Sx x Sy sizes : {len(self.Sy_range)} x {len(self.Sx_range)}')

        self.N_sensors = self.Nx_surf * self.Ny_surf

        # TODO: Check the memory usage by saving the scan positions matrix 
        self.scan_positions = np.ones((self.N_sensors, 3))
        X_pos, Y_pos = np.meshgrid(self.Sx_range, self.Sy_range)
        self.scan_positions[:, :2] = np.stack((X_pos, Y_pos), axis=2).reshape(-1, 2)
        self.scan_positions[:, 2] *= self.Hz
        LOGGER.info('Scan positions array memory: {:.4f} Mb'.format(self.scan_positions.nbytes / (1024 * 1024)))

    def generate_forward_matrix(self, optimization: _MethodOptions = 'numba'):
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
        LOGGER.info('Green matrix memory: {:.4f} Mb'.format(self.Q.nbytes / (1024 * 1024)))

        # print('pos array:', particle_positions.shape)
        t0 = time.time()

        mp_order = {'dipole': 1, 'quadrupole': 2, 'octupole': 3}

        if optimization == 'cuda':
            if len(self.sensor_dims) == 0:
                if HASCUDA is False:
                    raise RuntimeError('The cuda method is not available. Stopping calculation')

                # Verbose only if logger is NOTSET, DEBUG or INFO
                verb = 1 if LOGGER.level <= 20 else 0
                errorInt = sus_cudalib.SHB_populate_matrix(
                    self.particle_positions,
                    self.scan_positions,
                    self.Q,
                    self.N_particles, self.N_sensors,
                    mp_order[self.expansion_limit],
                    verb)
                if errorInt != 0:
                    errMsg = f'Cuda code exited with error type {errorInt}. See: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html'
                    LOGGER.error(errMsg)
                    raise RuntimeError(errMsg)
                    # LOGGER.exception(f'Cuda code exited with error type {errorInt}')
                    # raise RuntimeError(f'Cuda calculation exited with cuda error type {errorInt}')



        # For all the particles, whose positions are stored in the pos array
        # (N_particles x 3), compute the dipole (3 terms), quadrupole (5 terms)
        # or octupole (7 terms) contributions. Here we populate the Q array
        # using the numba-optimised susceptibility functions
        elif optimization == 'numba':
            if len(self.sensor_dims) == 0:
                self.sus_mod.dipole_Bz_sus(self.particle_positions, self.scan_positions,
                                           self.Q, self._N_cols)
                if self.expansion_limit in ['quadrupole', 'octupole']:
                    self.sus_mod.quadrupole_Bz_sus(self.particle_positions,
                                                   self.scan_positions,
                                                   self.Q, self._N_cols)
                if self.expansion_limit in ['octupole']:
                    self.sus_mod.octupole_Bz_sus(self.particle_positions,
                                                 self.scan_positions,
                                                 self.Q, self._N_cols)

            # AREA SENSOR
            elif len(self.sensor_dims) == 2:
                if self._expansion_limit == 'octupole':
                    self.Q = np.empty(0)
                    raise ValueError('Octupole expansion_limit for area sensors not implemented')
                self.sus_mod.multipole_Bz_sus(self.particle_positions, self.scan_positions,
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
                self.sus_mod.multipole_Bz_sus(self.particle_positions, self.scan_positions,
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
        LOGGER.info(f'Generation of Q matrix took: {t1 - t0:.4f} s')
        # print('Q shape:', Q.shape)


    def generate_field_mask(self, fieldMaskTool: Union[Callable[[np.ndarray], bool], np.ndarray, str, Path]):
        """Creates a mask array for the Bz field array

        Parameters
        ----------
        fieldMaskTool
            This method accepts different ways to create a filter where no data
            is specified.  FUNCTION: define a function that depends on
            `r=(x,y)` as argument, which returns `True` for sites with data and
            `False` where data is filtered. ARRAY: which has the same
            dimensions as the `Bz` scanning data array, and where True/False or
            1/0 is for sites with/without data. STR or PATH: to an image file
            in black and white, where black pixels are specified for sensor
            data that is filtered/removed.
        """

        # Function of 2 variables (scanning field plane): x,y
        if callable(fieldMaskTool):
            self.fieldMask.shape = (-1)
            for i, r in enumerate(self.scan_positions[:, :2]):
                self.fieldMask[i] = fieldMaskTool(r)
            self.fieldMask.shape = (self.Sy_range.shape[0], -1)
        # Numpy array:
        elif isinstance(fieldMaskTool, np.ndarray):
            # Check that shapes are the same; np also checks that it can copy correctly
            if not (fieldMaskTool.shape[0] == self.fieldMask.shape[0]
                    and fieldMaskTool.shape[1] == self.fieldMask.shape[1]):
                raise ValueError('Array fieldMaskTool does not match Bz array'
                                 f' with shape {self.fieldMask.shape[0]} x {self.fieldMask.shape[1]}')
            else:
                self.fieldMask[:] = fieldMaskTool
        elif isinstance(fieldMaskTool, (str, Path)):
            # if self.verbose:
            #     print('Using image')

            # Error opening file is handled by PIL.Image
            with PIL.Image.open(fieldMaskTool) as imFile:
                # Make a two color map by converting the image
                # im = imFile.convert(mode='P', palette=PIL.Image.Palette.ADAPTIVE, colors=2)
                # imFile.load()
                imFile = PIL.ImageChops.invert(imFile)  # Invert white/black to get 1 at white pixels, 0 at black
                imFile = imFile.convert(mode='P', palette=PIL.Image.Palette.ADAPTIVE, colors=2)
                # NOTE: PILlow image loading sets y axis in the opposite direction c/t Cartesian axes
                im_arr = np.asarray(imFile)[::-1, :]

            if not ((im_arr.shape[0] == self.fieldMask.shape[0]) and 
                    (im_arr.shape[1] == self.fieldMask.shape[1])):
                LOGGER.info('Interpolating image into mask')

                # NOTE: the xrange is discretized bythe number of columns
                im_xrange = np.linspace(self.Sx_range[0], self.Sx_range[-1], im_arr.shape[1])
                im_yrange = np.linspace(self.Sy_range[0], self.Sy_range[-1], im_arr.shape[0])
                # im_positions = np.ones((self.N_sensors, 2))
                # X_pos, Y_pos = np.meshgrid(im_xrange, im_yrange)
                # im_positions[:, :2] = np.stack((X_pos, Y_pos), axis=2).reshape(-1, 2)
                # im_positions[:, 2] *= self.Hz

                # NOTE: Easier: use rescaling from PILLOW
                # imRes = im.resize(size=(self.Sx_range.shape[0], self.Sy_range.shape[0]),
                #                   resample=PIL.Image.Resampling.NEAREST)
                # imRes = imRes.convert(mode='P', palette=PIL.Image.Palette.ADAPTIVE, colors=2)

                # More precise: use scipy interpolation in 2D; the method uses np.meshgrid with
                # indexing='ij' (matrix order), so the row coordinate increases first. To fit the image, we have
                # to transpose the image array
                interp = si.RegularGridInterpolator((im_xrange, im_yrange), im_arr.T, method='nearest') 
                idata = interp(self.scan_positions[:, [0, 1]])
                # Reshape to recover the mask with Cartesian axes coordinates
                self.fieldMask[:] = idata.reshape(self.Sy_range.shape[0], -1)

            else:
                LOGGER.info('Using the specified image with original resolution')
                self.fieldMask[:] = im_arr

        # Expect that the array only contain 0s and 1s
        self.fieldMask.astype(bool)


    def compute_inversion(self,
                          method: _InvMethodOps = 'sp_pinv',
                          apply_field_mask: bool = False,
                          sigma_field_noise: Optional[float] = None,
                          initial_moments: Optional[np.ndarray] = None,
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
                direct   -> direct inverse (quickest and most memory efficient)
        apply_field_mask
            Set `True` if a masking array is used for the magnetic field. The
            mask must be created using the `generate_field_mask` method, which
            sets the `self.fieldMask` array for the `Bz` field. Values labeled
            as `False` are ignored.
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
            Extra parameters passed to Numpy, Pylops, or Scipy functions. For Numpy, the
            tolerance can be set using `rcond` while for `Scipy` it is
            recommended to use `atol` and `rtol`. See their documentations for
            detailed information.
        """
        # WARNING: (05/08/2025) Iterative methods not working with masks
        if method == 'pylops':
            nT = 1e9
            LOGGER.info('Using the pylops library for an iterative inversion')
            scan_positions = np.ones((self.N_sensors, 3))
            X_pos, Y_pos = np.meshgrid(self.Sx_range, self.Sy_range)
            scan_positions[:, :2] = np.stack((X_pos, Y_pos),
                                             axis=2).reshape(-1, 2)
            scan_positions[:, 2] *= self.Hz
            Dlop = GreensMatrix(self.N_sensors, self._N_cols, self.N_particles,
                                self.particle_positions, self.expansion_limit,
                                scan_positions, self.verbose)

            # self.inv_multipole_moments, info = pylinv(
            #     Dlop, self.Bz_array.flatten(), pyl_regs, show=self.verbose,
            #     **method_kwargs)
            if not method_kwargs:
                method_kwargs = dict(Regs=None, x0=None)
            (self.inv_multipole_moments, self.info1, self.info2,
             self.info3, self.info4) = pylinv2(
                Dlop, self.Bz_array.flatten() * nT, **method_kwargs)

            # Back to Tesla units:
            self.inv_Bz_array = Dlop.dot(self.inv_multipole_moments.flatten()) / nT
            self.inv_Bz_array.shape = (self.Ny_surf, -1)
            # Scale the magnetic moments back to Ampere * meter^X units:
            µm = 1e-6
            self.inv_multipole_moments.shape = (self.N_particles, self._N_cols)
            self.inv_multipole_moments[:, :3] *= µm ** 3
            if self.expansion_limit in ['quadrupole', 'octupole']:
                self.inv_multipole_moments[:, 3:8] *= µm ** 5
            if self.expansion_limit in ['octupole']:
                self.inv_multipole_moments[:, 8:15] *= µm ** 7
            # if info != 0:
            #     print(f'Inversion failed, errorcode: {info}')

        elif method == 'minimize_bfgs':

            scan_positions = np.ones((self.N_sensors, 3))
            X_pos, Y_pos = np.meshgrid(self.Sx_range, self.Sy_range)
            scan_positions[:, :2] = np.stack((X_pos, Y_pos), axis=2).reshape(-1, 2)
            scan_positions[:, 2] *= self.Hz
            LOGGER.info('Using scipy minimize for an iterative inversion')

            # tBz = torch.from_numpy(self.Bz_array)
            # tParticlePositions = torch.from_numpy(self.particle_positions)
            # tScanPositions = torch.from_numpy(scan_positions)
            def minF(x):
                return spm_Bflux_residual_f(x, self.Bz_array.flatten(), self.N_sensors, self._N_cols,
                                            self.N_particles, self.particle_positions, self.expansion_limit,
                                            scan_positions, engine='numba')

            # Random init state:
            if initial_moments is None:
                rng = np.random.default_rng(42)
                x0 = (2 * rng.random(self.N_particles * self._N_cols) - 1.).reshape(-1, self._N_cols)
                x0[:, :3] /= np.linalg.norm(x0[:, :3], axis=1)
                x0[:, :3] *= 1e4
                if self.expansion_limit in ['quadrupole', 'octupole']:
                    x0[:, 3:8] /= np.linalg.norm(x0[:, 3:8], axis=1)
                    x0[:, 3:8] *= 1e-8
                if self.expansion_limit in ['octupole']:
                    x0[:, 8:15] /= np.linalg.norm(x0[:, 8:15], axis=1)
                    x0[:, 8:15] *= 1e-22
                x0.shape = (-1)
            else:
                x0 = initial_moments
            
            # Nelder-Mead working fine:
            # minResult = so.minimize(minF, x0, tol=1e-20, options=dict(maxiter=40, disp=True),
            #                         method='Nelder-Mead')

            # Use BFGS by default
            if not method_kwargs:
                method_kwargs = dict(method='BFGS', tol=1e-5, options=dict(gtol=1e-2, disp=True, xrtol=1e-2))
            minResult = so.minimize(minF, x0, **method_kwargs)
            # minResult = tmin.minimize(minF, x0, options=dict(gtol=1e-25, disp=2), method='cg', disp=2)

            self.inv_multipole_moments = np.array(minResult.x)
            self.inv_multipole_moments.shape = (self.N_particles, self._N_cols)

            self.inv_Bz_array, _ = spm_Bflux_residual_f(self.inv_multipole_moments.flatten(), self.Bz_array.flatten(), self.N_sensors,
                                                        self._N_cols, self.N_particles, self.particle_positions,
                                                        self.expansion_limit, scan_positions, engine='numba', full_output=True)

            # Back to Tesla units:
            self.inv_Bz_array *= 1e-9
            self.inv_Bz_array.shape = (self.Ny_surf, -1)

        elif method == 'torchmin':
            scan_positions = np.ones((self.N_sensors, 3))
            X_pos, Y_pos = np.meshgrid(self.Sx_range, self.Sy_range)
            scan_positions[:, :2] = np.stack((X_pos, Y_pos), axis=2).reshape(-1, 2)
            scan_positions[:, 2] *= self.Hz
            LOGGER.info('Using the pytorch minimize lib for an iterative inversion')

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            tBz = torch.from_numpy(self.Bz_array).to(device)
            tParticlePositions = torch.from_numpy(self.particle_positions).to(device)
            tScanPositions = torch.from_numpy(scan_positions).to(device)

            def minF(x):
                return Bflux_residual_f(x, tBz.flatten(), self.N_sensors, self._N_cols, self.N_particles,
                                        tParticlePositions, self.expansion_limit, tScanPositions)
            LOGGER.info(f'CUDA Bz: {tBz.is_cuda}')
            # sys.exit()

            # Random init state:
            if initial_moments is None:
                rng = np.random.default_rng(42)
                x0 = (2 * rng.random(self.N_particles * self._N_cols) - 1.).reshape(-1, self._N_cols)
                x0[:, :3] /= np.linalg.norm(x0[:, :3], axis=1)
                x0[:, :3] *= 1e4
                if self.expansion_limit in ['quadrupole', 'octupole']:
                    x0[:, 3:8] /= np.linalg.norm(x0[:, 3:8], axis=1)
                    x0[:, 3:8] *= 1e-8
                if self.expansion_limit in ['octupole']:
                    x0[:, 8:15] /= np.linalg.norm(x0[:, 8:15], axis=1)
                    x0[:, 8:15] *= 1e-22
                x0.shape = (-1)
            else:
                x0 = initial_moments

            x00 = torch.from_numpy(x0).to(device)

            # Uniform init state:
            # x0 = 1e-12 * np.ones(self.N_particles * self._N_cols)
            # if self._expansion_limit == 'quadrupole':
            #     x0[3:] = 1e-18

            # Default minimizer options:
            if not method_kwargs:
                method_kwargs = dict(options=dict(xtol=1e-2, gtol=1e-2, line_search='strong-wolfe', normp=2, disp=2),
                                     method='l-bfgs', disp=2)
            minResult = tmin.minimize(minF, x00, **method_kwargs)
            # minResult = tmin.minimize(minF, x0, options=dict(gtol=1e-25, disp=2), method='cg', disp=2)

            # DEBUG:
            #   print('MMOMENTS', self.inv_multipole_moments)

            # Note that we use the magnetic moments in scaled units and we obtain the field in nT
            inv_Bz_array, _ = Bflux_residual_f(minResult.x, tBz.flatten(), self.N_sensors, self._N_cols, self.N_particles,
                                               tParticlePositions, self.expansion_limit, tScanPositions, full_output=True)
            # Back to Tesla units:
            self.inv_Bz_array = inv_Bz_array.cpu().detach().numpy() * 1e-9

            # Scale the magnetic moments back to Ampere * meter^X units:
            µm = 1e-6
            self.inv_multipole_moments = minResult.x.cpu().detach().numpy()
            self.inv_multipole_moments.shape = (self.N_particles, self._N_cols)
            self.inv_multipole_moments[:, :3] *= µm**3
            if self.expansion_limit in ['quadrupole', 'octupole']:
                self.inv_multipole_moments[:, 3:8] *= µm**5
            if self.expansion_limit in ['octupole']:
                self.inv_multipole_moments[:, 8:15] *= µm**7

            # DEBUG:
            # print('INV BZ MAX')
            # print(self.inv_Bz_array.min())
            # print(self.inv_Bz_array.max())
            # print('BZ MAX')
            # print(self.Bz_array.min())
            # print(self.Bz_array.max())

            self.inv_Bz_array.shape = (self.Ny_surf, -1)
        else:
            if self.Q.size == 0:
                LOGGER.info('Generating forward matrix')
                self.generate_forward_matrix()
    
            #idx = np.arange(len(self.Q))
            #if mask is not None:
            #    assert len(mask) == len(self.Q), ('mask has incorrect length, '
            #                                      f'should be: {len(self.Q)}')
            #    idx = np.where(mask == 0)[0]
    
            # Reshape Bz (without copy!) to pass it to the C/cuda/numba libraries
            # NOTE: This reshape of Bz_array assumes it is using C order in memory (default in np)
    
    
            self._Bz_array.shape = (self.N_sensors,)  # Can also use -1
            if apply_field_mask:
                LOGGER.info('Using field mask from the self.fieldMask array. '
                            'Confirm that you are using the right mask by calling the generate_field_mask() method.')
                Qmatrix = self.Q[self.fieldMask.reshape(-1)]
                Bzdata = self._Bz_array[self.fieldMask.reshape(-1)]
            else:
                Qmatrix = self.Q
                Bzdata = self._Bz_array
    
            if method == 'direct':
                LOGGER.info('Using direct inversion')
                self.inv_multipole_moments, res, rnk, s = slin.lstsq(
                    self.Q, Bzdata, **method_kwargs)
                self.inv_multipole_moments.shape = (self.N_particles, self._N_cols)
                # Forward field
                self.inv_Bz_array = np.matmul(self.Q, self.inv_multipole_moments.reshape(-1))
                self.inv_Bz_array.shape = (self.Ny_surf, self.Nx_surf)
            else:
                if method == 'np_pinv':
                    LOGGER.info('Using numpy.pinv for inversion')
                    self.IQ = np.linalg.pinv(Qmatrix, **method_kwargs)
                elif method == 'sp_pinv' or method == 'sp_pinv2':
                    LOGGER.info('Using scipy.linalg.pinv for inversion')
                    self.IQ = slin.pinv(Qmatrix, **method_kwargs)
                # elif method == 'tf_pinv':
                #     if self.verbose:
                #         print('Using tf.linalg.pinv for inversion')
                #     self.IQ = tf.linalg.pinv(self.Q, rcond=rcond, **method_kwargs)
                else:
                    raise ValueError(f'Method {method} not implemented')
    
                LOGGER.info('Finished inversion')  # Useful to check calc timing
    
                self.inv_multipole_moments = np.dot(self.IQ, Bzdata)
                self.inv_multipole_moments.shape = (self.N_particles, self._N_cols)
    
                # Forward field
                self.inv_Bz_array = np.matmul(self.Q, self.inv_multipole_moments.reshape(-1))
                self.inv_Bz_array.shape = (self.Ny_surf, -1)
    
                # Generate covariance matrix if sigma not none
                if isinstance(sigma_field_noise, float):
                    self.covariance_matrix = (sigma_field_noise ** 2) * np.matmul(self.IQ, self.IQ.transpose())
                    # Compute the std deviation in the mag moments solutions
                    self.inv_moments_std = np.sqrt(np.diag(self.covariance_matrix))
                    # Reshape into (N_particles, N_multipoles) matrix
                    self.inv_moments_std.shape = (self.N_particles, -1)
    
            # Assuming that Sx/Sy ranges correspond to the computed sizes for the scanning array
            self._Bz_array.shape = (self.Sy_range.shape[0], -1)

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
>>>>>>> master
