import numpy as np
# import math
import matplotlib.pyplot as plt
import time
import datetime
import json
from pathlib import Path
# Load Bz field functions:
from . import magnetic_sample_modules as ms_mods

# Set a specific generator for random numbers
# np.random.seed(42)


# TODO: can be defined as static method
def dipole_field(dip_r, dip_m, pos_r):
    """
    Compute the dipole field at the `pos_r` position(s), from a group of `N`
    particles located at the `dip_r` dipole_positions, and which have magnetic
    dipole moments given in the `dip_m` array. For these arrays, `N` > 1.

    Parameters
    ----------
    dip_r
        N x 3 array with dipole dipole_positions (m)
    dip_m
        N x 3 array with dipole moments (Am^2)
    pos_r
        1 x 3 array (m)

    Returns

    dipole_field as N x 3 array, in Tesla units
    """

    r = pos_r - dip_r
    x, y, z = r[:, 0], r[:, 1], r[:, 2]

    rho2 = np.sum(r ** 2, axis=1)
    rho = np.sqrt(rho2)

    sp = dip_m[:, 0] * x + dip_m[:, 1] * y + dip_m[:, 2] * z
    f = 3e-7 * sp / (rho2 * rho2 * rho)
    g = -1e-7 / (rho2 * rho)

    # return([f * r[k] + g * dip_m[k] for k in range(3)])
    return np.einsum('i,ij->ij', f, r) + np.einsum('i,ij->ij', g, dip_m)


def ran_sphere(n, ran_generator):
    """Random number generator

    Parameters
    ----------
    ran_generator
        A Numpy `RandomState` object to generate random numbers
    """
    theta = ran_generator.uniform(0.0, 2 * np.pi, n)
    z = ran_generator.uniform(-1.0, 1.0, n)

    zp = z
    xp = np.sqrt(1 - z * z) * np.cos(theta)
    yp = np.sqrt(1 - z * z) * np.sin(theta)

    res = np.column_stack((xp, yp, zp))

    return res


class MagneticSample(object):
    """
    Class for the specification of a scan grid detecting the out plane flux of
    the magnetic field generated by an arbitrary number of magnetic sources.
    The magnetic sources can be dipoles, quadrupoles or octupoles. The scan
    grid is defined in the XY plane, at a given height above the magnetic
    sample containing the point sources.

    This class calculates the magnetic field from the point sources and adds
    them to generate the total flux at every area of the scan grid mesh.

    ::

                             Sdx
                          ___/___
                         /      /            Scan Grid
                   _     ______________________________
            Sdy_ /     /       /      /       /       /
                /_    /_______/______/_______/_______/__      _
                     /       /      /       /       /   /|     |
                    / ______/______/_______/_______/   / |     |_ Lz
                   /       /      /       /       /   /  |     |
               _  /_______/______/_______/_______/   /   |     |
          Hx _|     /                O              /  O |    _|
              |_   /______________________________ /    /
                  |                               |    /
                  |            O                  |   /
                  |   O                           |  /
                  |    dipole               O     | /
                  |_______________________________|/  Sample
    """

    # TODO: pass cfg file or json file with specifications
    # Can define a class method (@classmethod) to overload the __init__ func
    def __init__(self, Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz,
                 sensor_origin=(0.0, 0.0),
                 bz_field_module='spherical_harmonics_basis'
                 ):
        """
        Constructor for the sample generator class. The arguments are necessary
        to specify the scan grid and sample dimensions.

        Parameters
        ----------
        Hz
            Scan height in m
        Sx, Sy
            Scan area x and y - dimension in m
        Sdx, Sdy
            Scan x-step and y-step in m
        Lx, Ly
            Sample plane dimensions in m
        Lz
            Sample thickness in m
        sensor_origin
            2-sequence to specify the origin of the scan grid by the lower
            left sensor position
        bz_field_module
            Specify the basis to generate the Bz multipole fields

        Notes
        -----
        Particle positions and magnetic moments must be generated using the
        `generate_random_particles` or `generate_particles_from_array` methods
        """

        self.Hz = Hz
        self.Sx = Sx
        self.Sy = Sy
        self.Sdx = Sdx
        self.Sdy = Sdy
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        # Optional sequence to set the origin of scan-sensor positions
        self.sensor_origin_x = sensor_origin[0]
        self.sensor_origin_y = sensor_origin[1]

        # TODO: time stamps replaced by seed number
        ts = time.time()
        self.time_stamp = datetime.datetime.fromtimestamp(ts).strftime(
                '%Y%m%d-%H%M%S')

        # Set dictionary entries with values as corresponding class attrs
        # A metadict can be generated using the self.get_metadict() method
        self._metadict = {}
        self._metadict["Scan height Hz"] = 'Hz'
        self._metadict["Scan area x-dimension Sx"] = 'Sx'
        self._metadict["Scan area y-dimension Sy"] = 'Sy'
        self._metadict["Scan x-step Sdx"] = 'Sdx'
        self._metadict["Scan y-step Sdy"] = 'Sdy'
        self._metadict["Time stamp"] = 'time_stamp'
        self._metadict["Sensor origin x"] = 'sensor_origin_x'
        self._metadict["Sensor origin y"] = 'sensor_origin_y'
        # self._metadict["Scan origin x"] = 'scan_origin[0]'
        # self._metadict["Scan origin y"] = self.scan_origin[1]
        self._metadict["Number of particles"] = 'N_particles'

        # TEMPORARY: set higher order moments as zero
        self.quadrupole_moments = None
        self.octupole_moments = None

        # Load the Bz field functions according to the chosen basis
        self.bz_field_mod = getattr(ms_mods, bz_field_module)

    def get_metadict(self):
        """
        Generate a dictionary with the keys defined in the `_metadict`
        variable. Values are obtained by calling the class instance attributes
        defined in the `_metadict` values.  Because scan origin is a sequence,
        it is split in x and y components. This method is used to export sample
        properties as a `json` file.
        """
        metadict = {}
        for k in self._metadict:
            if k == 'Scan origin':
                metadict[k + ' x'] = getattr(self, self._metadict[k])[0]
                metadict[k + ' y'] = getattr(self, self._metadict[k])[1]
            else:
                metadict[k] = getattr(self, self._metadict[k])
        return metadict

    def generate_random_particles(self, N_particles=100, Ms=4.8e5, seed=42,
                                  rmin=[0.1, 0.1, 0.1],
                                  rmax=[0.9, 0.9, 0.9]):
        """
        Generate a sample of dipole particles randomly distributed across the
        sample region. The dipole moments of the particles are randomly
        generated based on the saturation magnetization value Ms

        Parameters
        ----------
        N_particles
            Number of particles
        Ms
            Saturation magnetisation
        seed
            random number generator seed
        rmin, rmax
            minimum and maximum scale factors for the limits of the locations
            of the particles. The factors scale the sample dimensions in every
            dimension. For example, particles spread over the sample but close
            to the surface can be modelled by

                rmin = [0.1, 0.1, 0.7]
                rmax = [0.9, 0.9, 0.9]

            This means particle positions in the x-direction will vary between
            0.1 and 0.9 of Lx, and so on.
        """

        rstate = np.random.RandomState(seed)

        self.Ms = Ms
        self.N_particles = N_particles

        rnd = rstate.rand(self.N_particles, 3)
        self.dipole_positions = np.zeros_like(rnd)
        self.dipole_positions[:, 0] = self.Lx * (rmin[0] + rmax[0] * rnd[:, 0])
        self.dipole_positions[:, 1] = self.Ly * (rmin[1] + rmax[1] * rnd[:, 1])
        self.dipole_positions[:, 2] = -self.Lz * (rmin[2] + rmax[2] * rnd[:, 2])

        # Generate random particle volumes ------------------------------------
        # Not necessary:
        rnd = rstate.rand(self.N_particles, 3)

        # mean=3, std_dev=0.5:
        rnd = rstate.normal(loc=3, scale=0.5, size=(self.N_particles, 3))

        # random volume in m^3:
        # (multiply rnd columns by multiplying row elements using np.prod)
        self.volumes = np.abs(np.prod(rnd, axis=1)) * 1e-18  # micro-meter^3 ?

        # Uniformly magnetized with magnetization Ms --------------------------
        # Random magnetization directions
        mdir = ran_sphere(self.N_particles, rstate)
        # mag = np.column_stack((Ms * volumes * mdir[:, 0],
        #                        Ms * volumes * mdir[:, 1],
        #                        Ms * volumes * mdir[:, 2],
        #                        ))
        # Same but shorter: multiply 1D volumes array to every column of mdir
        self.dipole_moments = self.Ms * np.einsum('i,ij->ij',
                                                  self.volumes, mdir)

    def generate_particles_from_array(self,
                                      positions,
                                      dipole_moments,
                                      volumes,
                                      quadrupole_moments=None,
                                      octupole_moments=None,
                                      identifier=None,
                                      ):
        """
        Generate particles in the sample from arrays specified manually

        Parameters
        ----------
        positions
            N x 3 array (m units)
        dipole_moments
            N x 3 array (A m^2 unitS)
        volumes
            N x 3 array (m^3 units)
        quadrupole_moments
            N x 5 array with quadrupole moments
        octupole_moments
            N x 7 array with octupole moments
        identifier
            N array with integers representing particles

        Notes
        -----
        The multipole options need to be redefined (it depends on the basis we
        are expressing the multipole expansion)
        """

        self.dipole_positions = np.array(positions)
        self.dipole_moments = np.array(dipole_moments)
        if isinstance(quadrupole_moments, np.ndarray):
            self.quadrupole_moments = quadrupole_moments
        if isinstance(octupole_moments, np.ndarray):
            self.octupole_moments = octupole_moments
        self.volumes = volumes

        self.N_particles = len(positions)
        # identifying particles to compare results
        if isinstance(identifier, np.ndarray):
            self.identifier = identifier
        else:
            self.identifier = np.arange(1, len(positions) + 1)

    def generate_measurement_mesh(self):
        """
        Generate the magnetic flux array at the scan surface, i.e. calculate
        the total Bz contribution from the particles at every grid point of
        the scan surface

        """

        # Generate measurement mesh (maybe replace 0.0 by a shifted origin)
        self.Sx_range = self.sensor_origin_x + np.arange(round(self.Sx / self.Sdx)) * self.Sdx
        self.Sy_range = self.sensor_origin_y + np.arange(round(self.Sy / self.Sdy)) * self.Sdy
        Bz_grid = np.zeros((len(self.Sy_range), len(self.Sx_range)))

        pos = np.ones((Bz_grid.shape[0] * Bz_grid.shape[1], 3))
        X_pos, Y_pos = np.meshgrid(self.Sx_range, self.Sy_range)
        pos[:, :2] = np.stack((X_pos, Y_pos), axis=2).reshape(-1, 2)
        pos[:, 2] *= self.Hz
        pos.shape = (len(self.Sy_range), len(self.Sx_range), 3)

        self.bz_field_mod.dipole_Bz(self.dipole_positions, self.dipole_moments,
                                    pos, Bz_grid, self.Sx_range, self.Sy_range)

        if isinstance(self.quadrupole_moments, np.ndarray):
            self.bz_field_mod.quadrupole_Bz(self.dipole_positions,
                                            self.quadrupole_moments,
                                            pos, Bz_grid,
                                            self.Sx_range, self.Sy_range)

        if isinstance(self.octupole_moments, np.ndarray):
            self.bz_field_mod.octupole_Bz(self.dipole_positions,
                                          self.octupole_moments,
                                          pos, Bz_grid,
                                          self.Sx_range, self.Sy_range)

        self.Bz_array = Bz_grid

    def generate_noised_Bz_array(self, std_dev, seed=4242):
        """
        Add uncorrelated noise to the magnetic flux (Bz array). The new
        array is stored in self.Bz_array_noised
        Update the seed if necessar.
        For the seed a random number generator can be passed instead of an int.
        """
        if type(seed) == np.random.mtrand.RandomState:
            rstate = seed
        else:
            rstate = np.random.RandomState(seed)
        noise = rstate.normal(loc=0.0, scale=std_dev, size=self.Bz_array.shape)
        self.Bz_array_noised = np.copy(self.Bz_array) + noise

    def save_data(self, filename='TIME_STAMP', basedir='', noised_array=False):
        """
        Save the system properties as a `json` file and relevant arrays in a
        `npz` file: Bz_array, particle_positions, magnetization, volumes,
        and identifiers.

        Parameters
        ----------
        filename
            Name appended to the dictionary and arrays base name
        basedir
            An optional directory to which data is going to be stored
        noised_array
            Save the noised Bz_array instead of the original array
        """

        BASEDIR = Path(basedir)

        if filename == 'TIME_STAMP':
            save_fname = BASEDIR / f'MagneticSample_{self.time_stamp}'
            stp = self.time_stamp + '.json'
            json_fname = str(BASEDIR / "MetaDict_") + stp
        else:
            save_fname = BASEDIR / f'MagneticSample_{filename}'
            json_fname = BASEDIR / f'MetaDict_{filename}.json'

        if noised_array:
            Bz_data = self.Bz_array_noised
        else:
            Bz_data = self.Bz_array

        # Save uncompressed file
        np.savez(save_fname,
                 Bz_array=Bz_data,
                 particle_positions=self.dipole_positions,
                 dipole_moments=self.dipole_moments,
                 volumes=self.volumes,
                 identifier=self.identifier)

        # # Will leave this but if we only use Python this should be deprecated:
        # np.savetxt("Bz_array" + st, self.Bz_array, delimiter=", ", fmt='%.8e')
        # np.savetxt("Pos_array" + st, np.array(self.dipole_positions),
        #            delimiter=",", fmt='%.8e')
        # np.savetxt("Mag_array" + st, np.array(self.dipole_moments),
        #            delimiter=",", fmt='%.8e')
        # np.savetxt("Vol_array" + st, np.array(self.volumes),
        #            delimiter=",", fmt='%.8e')

        metadict = self.get_metadict()  # Generate dic with the sample props
        with open(json_fname, 'w') as f:
            json.dump(metadict, f)

    def plot_sample(self, ax,
                    contours=30,
                    contourlines=15,
                    contourf_args=dict(cmap='RdYlBu'),
                    contour_args=dict(colors='k', linewidths=0.2),
                    scatter_args=dict(c='k'),
                    dimension_scale=1., data_scale=1.,
                    noised_array=False,
                    imshow_args=None,
                    ):
        """
        (WILL BE REMOVED IN FUTURE)
        Plot the scan surface and the particles beneath using their xy position

        Optional:

            If imshow_args is specified, this functions uses imshow instead
            of contourf to plot the colored background with Bz_array. In this
            case, all the contourf args are ignored

        Returns     :: cf, c1, c2
                       where cf is the contour plot object showing Bz,
                       c1 its contour lines and
                       c2 the scatter plot with the particle positions
        """

        if noised_array:
            Bz_data = self.Bz_array_noised
        else:
            Bz_data = self.Bz_array

        dms = dimension_scale
        dds = data_scale

        Sx, Sy = self.Sx_range * dms, self.Sy_range * dms

        if not imshow_args:
            cf = ax.contourf(Sx, Sy, Bz_data * dds,
                             contours,
                             # cmap=Geyser_5.mpl_colormap
                             **contourf_args)
        else:
            dx, dy = 0.5 * self.Sdx * dms, 0.5 * self.Sdy * 0.5
            cf = ax.imshow(Bz_data * dds,
                           extent=[Sx.min() - dx, Sx.max() + dx,
                                   Sy.min() - dy, Sy.max() + dy],
                           origin='lower',
                           **imshow_args)

        # Use the contours of the original unperturbed array
        c1 = ax.contour(self.Sx_range * dms, self.Sy_range * dms,
                        self.Bz_array * dds,
                        contourlines,
                        **contour_args)
        c2 = ax.scatter(self.dipole_positions[:, 0] * dms,
                        self.dipole_positions[:, 1] * dms,
                        **scatter_args)
        return cf, c1, c2
