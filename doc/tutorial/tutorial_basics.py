# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial: Basics

# %% [markdown]
# This tutorial shows how to perform a multipole inversion from a magnetic scan surface into one or more magnetic sources that are represented as physical point sources. The `multipole_inversion` library contains two main classes to perform the numerical inversions:
#
# - `MultipoleInversion` from the `multipole_inversion.multipole_inversion` library
# - `MagneticSample` from the `multipole_inversion.magnetic_sample` library
#
# With the `MagneticSample` module it is possible to define the scan grid dimensions and the location of the magnetic sources. From this class we save this information in `json` and `npz` files that can be inputted into the `MultipoleInversion` class. Both of these classes have extensive docstrings that can be read in this notebook for more information on the input parameters/arguments.
#
# Additional tools include plotting functions defined in the class libraries or in the `multipole_inversion.plot_tools` module.

# %% [markdown]
# ## Import and definitions

# %%
# %matplotlib inline

# %%
import numpy as np
import matplotlib.pyplot as plt
# from palettable.cartocolors.diverging import Geyser_5
from mpl_toolkits.axes_grid1 import make_axes_locatable
import importlib as imp  # to reload libraries if implementing new features


# %%
# Load the libraries for the calculation of dipole fields
import mmt_multipole_inversion as minv
from mmt_multipole_inversion import MultipoleInversion
from mmt_multipole_inversion import MagneticSample


# %%
# Define a colorbar for the plots
def colorbar(mappable, ax=None, location='right', size='5%', pad=0.05,
             orientation='vertical', ticks_pos='right', **kwargs):
    """
    Note: append_axes() reduces the size of ax to make room for the colormap
    ticks_pos       :: if orientation is vertical -> 'right' or 'left'
                       if orientation is horizontal -> 'top' or 'bottom'
    """

    if not ax:
        ax = plt.gca()

    # fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    # cbar = fig.colorbar(mappable, cax=cax)
    cbar = plt.colorbar(mappable, cax=cax, orientation=orientation,
                        **kwargs)
    if orientation == 'vertical':
        cax.yaxis.set_ticks_position(ticks_pos)
    elif orientation == 'horizontal':
        cax.xaxis.set_ticks_position(ticks_pos)
    # plt.sca(ax)
    return cbar


# %% [markdown]
# An example of docstring for the MultipoleInversion class:

# %%
# MultipoleInversion?

# %% [markdown]
# ## Testing the dipole field function

# %% [markdown]
# Here we are testing the dipole field from the `dipole.py` library. We set the dipole as close to the origin as possible, and oriented in the $\hat{x}$ direction.

# %%
dip_r = np.array([[0., 0., -1e-10]])
dip_m = np.array([[1, 0., 0.]])

# %%
# we set the space grid in the -1,1 range in both x and y directions
x = np.linspace(-1, 1, 150)
X, Y = np.meshgrid(x, x)
positions = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])

# %%
# Bz = msp.dipole_Bz(dip_r, dip_m, positions)
B = minv.magnetic_sample.dipole_field(dip_r, dip_m, positions)

# %% [markdown]
# We can now plot the dipole field around the origin, isolines are plotted for the $B_z$ component

# %%
plt.contour(X, Y, B[:, 2].reshape(-1, len(x)),
            levels=[-1e-11, -1e-12, 1e-12, 1e-11],
            colors='k', linestyles='-')
# plt.contourf(X, Y, B[:, 1].reshape(-1, len(x)).T)

# Normalise arrows
U, V = B[:, 0], B[:, 1]
norm = np.sqrt(U ** 2 + V ** 2)
# print(norm)
U, V = U / norm, V / norm

p = plt.quiver(positions[:, 0], positions[:, 1], U, V, norm, scale=25,
               cmap='magma', width=.005, edgecolor='k', linewidth=.5)
plt.colorbar(p)

plt.scatter(dip_r[:, 0], dip_r[:, 1], c='C2', s=50)
plt.xlim(-0.085, 0.085)
plt.ylim(-0.085, 0.085)

# %% [markdown]
# The same using streamlines:

# %%
# Normalise arrows
U, V = B[:, 0], B[:, 1]

plt.streamplot(X, Y, B[:, 0].reshape(-1, len(x)), B[:, 1].reshape(-1, len(x)),
               density=2, linewidth=1, color='k')

plt.scatter(dip_r[:, 0], dip_r[:, 1], c='C2', s=50)
# plt.xlim(-0.085, 0.085)
# plt.ylim(-0.085, 0.085)

# %% [markdown]
# ## Inversion of a single dipole source

# %% [markdown]
# In this Section we test the inversion of the field flux of a single dipole measured at a surface located at $H_z$ above a $Lx\times Ly\times Lz$ rectangular sample region, which contains the sipole at its centre.

# %%
Hz = 2e-6  # Scan height in m
Sx = 20e-6  # Scan area x - dimension in m
Sy = 20.1e-6  # Scan area y - dimension in m
Sdx = 0.1e-6  # Scan x - step in m
Sdy = 0.1e-6  # Scan y - step in m
Lx = Sx * 0.9  # Sample x - dimension in m
Ly = Sy * 0.9  # Sample y - dimension in m
Lz = 5e-6  # Sample thickness in m

# Initialise the dipole class
sample = MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz)

# Generate two random particles in the sample region, which we are going to
# redefine (this might not be necessary, we need to add more methods to the class)
# sample.generate_particles(N_particles=1)

# Manually set the positions and magnetization of the two dipoles
Ms = 4.8e5
dipole_positions = np.array([[sample.Lx * 0.5, sample.Ly * 0.5, -sample.Lz * 0.5]])
magnetization = Ms * (1 * 1e-18) * np.array([[1., 0., 0.]])
volumes = np.array([1e-18])
sample.generate_particles_from_array(dipole_positions, magnetization, volumes)

print('Magnetization:', sample.dipole_moments)

# Generate the dipole field measured as the Bz field flux through the
# measurement surface
sample.generate_measurement_mesh()

# %% [markdown]
# We can visualise the field generated by the dipole at the measurement surface. The colormap refers to the $B_z$ flux, and we add a streamplot to observe the dipole field direction $(B_x, B_y)$ at the surface.

# %%
f, ax = plt.subplots()
cf, c1, c2 = sample.plot_sample(ax)
colorbar(cf)
ax.set_aspect('equal')
c2.set_color('C3')

# Streamplot: take the measurement surface range and generate a regular
# rectangular mesh grid. We take these mesh points to compute the field in them
x, y = sample.Sx_range, sample.Sy_range
X, Y = np.meshgrid(x, y)
positions = np.column_stack([X.ravel(), Y.ravel(), sample.Hz * np.ones_like(X.ravel())])
B = minv.magnetic_sample.dipole_field(sample.dipole_positions, sample.dipole_moments, positions)
# Generate random seed points from where streamlines emerge (density not
# necessary if random seeds are used)
seed_points_x = sample.Lx * np.random.random(150)
seed_points_y = sample.Ly * np.random.random(150)
ax.streamplot(X, Y,
              B[:, 0].reshape(-1, len(x)),
              B[:, 1].reshape(-1, len(x)),
              density=1.5, linewidth=0.5, color='k',
              start_points=np.column_stack((seed_points_x, seed_points_y)),
              # start_points=[[0, 0]],
               )
# plt.xlim(5e-6, 15e-6)
# plt.ylim(5e-6, 15e-6)

# %%
sample.save_data(filename='dipole_y-orientation')

# %% [markdown]
# Now we use the `inv_quadrupole.py` library to load the dipole field data and inverse the signal into the particle position, which gives us the dipole and quadrupole moments. The latter should be close to zero.
#
# In the plot we observe that the inverte dsignal reproduces the original field accurately.

# %%
qinv = MultipoleInversion('./MetaDict_dipole_y-orientation.json',
                          './MagneticSample_dipole_y-orientation.npz',
                          expansion_limit='quadrupole')
qinv.compute_inversion(method='sp_pinv2', atol=1e-25)

f, ax = plt.subplots()
cf, c1, c2 = minv.plot_tools.plot_inversion_Bz(ax, qinv)
colorbar(cf)
ax.set_aspect('equal')

# %% [markdown]
# We can also check the residual is small compared to the inverted field:

# %%
f, ax = plt.subplots()
cf, c1 = minv.plot_tools.plot_difference_Bz(ax, qinv)
colorbar(cf)
ax.set_aspect('equal')

# %% [markdown]
# The inverted magnetic moments (3 dipole moments and 5 quadrupole moments) shows us the original dipole moment in the $x$-direction. The other moments are significantly small with respecto to $m_x$

# %%
qinv.inv_multipole_moments

# %% [markdown]
# ## Quadrupole

# %% [markdown]
# Here we define a quadrupole by specifying two magnetic dipoles oriented in opposite directions and located close to the center of the sample. Before saving the data, we redefine the two dipoles as a single particle at the center of the sample. The purpose of this idea is to analyse the strength of a magnetic quadrupole when solving the inversion problem. Accordingly, the dipole moments should be  close to zero and one or more quadrupole moments should be stronger.

# %% [markdown]
# ### Quadrupole y-direction
#
# The first example is a quadrupole oriented in the $+\hat{y}$ and $-\hat{y}$ directions, located at 4.5 micrometers from the measurement surface of the sample.

# %%
Hz = 2e-6  # Scan height in m
Sx = 20e-6  # Scan area x - dimension in m
Sy = 20e-6  # Scan area y - dimension in m
Sdx = 0.1e-6  # Scan x - step in m
Sdy = 0.1e-6  # Scan y - step in m
Lx = Sx * 0.9  # Sample x - dimension in m
Ly = Sy * 0.9  # Sample y - dimension in m
Lz = 5e-6  # Sample thickness in m

# Initialise the dipole class
sample = MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz)

# Manually set the positions and magnetization of the two dipoles
Ms = 4.8e5
dipole_positions = np.array([[sample.Lx * 0.5 - 1e-6, sample.Ly * 0.5, -sample.Lz * 0.5],
                             [sample.Lx * 0.5 + 1e-6, sample.Ly * 0.5, -sample.Lz * 0.5]])
magnetization = Ms * (1 * 1e-18) * np.array([[0., 1., 0], [0., -1, 0]])
volumes = np.array([1e-18, 1e-18])
sample.generate_particles_from_array(dipole_positions, magnetization, volumes)

# Generate the dipole field measured as the Bz field flux through the
# measurement surface
sample.generate_measurement_mesh()

# %%
# DEBUG:
# pos_r = np.array([sample.Sx_range[len(sample.Sx_range) // 2],
#                   sample.Sy_range[len(sample.Sy_range) // 2],
#                   sample.Hz])
# r = pos_r - sample.dipole_positions
# print(r)

# %% [markdown]
# Here we show the dipole field at the measurement surface, generated from the two dipoles in the sample region

# %%
f, ax = plt.subplots()
cf, c1, c2 = sample.plot_sample(ax)
colorbar(cf)
ax.set_aspect('equal')


# %% [markdown]
# Now we redefine the `dipole_positions` in the `sample` instance in order to make a single quadrupole source rather than two dipoles:

# %%
# Hack the positions array making a single particle at the centre
# (ideal quadrupole)
sample.dipole_positions = np.array([[sample.Lx * 0.5, sample.Ly * 0.5, -sample.Lz * 0.5]])
# This magnetisation direction should not matter (?)
sample.magnetization = Ms * (1 * 1e-18) * np.array([[0., 1., 0]])

# Update the N of particles to update the internal dict
sample.N_particles = 1

# %%
sample.save_data(filename='quadrupole_y-orientation')

# %%
# !cat MetaDict_dipole_y-orientation.json

# %% [markdown]
# At this point we can load the data for the inversion of the measurement generated in the previous steps, in the inversion code/class. Notice we are going to use the `maxwell_cartesian_polynomials` as a basis for the multipole expansion in order to physically interepret the results. Strictly, this basis is not orthogonal so it is not the most robust basis for the expansion. Nevertheless, for a single magnetic source it should solve the problem:

# %%
qinv = MultipoleInversion('./MetaDict_quadrupole_y-orientation.json',
                          './MagneticSample_quadrupole_y-orientation.npz',
                          expansion_limit='quadrupole',
                          sus_functions_module='maxwell_cartesian_polynomials')

# %%
qinv.compute_inversion(method='sp_pinv2')

# %% [markdown]
# We can compute the inverted measurement grid to compare it with the original measurement grid. We also notice we have now a single particle at the centre of the sample:

# %%
f, ax = plt.subplots()
cf, c1, c2 = minv.plot_tools.plot_inversion_Bz(ax, qinv)
colorbar(cf)
ax.set_aspect('equal')

# %% [markdown]
# From the inversion we can now check the magnitude of the inverted multipole moments. In this case, the inverted magnetization array is
#
# $$
# \texttt{inv magnetization} = [m_x, m_y, m_z, Q_1, Q_2, \ldots, Q_5]
# $$
#
# where $m_i$ are the dipole moments and $Q_i$ are the quadrupole moments.
#
# In this example, we can see that $Q_2=Q_{xy}$ has the highest magnitude among the quadrupoles, and the magnitude is around $10^{-18}$ (check units). The dipole moments should be around $\approx 10^{-12}$ if we had only a dipolar field.

# %%
qinv.inv_multipole_moments

# %% [markdown]
# And finally the difference between the measured field $B_z$ and the field from the inversion. The residual has an octupole character:

# %%
f, ax = plt.subplots()
cf, c1 = minv.plot_tools.plot_difference_Bz(ax, qinv)
colorbar(cf)
ax.set_aspect('equal')

# %% [markdown]
# ### Quadrupole x-direction

# %% [markdown]
# We repeat the same calculations here, but setting the two dipoles in the $x$-direction

# %%
Hz = 2e-6  # Scan height in m
Sx = 20e-6  # Scan area x - dimension in m
Sy = 20e-6  # Scan area y - dimension in m
Sdx = 0.1e-6  # Scan x - step in m
Sdy = 0.1e-6  # Scan y - step in m
Lx = Sx * 0.9  # Sample x - dimension in m
Ly = Sy * 0.9  # Sample y - dimension in m
Lz = 5e-6  # Sample thickness in m

sample = MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz)
# Manually set positions
Ms = 4.8e5
dipole_positions = np.array([[sample.Lx * 0.5, sample.Ly * 0.5 - 1e-6, -sample.Lz * 0.5],
                             [sample.Lx * 0.5, sample.Ly * 0.5 + 1e-6, -sample.Lz * 0.5]])
magnetization = Ms * (1 * 1e-18) * np.array([[-1., 0., 0], [1., 0, 0]])
volumes = np.array([1e-18, 1e-18])
sample.generate_particles_from_array(dipole_positions, magnetization, volumes)

sample.generate_measurement_mesh()

# %%
f, ax = plt.subplots()
p, *_ = sample.plot_sample(ax)
colorbar(p)

ax.set_aspect('equal')

# %%
# Now hack the positions array making a single particle at the centre
# (ideal quadrupole)
sample.dipole_positions = np.array([[sample.Lx * 0.5, sample.Ly * 0.5, -sample.Lz * 0.5]])
# This magnetisation direction should not matter (?)
sample.magnetization = Ms * (1 * 1e-18) * np.array([[1., 0., 0]])

sample.N_particles = 1  # Need to modify the JSON file!

sample.save_data(filename='quadrupole_x-orientation')

# %%
qinv = minv.MultipoleInversion('./MetaDict_quadrupole_x-orientation.json',
                               './MagneticSample_quadrupole_x-orientation.npz',
                               sus_functions_module='maxwell_cartesian_polynomials')
qinv.compute_inversion(method='sp_pinv2')

# %% [markdown]
# We again obtain the highest quadrupole for the $Q_{xy}$ component

# %%
qinv.inv_multipole_moments

# %% [markdown]
# ### Quadrupole xy-direction

# %% [markdown]
# And the calculation for the dipoles in the $xy$ direction ($\phi = \pi/4$ in polar)

# %%
Hz = 2e-6  # Scan height in m
Sx = 20e-6  # Scan area x - dimension in m
Sy = 20e-6  # Scan area y - dimension in m
Sdx = 0.1e-6  # Scan x - step in m
Sdy = 0.1e-6  # Scan y - step in m
Lx = Sx * 0.9  # Sample x - dimension in m
Ly = Sy * 0.9  # Sample y - dimension in m
Lz = 5e-6  # Sample thickness in m

sample = MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz)
# Manually set positions
Ms = 4.8e5
dipole_positions = np.array([[sample.Lx * 0.5 - 1e-6, sample.Ly * 0.5 - 1e-6, -sample.Lz * 0.5],
                             [sample.Lx * 0.5 + 1e-6, sample.Ly * 0.5 + 1e-6, -sample.Lz * 0.5]])

n = np.sqrt(2)
magnetization = Ms * (1 * 1e-18) * np.array([[-1 / n, 1 / n, 0],
                                             [1 / n, -1 / n, 0]])
volumes = np.array([1e-18, 1e-18])
sample.generate_particles_from_array(dipole_positions, magnetization, volumes)

sample.generate_measurement_mesh()

# %%
f, ax = plt.subplots()
p, *_ = sample.plot_sample(ax)
colorbar(p)

ax.set_aspect('equal')

# %%
# Now hack the positions array making a single particle at the centre
# (ideal quadrupole)
sample.dipole_positions = np.array([[sample.Lx * 0.5, sample.Ly * 0.5, -sample.Lz * 0.5]])
# This magnetisation direction should not matter (?)
sample.magnetization = Ms * (1 * 1e-18) * np.array([[-1 / n, 1 / n, 0]])

sample.N_particles = 1

sample.save_data(filename='quadrupole_xy-orientation')

# %%
qinv = minv.MultipoleInversion('./MetaDict_quadrupole_xy-orientation.json',
                               './MagneticSample_quadrupole_xy-orientation.npz',
                               expansion_limit='quadrupole')
qinv.compute_inversion(method='sp_pinv2')

# %% [markdown]
# Now the highest moments are the $Q_{1}=Q_{11}=Q_{xx}$ and the $Q_{4}=Q_{22}=Q_{yy}$ components

# %%
qinv.inv_multipole_moments

# %%
f, ax = plt.subplots()
cf, c1, c2 = minv.plot_tools.plot_inversion_Bz(ax, qinv)
colorbar(cf)
ax.set_aspect('equal')

# %% [markdown]
# ## Octupole

# %% [markdown]
# In this case we generate an artificial octupole using four dipoles. Below it can be seen that the inversion fails to produce an optimal solution for the octupole. It is possible to obtain a better solution by defining the octupole using two quadrupoles with two point sources, rather than a single point source.

# %%
Hz = 2e-6  # Scan height in m
Sx = 20e-6  # Scan area x - dimension in m
Sy = 20e-6  # Scan area y - dimension in m
Sdx = 0.1e-6  # Scan x - step in m
Sdy = 0.1e-6  # Scan y - step in m
Lx = Sx * 0.9  # Sample x - dimension in m
Ly = Sy * 0.9  # Sample y - dimension in m
Lz = 5e-6  # Sample thickness in m

sample = MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz)
# Manually set positions
Ms = 4.8e5
dipole_positions = np.array([[sample.Lx * 0.5 + 1e-6, sample.Ly * 0.5 + 1e-6, -sample.Lz * 0.5],
                             [sample.Lx * 0.5 - 1e-6, sample.Ly * 0.5 + 1e-6, -sample.Lz * 0.5],
                             [sample.Lx * 0.5 - 1e-6, sample.Ly * 0.5 - 1e-6, -sample.Lz * 0.5],
                             [sample.Lx * 0.5 + 1e-6, sample.Ly * 0.5 - 1e-6, -sample.Lz * 0.5]])

n = np.sqrt(2)
magnetization = Ms * (1 * 1e-18) * np.array([[-1 / n, 1 / n, 0],
                                             [-1 / n, -1 / n, 0],
                                             [1 / n, -1 / n, 0],
                                             [1 / n, 1 / n, 0]])
volumes = np.array([1e-18, 1e-18, 1e-18, 1e-18])
sample.generate_particles_from_array(dipole_positions, magnetization, volumes)

sample.generate_measurement_mesh()

# %%
f, ax = plt.subplots()
p, *_ = sample.plot_sample(ax)
colorbar(p)

ax.set_aspect('equal')

# %%
# Now hack the positions array making a single particle at the centre
# (ideal octupole)
sample.dipole_positions = np.array([[sample.Lx * 0.5, sample.Ly * 0.5, -sample.Lz * 0.5]
                                    ])
# This magnetisation direction should not matter (?)
sample.magnetization = Ms * (1 * 1e-18) * np.array([[-1 / n, 1 / n, 0]])
sample.N_particles = 1 # Need to modify the JSON file!
sample.save_data(filename='octupole')

# %%
# We could try to use 2 quadrupoles as well:

# # Now hack the positions array making a single particle at the centre
# # (ideal octupole)
# sample.dipole_positions = np.array([[sample.Lx * 0.5 + 1e-6, sample.Ly * 0.5 + 1e-6, -sample.Lz * 0.5],
#                                     [sample.Lx * 0.5 - 1e-6, sample.Ly * 0.5 - 1e-6, -sample.Lz * 0.5]
#                                     ])
# # This magnetisation direction should not matter (?)
# sample.magnetization = Ms * (1 * 1e-18) * np.array([[-1 / n, 1 / n, 0],
#                                                     [1 / n, -1 / n, 0]
#                                                     ])
# sample.N_particles = 2
# sample.save_data(filename='octupole')

# %%
oinv = minv.MultipoleInversion('./MetaDict_octupole.json',
                               './MagneticSample_octupole.npz',
                               expansion_limit='octupole',
                               sus_functions_module='spherical_harmonics_basis'
                               )
oinv.compute_inversion(method='sp_pinv2', atol=1e-20)

# %%
f, ax = plt.subplots()
cf, c1, c2 = minv.plot_tools.plot_inversion_Bz(ax, oinv)
ax.set_aspect('equal')
colorbar(cf)

# %%
f, ax = plt.subplots()
cf, c1 = minv.plot_tools.plot_difference_Bz(ax, oinv)
ax.set_aspect('equal')
colorbar(cf)

# %%
qinv.inv_multipole_moments

# %% [markdown]
# ## Multiple particles sample

# %%
Hz = 5e-6  # Scan height in m
Sx = 200e-6  # Scan area x - dimension in m
Sy = 300e-6  # Scan area y - dimension in m
Sdx = 2e-6  # Scan x - step in m
Sdy = 3e-6  # Scan y - step in m
Lx = Sx * 0.9  # Sample x - dimension in m
Ly = Sy * 0.9  # Sample y - dimension in m
Lz = 30e-6  # Sample thickness in m

sample = MagneticSample(Hz, Sx, Sy, Sdx, Sdy, Lx, Ly, Lz)

sample.generate_random_particles(seed=42)
# print(sample.dipole_positions)

sample.generate_measurement_mesh()
sample.save_data(filename='seed42')

# %%
sample.Bz_array.shape

# %%
f, ax = plt.subplots(figsize=(6, 6))

cf, c1, c2 = sample.plot_sample(ax)
c2.set_sizes(c2.get_sizes() / 4)
colorbar(cf)
ax.set_aspect('equal')

# %%
plt.plot(sample.dipole_moments[:, 0], 'o', label=r'$m_x$')
plt.plot(sample.dipole_moments[:, 1], 'v', label=r'$m_y$')
plt.plot(sample.dipole_moments[:, 2], 's', label=r'$m_z$')

plt.legend()

# %% [markdown]
# ### Inversion

# %%
qinv = minv.MultipoleInversion('./MetaDict_seed42.json',
                               './MagneticSample_seed42.npz')

# %%
qinv.compute_inversion(rcond=1e-10, method='np_pinv')

# %%
qinv.inv_multipole_moments[:10]

# %%
f, ax = plt.subplots(figsize=(6, 6))
cf, c1, c2 = minv.plot_tools.plot_inversion_Bz(ax, qinv)
c2.set_sizes(c2.get_sizes() / 4)
colorbar(cf)
ax.set_aspect('equal')

# %%
plt.plot(qinv.inv_multipole_moments[:, 0], 'o', label=r'$m_x$')
plt.plot(qinv.inv_multipole_moments[:, 1], 'v', label=r'$m_y$')
plt.plot(qinv.inv_multipole_moments[:, 2], 's', label=r'$m_z$')

plt.legend()

# %%
f, ax = plt.subplots(figsize=(6, 6))
cf, c1 = minv.plot_tools.plot_difference_Bz(ax, qinv)
colorbar(cf)
ax.set_aspect('equal')

# %%
f, ax = plt.subplots(figsize=(6, 6))
ax.plot(qinv.Bz_array.flatten() - qinv.inv_Bz_array.flatten(),
        'o', label=r'$m_x$', ms=1)

# %%
print(f'B_z (max) = {np.max(qinv.Bz_array)}  | B_z (min) = {np.min(qinv.Bz_array)}')

# %%
# Residual root mean square
L = len(qinv.Bz_array.flatten())
Bzinv_minus_Bz = (qinv.Bz_array.flatten() - qinv.inv_Bz_array.flatten())
RRMS = np.sqrt(np.sum(Bzinv_minus_Bz ** 2)) / np.sqrt(L)
print(RRMS)

# %%
NRRMS = RRMS / (Bzinv_minus_Bz.max() - Bzinv_minus_Bz.min())
print(f'Normalised RRMS: {NRRMS:.4f} %')

# %%
f, ax = plt.subplots(figsize=(6, 6))

ax.plot(qinv.inv_multipole_moments[:, 0] - qinv.dipole_moments[:, 0], 'o-', label=r'$m_x$')
ax.plot(qinv.inv_multipole_moments[:, 1] - qinv.dipole_moments[:, 1], 'v-', label=r'$m_y$')
ax.plot(qinv.inv_multipole_moments[:, 2] - qinv.dipole_moments[:, 2], 's-', label=r'$m_z$')

ax.legend()
ax.set_ylabel(r'$m_{i}^{\mathrm{predicted}} - m_{i}^{\mathrm{data}}$')
