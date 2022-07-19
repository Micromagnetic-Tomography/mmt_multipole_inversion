# import matplotlib.pyplot as plt
# import matplotlib as mpl

# -----------------------------------------------------------------------------

def get_inversion_plot_objects(inv):
    """Helper function to return a tuple with 5 objects from a multipole
    inversion instance, to be used with the plot functions

    Parameters
    ----------
    inv
        Instance of MultipoleInversion

    Returns
    -------
    tuple
        (Sx_range, Sy_range, Sdx, Sdy, particle_positions) from `inv`

    """
    return (inv.Sx_range, inv.Sy_range, inv.Sdx, inv.Sdy,
            inv.particle_positions)

# -----------------------------------------------------------------------------


def plot_sample(ax,
                Bz_array,
                Sx_range, Sy_range, Sdx, Sdy, particle_positions,
                contourf_args={'levels': 50},
                contour_args={},
                scatter_args={'c': 'k', 's': 1},
                imshow_args=None,
                dimension_scale=1., data_scale=1.,
                ):
    """ Plot the magnetic scan grid signal stored in the `Bz_array` variable

    Parameters
    ----------
    ax
        Matplotlib axis object
    Bz_array
        2D matrix with the magnetic scan signal
    Sx_range, Sy_range, Sdx, Sdy, particle_positions
        Specifications of the scan grid and the particle locations
    contourf_args,
        Plots Bz using a colormap and filled contour levels. This options is
        a `dict` passed to the corresponding matplotlib function.
    contour_args,
        Plots the line profile of the contour levels. Can be deactivated by
        setting this as `None` instead of a `dict`
    scatter_args
        Plots the particle positions as data points. Can be deactivated by
        setting this as `None` instead of a `dict`
    imshow_args
        If specified, use `imshow` instead of `contourf` for the colourmap. In
        this case all the contourf args are ignored
    dimension_scale, data_scale
        Scaling for the spatial and field data

    Returns
    -------
    tuple
        (contourf/imshow, scatter, contours) plot objects

    """

    dms = dimension_scale
    dds = data_scale

    cf, sc, contours = None, None, None

    if not imshow_args:
        cf = ax.contourf(Sx_range * dms, Sy_range * dms, Bz_array * dds,
                         **contourf_args)
    else:
        dx, dy = Sdx * dms * 0.5, Sdy * dms * 0.5
        cf = ax.imshow(Bz_array * dds,
                       origin='lower',
                       extent=[Sx_range.min() * dms - dx,
                               Sx_range.max() * dms + dx,
                               Sy_range.min() * dms - dy,
                               Sy_range.max() * dms + dy],
                       **imshow_args)

    if contour_args:
        contours = ax.contour(Sx_range * dms, Sy_range * dms,
                              Bz_array * dds,
                              **contour_args)

    if scatter_args:
        sc = ax.scatter(particle_positions[:, 0] * dms,
                        particle_positions[:, 1] * dms,
                        **scatter_args)

    return cf, sc, contours


def plot_inversion_Bz(ax,
                      inv_Bz_array,
                      Sx_range, Sy_range, Sdx, Sdy, particle_positions,
                      contourf_args={'cmap': 'RdYlBu', 'levels': 10},
                      contour_args={'colors': 'k', 'linewidths': .2, 'levels': 10},
                      scatter_args={'c': 'k'},
                      imshow_args=None,
                      dimension_scale=1., data_scale=1.,
                      ):
    """Plot the inverted field Bz and the positions of the particles

    Parameters
    ----------
    ax
        Matplotlib axis object
    inv_Bz_array
        2D matrix with the inverted magnetic scan signal
    Sx_range, Sy_range, Sdx, Sdy, particle_positions
        Specifications of the scan grid and the particle locations
    contourf_args,
        Plots Bz using a colormap and filled contour levels. This options is
        a `dict` passed to the corresponding matplotlib function.
    contour_args,
        Plots the line profile of the contour levels. Can be deactivated by
        setting this as `None` instead of a `dict`
    scatter_args
        Plots the particle positions as data points. Can be deactivated by
        setting this as `None` instead of a `dict`
    imshow_args
        If specified, use `imshow` instead of `contourf` for the colourmap. In
        this case all the contourf args are ignored
    dimension_scale, data_scale
        Scaling for the spatial and field data

    Returns
    -------
    tuple
        (contourf/imshow, scatter, contours) plot objects

    """

    dms = dimension_scale
    dds = data_scale

    cf, c1, c2 = None, None, None

    # plt.imshow(computed_FF.reshape(100, 101))
    # plt.colorbar()
    if not imshow_args:
        cf = ax.contourf(Sx_range * dms, Sy_range * dms,
                         inv_Bz_array * dds, **contourf_args)
    else:
        dx, dy = 0.5 * Sdx * dms, 0.5 * Sdy * dms
        cf = ax.imshow(inv_Bz_array * dds,
                       origin='lower',
                       extent=[Sx_range.min() * dms - dx,
                               Sx_range.max() * dms + dx,
                               Sy_range.min() * dms - dy,
                               Sy_range.max() * dms + dy],
                       **imshow_args)

    c1 = ax.contour(Sx_range * dms, Sy_range * dms,
                    inv_Bz_array * dds, **contour_args)

    if scatter_args:
        c2 = ax.scatter(particle_positions[:, 0] * dms,
                        particle_positions[:, 1] * dms,
                        **scatter_args)

    # plt.savefig(f'FORWARD_scanning_array_{ts}.pdf', bbox_inches='tight')

    return cf, c1, c2


def plot_difference_Bz(ax,
                       Bz_array, inv_Bz_array,
                       Sx_range, Sy_range, Sdx, Sdy, particle_positions,
                       contourf_args={'cmap': 'RdYlBu', 'levels': 50},
                       scatter_args={'c': 'k', 's': 1},
                       imshow_args=None,
                       dimension_scale=1., data_scale=1.,
                       ):
    """Plot the residual field and the positions of the particles

    Parameters
    ----------
    ax
        Matplotlib axis object
    Bz_array, inv_Bz_array
        2D matrices with the forward and the inverted magnetic scan signals
    Sx_range, Sy_range, Sdx, Sdy, particle_positions
        Specifications of the scan grid and the particle locations
    contourf_args,
        Plots Bz using a colormap and filled contour levels. This options is\
        a `dict` passed to the corresponding matplotlib function.
    contour_args,
        Plots the line profile of the contour levels. Can be deactivated by\
        setting this as `None` instead of a `dict`
    scatter_args
        Plots the particle positions as data points. Can be deactivated by\
        setting this as `None` instead of a `dict`
    imshow_args
        If specified, use `imshow` instead of `contourf` for the colourmap. In\
        this case all the contourf args are ignored
    dimension_scale, data_scale
        Scaling for the spatial and field data

    Returns
    -------
    tuple
        (contourf/imshow, scatter) plot objects

    """

    dms = dimension_scale
    dds = data_scale

    cf, c1 = None, None

    # plt.imshow((computed_FF - Bz_Data).reshape(100, 101))
    if not imshow_args:
        cf = ax.contourf(Sx_range * dms, Sy_range * dms,
                         (inv_Bz_array - Bz_array) * dds,
                         **contourf_args)
    else:
        dx, dy = 0.5 * Sdx * dms, 0.5 * Sdy * dms
        cf = ax.imshow((inv_Bz_array - Bz_array) * dds,
                       origin='lower',
                       extent=[Sx_range.min() * dms - dx,
                               Sx_range.max() * dms + dx,
                               Sy_range.min() * dms - dy,
                               Sy_range.max() * dms + dy],
                       **imshow_args)

    if scatter_args:
        c1 = ax.scatter(particle_positions[:, 0] * dms,
                        particle_positions[:, 1] * dms,
                        **scatter_args)

    # plt.savefig(f'ERROR_scanning_array_{ts}.pdf', bbox_inches='tight')
    # plt.show()

    return cf, c1
