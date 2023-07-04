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
                MultInvInst,
                plot_height=0,
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
    MultInvInst
        An instance of the `MultipoleInversion` class. This function uses the
        attributes `Bz_array` `Sx_range`, `Sy_range`, `Sdx`, `Sdy` and
        `particle_positions`.
    plot_height
        determines which field to plot, default to 0.
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
    mi = MultInvInst

    dms = dimension_scale
    dds = data_scale

    cf, sc, contours = None, None, None

    if not imshow_args:
        cf = ax.contourf(mi.Sx_range * dms, mi.Sy_range * dms,
                         mi.Bz_matrix[plot_height] * dds, **contourf_args)
    else:
        dx, dy = mi.Sdx * dms * 0.5, mi.Sdy * dms * 0.5
        cf = ax.imshow(mi.Bz_matrix[plot_height] * dds,
                       origin='lower',
                       extent=[mi.Sx_range.min() * dms - dx,
                               mi.Sx_range.max() * dms + dx,
                               mi.Sy_range.min() * dms - dy,
                               mi.Sy_range.max() * dms + dy],
                       **imshow_args)

    if contour_args:
        contours = ax.contour(mi.Sx_range * dms, mi.Sy_range * dms,
                              mi.Bz_matrix[plot_height] * dds,
                              **contour_args)

    if scatter_args:
        sc = ax.scatter(mi.particle_positions[:, 0] * dms,
                        mi.particle_positions[:, 1] * dms,
                        **scatter_args)

    return cf, sc, contours


def plot_inversion_Bz(ax,
                      MultInvInst,
                      plot_height=0,
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
    MultInvInst
        An instance of the `MultipoleInversion` class. This function uses the
        attributes `inv_Bz_array`, `Sx_range`, `Sy_range`, `Sdx`, `Sdy` and
        `particle_positions`.
    plot_height
        determines which field to plot, default to 0.
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
    mi = MultInvInst

    dms = dimension_scale
    dds = data_scale

    cf, c1, c2 = None, None, None

    # plt.imshow(computed_FF.reshape(100, 101))
    # plt.colorbar()
    if not imshow_args:
        cf = ax.contourf(mi.Sx_range * dms, mi.Sy_range * dms,
                         mi.inv_Bz_matrix[plot_height] * dds, **contourf_args)
    else:
        dx, dy = 0.5 * mi.Sdx * dms, 0.5 * mi.Sdy * dms
        cf = ax.imshow(mi.inv_Bz_matrix[plot_height] * dds,
                       origin='lower',
                       extent=[mi.Sx_range.min() * dms - dx,
                               mi.Sx_range.max() * dms + dx,
                               mi.Sy_range.min() * dms - dy,
                               mi.Sy_range.max() * dms + dy],
                       **imshow_args)

    c1 = ax.contour(mi.Sx_range * dms, mi.Sy_range * dms,
                    mi.inv_Bz_matrix[plot_height] * dds, **contour_args)

    if scatter_args:
        c2 = ax.scatter(mi.particle_positions[:, 0] * dms,
                        mi.particle_positions[:, 1] * dms,
                        **scatter_args)

    # plt.savefig(f'FORWARD_scanning_array_{ts}.pdf', bbox_inches='tight')

    return cf, c1, c2


def plot_difference_Bz(ax,
                       MultInvInst,
                       plot_height=0,
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
    MultInvInst
        An instance of the `MultipoleInversion` class. This function uses the
        attributes `Bz_array`, `inv_Bz_array`, `Sx_range`, `Sy_range`, `Sdx`,
        `Sdy` and `particle_positions`.
    plot_height
        determines which field to plot, default to 0.
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
    mi = MultInvInst

    dms = dimension_scale
    dds = data_scale

    cf, c1 = None, None

    # plt.imshow((computed_FF - Bz_Data).reshape(100, 101))
    if not imshow_args:
        cf = ax.contourf(mi.Sx_range * dms, mi.Sy_range * dms,
                         (mi.inv_Bz_matrix[plot_height]
                          - mi.Bz_matrix[plot_height]) * dds,
                         **contourf_args)
    else:
        dx, dy = 0.5 * mi.Sdx * dms, 0.5 * mi.Sdy * dms
        cf = ax.imshow((mi.inv_Bz_matrix[plot_height]
                        - mi.Bz_matrix[plot_height]) * dds,
                       origin='lower',
                       extent=[mi.Sx_range.min() * dms - dx,
                               mi.Sx_range.max() * dms + dx,
                               mi.Sy_range.min() * dms - dy,
                               mi.Sy_range.max() * dms + dy],
                       **imshow_args)

    if scatter_args:
        c1 = ax.scatter(mi.particle_positions[:, 0] * dms,
                        mi.particle_positions[:, 1] * dms,
                        **scatter_args)

    # plt.savefig(f'ERROR_scanning_array_{ts}.pdf', bbox_inches='tight')
    # plt.show()

    return cf, c1
