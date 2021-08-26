# import matplotlib.pyplot as plt
# import matplotlib as mpl


# -----------------------------------------------------------------------------


def plot_sample(ax,
                Bz_array, Sx_range, Sy_range, Sdx, Sdy, particle_positions,
                contourf_args={'levels': 50},
                contour_args={},
                scatter_args={'c': 'k', 's': 1},
                imshow_args=None,
                dimension_scale=1., data_scale=1.,
                ):
    """
    """

    dms = dimension_scale
    dds = data_scale

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
        ax.contour(Sx_range * dms, Sy_range * dms,
                   Bz_array * dds,
                   **contour_args)

    sc = ax.scatter(particle_positions[:, 0] * dms,
                    particle_positions[:, 1] * dms,
                    **scatter_args)
    return cf, sc


def plot_inversion_Bz(ax,
                      inv_Bz_array, Sx_range, Sy_range,
                      Sdx, Sdy, particle_positions,
                      dimension_scale=1., data_scale=1.,
                      #
                      imshow_args=None,
                      contourf_args={'cmap': 'RdYlBu', 'levels': 10},
                      #
                      contour_args={'colors': 'k', 'linewidths': .2, 'levels': 10},
                      scatter_args={'c': 'k'}
                      ):
    """
    Given a matplotlib axis, plot the inverted field Bz on it, and the
    positions of the particles

    Optional:

        If imshow_args is specified, this functions uses imshow instead
        of contourf to plot the colored background with Bz_array. In this
        case, all the contourf args are ignored
    """

    dms = dimension_scale
    dds = data_scale

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
    c2 = ax.scatter(particle_positions[:, 0] * dms,
                    particle_positions[:, 1] * dms,
                    **scatter_args)
    # plt.savefig(f'FORWARD_scanning_array_{ts}.pdf', bbox_inches='tight')

    return cf, c1, c2


def plot_difference_Bz(ax,
                       Bz_array, inv_Bz_array,
                       Sx_range, Sy_range, Sdx, Sdy, particle_positions,
                       dimension_scale=1., data_scale=1.,
                       contourf_args={'cmap': 'RdYlBu', 'levels': 50},
                       imshow_args=None,
                       scatter_args={'c': 'k', 's': 1}
                       ):

    dms = dimension_scale
    dds = data_scale

    # plt.imshow((computed_FF - Bz_Data).reshape(100, 101))
    if not imshow_args:
        cf = ax.contourf(Sx_range * dms, Sy_range * dms,
                         (inv_Bz_array - Bz_array) * dds,
                         contours,
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

    c1 = ax.scatter(particle_positions[:, 0] * dms,
                    particle_positions[:, 1] * dms,
                    **scatter_args)
    # plt.savefig(f'ERROR_scanning_array_{ts}.pdf', bbox_inches='tight')
    # plt.show()

    return cf, c1
