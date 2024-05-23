import numpy as np
import torch


def dipole_Bz_sus(xin, N_sensors, N_particles, dip_r, pos_r, yout, n_col_stride):
    for i in range(N_sensors):
        dr = pos_r[i] - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        z2 = z ** 2

        r2 = torch.sum(dr ** 2, dim=1)
        r = np.sqrt(r2)
        f = 1e-7 / (r2 * r2 * r)

        #  Only return Bz
        yout[i] += torch.dot(f * (3 * x * z), xin[::n_col_stride])
        yout[i] += torch.dot(f * (3 * y * z), xin[1::n_col_stride])
        yout[i] += torch.dot(f * (3 * z2 - r2), xin[2::n_col_stride])

    return None


def quadrupole_Bz_sus(xin, N_sensors, N_particles,
                      dip_r, pos_r, yout, n_col_stride):
    for i in range(N_sensors):
        dr = pos_r[i] - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = torch.sum(dr ** 2, dim=1)
        r = np.sqrt(r2)
        g = 1e-7 / (r2 * r2 * r2 * r)

        # Fill the Q array in the corresponding entries
        yout[i] += torch.dot(g * np.sqrt(3 / 2) * z * (-3 * r2 + 5 * z2), xin[3::n_col_stride])
        yout[i] += torch.dot(g * -np.sqrt(2) * x * (r2 - 5 * z2), xin[4::n_col_stride])
        yout[i] += torch.dot(g * -np.sqrt(2) * y * (r2 - 5 * z2), xin[5::n_col_stride])
        yout[i] += torch.dot(g * (5 / np.sqrt(2)) * (x2 - y2) * z, xin[6::n_col_stride])
        yout[i] += torch.dot(g * 5 * np.sqrt(2) * x * y * z, xin[7::n_col_stride])

    return None


# TODO: UPDATE to strides
def octupole_Bz_sus(xin, N_sensors, N_particles,
                    dip_r, pos_r, yout, n_col_stride):
    for i in range(N_sensors):
        dr = pos_r[i] - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = torch.sum(dr ** 2, dim=1)
        r4 = r2 ** 2
        r = np.sqrt(r2)
        g = 1e-7 / (r4 * r4 * r)

        # Fill the Q array using n_col_stride = 8
        yout[i] += torch.dot(g * (3 * (r2 ** 2) - 30 * r2 * z2
                               + 35 * (z2 * z2)) / np.sqrt(10),
                          xin[8*N_particles:9*N_particles])
        yout[i] += torch.dot(g * np.sqrt(15) * x * z * (-3 * r2 + 7 * z2) / 2,
                          xin[9*N_particles:10*N_particles])
        yout[i] += torch.dot(g * np.sqrt(15) * y * z * (-3 * r2 + 7 * z2) / 2,
                          xin[10*N_particles:11*N_particles])
        yout[i] += torch.dot(g * -np.sqrt(1.5) * (x2 - y2) * (r2 - 7 * z2),
                          xin[11*N_particles:12*N_particles])
        yout[i] += torch.dot(g * -np.sqrt(6) * x * y * (r2 - 7 * z2),
                          xin[12*N_particles:13*N_particles])
        yout[i] += torch.dot(g * 7 * x * (x2 - 3 * y2) * z / 2,
                          xin[13*N_particles:14*N_particles])
        yout[i] += torch.dot(g * -7 * y * (-3 * x2 + y2) * z / 2,
                          xin[14*N_particles:15*N_particles])

    return None


def Bflux_residual_f(xin, Bdata, N_sensors, N_cols, N_particles, particle_positions,
                     expansion_limit, scan_positions, engine='numba'):

    # x is input magnetic moment, output y data vector
    yout = torch.zeros(N_sensors)
    # loop through all scan points to calculate magnetic moment
    if engine == 'numba':
        # reshape the magnetic moment vector to order: [mx1 mx2 ... my1 my2 ... ]
        # so Numba can use the dot product more efficiently
        # xin = xin.reshape(N_particles, N_cols).T.flatten()

        dipole_Bz_sus(xin, N_sensors, N_particles, particle_positions, scan_positions, yout, N_cols)
        if expansion_limit in ['quadrupole', 'octupole']:
            quadrupole_Bz_sus(xin, N_sensors, N_particles, particle_positions, scan_positions, yout, N_cols)
        if expansion_limit in ['octupole']:
            octupole_Bz_sus(xin, N_sensors, N_particles, particle_positions, scan_positions, yout, N_cols)

    # the misfit functional that is minimised:
    mf = torch.sum((yout - Bdata) ** 2)
    print(f'{mf=}')
    # mf = torch.max(torch.abs((yout - Bdata) / torch.abs(Bdata)))
    return mf
