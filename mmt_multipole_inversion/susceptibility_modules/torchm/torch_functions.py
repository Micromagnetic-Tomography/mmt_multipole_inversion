import numpy as np
import torch

# NOTE: To make Torch minimization happy, using a larger scale for the field and
# the space dimensions, we will scale position, locally in this file, Using
# micrometers and nano Tesla units
µm = 1e6
nT = 1e9
# NOTE: Every term in the multipole expansion of the magnetic potential must have
#       units of Ampere A, so the magnetic moments must be scaled according to the
#       spatial factor proportional to the corresponding moment:
#       #
#                        Spatial factor     Mag Moment Units
#           Dipole:      1 / R^3            A * m^3
#           Quadrupole:  1 / R^5            A * m^5
#           Octupole:    1 / R^7            A * m^7
#       #
#       Also remember that the field is scaled by vacuum permeability ~ 4 pi 1e-7

def dipole_Bz_sus(xin, N_sensors, N_particles, dip_r, pos_r, yout, n_col_stride):
    for i in range(N_sensors):
        dr = µm *(pos_r[i] - dip_r)
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        z2 = z ** 2

        r2 = torch.sum(dr ** 2, dim=1)
        r = np.sqrt(r2)
        f = 1e-7 / (r2 * r2 * r)

        #  Only return Bz
        yout[i] += nT * torch.dot(f * (3 * x * z), xin[::n_col_stride])
        yout[i] += nT * torch.dot(f * (3 * y * z), xin[1::n_col_stride])
        yout[i] += nT * torch.dot(f * (3 * z2 - r2), xin[2::n_col_stride])

    return None


def quadrupole_Bz_sus(xin, N_sensors, N_particles, dip_r, pos_r, yout, n_col_stride):
    for i in range(N_sensors):
        dr = µm * (pos_r[i] - dip_r)
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        # r2 = torch.sum(dr ** 2, dim=1)
        r2 = x2 + y2 + z2
        r = np.sqrt(r2)
        g = 1e-7 / (r2 * r2 * r2 * r)
        
        yout[i] += nT * torch.dot(g * np.sqrt(3 / 2) * z * (-3 * r2 + 5 * z2), xin[3::n_col_stride])
        yout[i] += nT * torch.dot(g * -np.sqrt(2) * x * (r2 - 5 * z2), xin[4::n_col_stride])
        yout[i] += nT * torch.dot(g * -np.sqrt(2) * y * (r2 - 5 * z2), xin[5::n_col_stride])
        yout[i] += nT * torch.dot(g * (5 / np.sqrt(2)) * (x2 - y2) * z, xin[6::n_col_stride])
        yout[i] += nT * torch.dot(g * 5 * np.sqrt(2) * x * y * z, xin[7::n_col_stride])

    return None


# TODO: UPDATE to strides
def octupole_Bz_sus(xin, N_sensors, N_particles,
                    dip_r, pos_r, yout, n_col_stride):
    for i in range(N_sensors):
        dr = µm * (pos_r[i] - dip_r)
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = torch.sum(dr ** 2, dim=1)
        r4 = r2 ** 2
        r = np.sqrt(r2)
        g = 1e-7 / (r4 * r4 * r)

        # Fill the Q array using n_col_stride = 8
        yout[i] += nT * torch.dot(g * (3 * (r2 ** 2) - 30 * r2 * z2 + 35 * (z2 * z2)) / np.sqrt(10), xin[8::n_col_stride])
        yout[i] += nT * torch.dot(g * np.sqrt(15) * x * z * (-3 * r2 + 7 * z2) / 2, xin[9::n_col_stride])
        yout[i] += nT * torch.dot(g * np.sqrt(15) * y * z * (-3 * r2 + 7 * z2) / 2, xin[10::n_col_stride])
        yout[i] += nT * torch.dot(g * -np.sqrt(1.5) * (x2 - y2) * (r2 - 7 * z2), xin[11::n_col_stride])
        yout[i] += nT * torch.dot(g * -np.sqrt(6) * x * y * (r2 - 7 * z2), xin[12::n_col_stride])
        yout[i] += nT * torch.dot(g * 7 * x * (x2 - 3 * y2) * z / 2, xin[13::n_col_stride])
        yout[i] += nT * torch.dot(g * -7 * y * (-3 * x2 + y2) * z / 2, xin[14::n_col_stride])

    return None


def Bflux_residual_f(xin, Bdata, N_sensors, N_cols, N_particles, particle_positions,
                     expansion_limit, scan_positions, engine='numba', full_output=False):

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

    # The misfit functional that is minimised in Tesla units
    mf = torch.sum((yout - nT * Bdata) ** 2)

    # Using torch's norm:
    # mf = torch.norm(yout - Bdata, p=torch.inf)

    # print(f'{mf=}')
    # mf = torch.max(torch.abs((yout - Bdata)))
    if full_output:
        return yout, mf
    else:
        return mf
