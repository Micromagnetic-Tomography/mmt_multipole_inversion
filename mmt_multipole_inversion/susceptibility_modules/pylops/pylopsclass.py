from pylops import LinearOperator
import numpy as np
import numba
from numba import prange


# basic functions
@numba.jit(nopython=True, parallel=True)
def dipole_Bz_sus(xin, N_sensors, N_particles,
                  dip_r, pos_r, yout, n_col_stride):
    for i in prange(N_sensors):
        dr = pos_r[i] - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        z2 = z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)
        f = 1e-7 / (r2 * r2 * r)

        #  Only return Bz
        yout[i] += np.dot(f * (3 * x * z), xin[:N_particles])
        yout[i] += np.dot(f * (3 * y * z), xin[N_particles:2*N_particles])
        yout[i] += np.dot(f * (3 * z2 - r2), xin[2*N_particles:3*N_particles])

    return None


@numba.jit(nopython=True, parallel=True)
def quadrupole_Bz_sus(xin, N_sensors, N_particles,
                      dip_r, pos_r, yout, n_col_stride):
    for i in prange(N_sensors):
        dr = pos_r[i] - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)
        g = 1e-7 / (r2 * r2 * r2 * r)

        # Fill the Q array in the corresponding entries
        yout[i] += np.dot(g * np.sqrt(3 / 2) * z * (-3 * r2 + 5 * z2),
                          xin[3*N_particles:4*N_particles])
        yout[i] += np.dot(g * -np.sqrt(2) * x * (r2 - 5 * z2),
                          xin[4*N_particles:5*N_particles])
        yout[i] += np.dot(g * -np.sqrt(2) * y * (r2 - 5 * z2),
                          xin[5*N_particles:6*N_particles])
        yout[i] += np.dot(g * (5 / np.sqrt(2)) * (x2 - y2) * z,
                          xin[6*N_particles:7*N_particles])
        yout[i] += np.dot(g * 5 * np.sqrt(2) * x * y * z,
                          xin[7*N_particles:8*N_particles])

    return None


@numba.jit(nopython=True, parallel=True)
def octupole_Bz_sus(xin, N_sensors, N_particles,
                    dip_r, pos_r, yout, n_col_stride):
    for i in prange(N_sensors):
        dr = pos_r[i] - dip_r
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2

        r2 = np.sum(dr ** 2, axis=1)
        r4 = r2 ** 2
        r = np.sqrt(r2)
        g = 1e-7 / (r4 * r4 * r)

        # Fill the Q array using n_col_stride = 8
        yout[i] += np.dot(g * (3 * (r2 ** 2) - 30 * r2 * z2
                               + 35 * (z2 * z2)) / np.sqrt(10),
                          xin[8*N_particles:9*N_particles])
        yout[i] += np.dot(g * np.sqrt(15) * x * z * (-3 * r2 + 7 * z2) / 2,
                          xin[9*N_particles:10*N_particles])
        yout[i] += np.dot(g * np.sqrt(15) * y * z * (-3 * r2 + 7 * z2) / 2,
                          xin[10*N_particles:11*N_particles])
        yout[i] += np.dot(g * -np.sqrt(1.5) * (x2 - y2) * (r2 - 7 * z2),
                          xin[11*N_particles:12*N_particles])
        yout[i] += np.dot(g * -np.sqrt(6) * x * y * (r2 - 7 * z2),
                          xin[12*N_particles:13*N_particles])
        yout[i] += np.dot(g * 7 * x * (x2 - 3 * y2) * z / 2,
                          xin[13*N_particles:14*N_particles])
        yout[i] += np.dot(g * -7 * y * (-3 * x2 + y2) * z / 2,
                          xin[14*N_particles:15*N_particles])

    return None


@numba.jit(nopython=True, parallel=True)
def dipole_Bz_sus_t(xin, dip_r, pos_r, yout, N_particles, _N_cols):
    for i in prange(N_particles):
        dr = pos_r - dip_r[i]
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        z2 = z ** 2
        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)
        f = 1e-7 / (r2 * r2 * r)
        yout[0+i*_N_cols] = np.dot(f * (3 * x * z), xin)
        yout[1+i*_N_cols] = np.dot(f * (3 * y * z), xin)
        yout[2+i*_N_cols] = np.dot(f * (3 * z2 - r2), xin)

    return None


@numba.jit(nopython=True, parallel=True)
def quadrupole_Bz_sus_t(xin, dip_r, pos_r, yout, N_particles, _N_cols):
    for i in prange(N_particles):
        dr = pos_r - dip_r[i]
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2
        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)
        g = 1e-7 / (r2 * r2 * r2 * r)
        yout[3+i*_N_cols] = np.dot(g * np.sqrt(3 / 2) * z * (-3 * r2 + 5 * z2),
                                   xin)
        yout[4+i*_N_cols] = np.dot(g * -np.sqrt(2) * x * (r2 - 5 * z2), xin)
        yout[5+i*_N_cols] = np.dot(g * -np.sqrt(2) * y * (r2 - 5 * z2), xin)
        yout[6+i*_N_cols] = np.dot(g * (5 / np.sqrt(2)) * (x2 - y2) * z, xin)
        yout[7+i*_N_cols] = np.dot(g * 5 * np.sqrt(2) * x * y * z, xin)

    return None


@numba.jit(nopython=True, parallel=True)
def octupole_Bz_sus_t(xin, dip_r, pos_r, yout, N_particles, _N_cols):
    for i in prange(N_particles):
        dr = pos_r - dip_r[i]
        x, y, z = dr[:, 0], dr[:, 1], dr[:, 2]
        x2, y2, z2 = x ** 2, y ** 2, z ** 2
        r2 = np.sum(dr ** 2, axis=1)
        r = np.sqrt(r2)
        r4 = r2 ** 2
        g = 1e-7 / (r4 * r4 * r)
        yout[8+i*_N_cols] = np.dot(g * (3 * (r2 ** 2) - 30 * r2 * z2 + 35 *
                                        (z2 * z2)) / np.sqrt(10), xin)
        yout[9+i*_N_cols] = np.dot(g * np.sqrt(15) * x * z *
                                   (-3 * r2 + 7 * z2) / 2, xin)
        yout[10+i*_N_cols] = np.dot(g * np.sqrt(15) * y * z *
                                    (-3 * r2 + 7 * z2) / 2, xin)
        yout[11+i*_N_cols] = np.dot(g * -np.sqrt(1.5) * (x2 - y2) *
                                    (r2 - 7 * z2), xin)
        yout[12+i*_N_cols] = np.dot(g * -np.sqrt(6) * x * y *
                                    (r2 - 7 * z2), xin)
        yout[13+i*_N_cols] = np.dot(g * 7 * x * (x2 - 3 * y2) * z / 2, xin)
        yout[14+i*_N_cols] = np.dot(g * -7 * y * (-3 * x2 + y2) * z / 2, xin)

    return None


class GreensMatrix(LinearOperator):
    def __init__(self, N_sensors, _N_cols, N_particles, particle_positions,
                 expansion_limit, scan_positions, verbose=False,
                 engine='numba', dtype='float64'):
        self.N_sensors = N_sensors
        self._N_cols = _N_cols
        self.N_particles = N_particles
        self.shape = (self.N_sensors, self.N_particles * self._N_cols)
        self.particle_positions = particle_positions
        if expansion_limit in ['dipole', 'quadrupole', 'octupole']:
            self.expansion_limit = expansion_limit
        else:
            raise Exception('Specify a correct expansion limit')
        self.scan_positions = scan_positions
        self.verbose = verbose

        self.engine = engine
        self.explicit = False
        self.dtype = dtype
        self.matvec_count = 0
        self.rmatvec_count = 0

        self.matvecList = [dipole_Bz_sus]
        self.rmatvecList = [dipole_Bz_sus_t]
        if self.expansion_limit in ['quadrupole', 'octupole']:
            self.matvecList.append(quadrupole_Bz_sus)
            self.rmatvecList = [quadrupole_Bz_sus_t]
        if self.expansion_limit in ['octupole']:
            self.matvecList.append(octupole_Bz_sus)
            self.rmatvecList = [octupole_Bz_sus_t]

    def _matvec(self, xin):
        # x is input magnetic moment, output y data vector
        yout = np.zeros(self.shape[0])
        # loop through all scan points to calculate magnetic moment
        if self.engine == 'numba':
            # reshape the magnetic moment vector to order: [mx1 mx2 ... my1 my2 ... ]
            # so Numba can use the dot product more efficiently
            xin = xin.reshape(self.N_particles, self._N_cols).flatten('F')

            for FSus in self.matvecList:
                FSus(xin, self.N_sensors, self.N_particles,
                     self.particle_positions, self.scan_positions, yout,
                     self._N_cols)

        return yout

    def _rmatvec(self, xin):
        # x is the data vector, output y is adjoint magnetic moment vector
        yout = np.zeros(self.shape[1])
        if self.engine == 'numba':
            for RFSus in self.rmatvecList:
                RFSus(xin, self.particle_positions, self.scan_positions,
                      yout, self.N_particles, self._N_cols)
        return yout
