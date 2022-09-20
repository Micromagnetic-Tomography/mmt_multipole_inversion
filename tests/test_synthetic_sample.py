import numpy as np
import mmt_multipole_inversion.magnetic_sample as msp
import mmt_multipole_inversion.multipole_inversion as minv
import mmt_multipole_inversion.plot_tools as minvp
from pathlib import Path
import pytest
import json
#
import requests, zipfile, io

import shapely.geometry as shg
import shapely.ops as sho

import matplotlib.pyplot as plt

# DATA ------------------------------------------------------------------------

# Download the data to this directory:
data_dir = Path('deGroot2018_data')
data_dir.mkdir(exist_ok=True)

if not any(data_dir.iterdir()):
    data_url = 'https://store.pangaea.de/Publications/deGroot-etal_2018/Micro-grain-data.zip'
    r = requests.get(data_url)
    # Pass the request output `r` (in byte-like format) to get a binary stream
    # from a data piece in memory (buffer) using BytesIO -> like a file in mem
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(data_dir)

# Now open the ZIP file with formatted data:
z = zipfile.ZipFile(data_dir / 'V2_2021-04.zip')
z.extractall(data_dir)

# -----------------------------------------------------------------------------
# DEFINE PARTICLES

data_dir = Path('deGroot2018_data/PDI-16803')

# location and name of QDM and cuboid file
ScanFile = data_dir / 'Area1-90-fig2MMT.txt'
CuboidFile = data_dir / 'FWInput-FineCuboids-A1.txt'

cuboid_data = np.loadtxt(CuboidFile, skiprows=0)
cuboid_data[:, 2] *= -1
cuboid_data_idxs = cuboid_data[:, 6].astype(np.int16)
cx, cy, cz, cdx, cdy, cdz = (cuboid_data[:, i] for i in range(6))
vols = 8 * cdx * cdy * cdz

vertexes = np.column_stack((cx - cdx, cy - cdy,
                            cx + cdx, cy - cdy,
                            cx + cdx, cy + cdy,
                            cx - cdx, cy + cdy)).reshape(-1, 4, 2)

particle_geoms_area1 = []
for p in np.unique(cuboid_data_idxs):
    polygons = map(shg.Polygon, vertexes[cuboid_data_idxs == p])
    pols = sho.unary_union([shg.Polygon(pt.exterior).buffer(0.0001, cap_style=3)
                            for pt in polygons])
    particle_geoms_area1.append(pols)

# Compute centre of mass (geometric centre)
particles = np.zeros((len(np.unique(cuboid_data_idxs)), 4))
centre = np.zeros(3)
for i, particle_idx in enumerate(np.unique(cuboid_data_idxs)):

    p = cuboid_data_idxs == particle_idx
    particle_vol = vols[p].sum()
    centre[0] = np.sum(cx[p] * vols[p]) / particle_vol
    centre[1] = np.sum(cy[p] * vols[p]) / particle_vol
    centre[2] = np.sum(cz[p] * vols[p]) / particle_vol

    particles[i][:3] = centre
    particles[i][3] = particle_vol

# -----------------------------------------------------------------------------
# SAMPLE
data_dir = Path('deGroot2018_data/PDI-16803')
BASE_DIR = Path('SyntheticSampleFiles')
BASE_DIR.mkdir(exist_ok=True)

# Scale the positions and columes by micrometres
np.savez(BASE_DIR / 'Area1_UMS_NPZ_ARRAYS',
         # Bz_array=,
         particle_positions=particles[:, :3] * 1e-6,
         # magnetization=self.magnetization,
         volumes=particles[:, 3] * 1e-18
         )

# Set dictionary
metadict = {}
metadict["Scan height Hz"] = 2e-6
metadict["Scan area x-dimension Sx"] = 351 * 1e-6
metadict["Scan area y-dimension Sy"] = 201 * 1e-6
metadict["Scan x-step Sdx"] = 1e-6
metadict["Scan y-step Sdy"] = 1e-6
metadict["Time stamp"] = '0000'
metadict["Number of particles"] = 8
# Important!:
metadict["Sensor dimensions"] = (0.5e-6, 0.5e-6)

with open(BASE_DIR / "AREA1_UMS_METADICT.json", 'w') as f:
    json.dump(metadict, f)


dG_Area1UMS = np.array([3544.3, 3923.7, 15346.8, 3770.7, 28147.8, 2845.9, 92191.2, 7154.4])

inv_area1_ums = minv.MultipoleInversion(BASE_DIR / "AREA1_UMS_METADICT.json",
                                        BASE_DIR / 'Area1_UMS_NPZ_ARRAYS.npz',
                                        expansion_limit='quadrupole',
                                        sus_functions_module='spherical_harmonics_basis_area'
                                        )

inv_area1_ums.Bz_array = np.loadtxt(data_dir / 'Area1-90-fig2MMT.txt')
inv_area1_ums.compute_inversion(rcond=1e-25, method='sp_pinv')

mag_area1_ums = inv_area1_ums.inv_multipole_moments / inv_area1_ums.volumes[:, None]
mag_area1_ums = np.sqrt(np.sum(mag_area1_ums[:, :3] ** 2, axis=1))

print('Magnetization using area sensors')
for i, m in enumerate(mag_area1_ums):
    print(f'i = {i + 1}   '
          f'M_i = {m:>11.4f} A/m   '
          f'M_dGroot = {dG_Area1UMS[i]:>11.4f} A/m   '
          f'Volume: {inv_area1_ums.volumes[i] * 1e18:>9.4f} µm^3',
          )
print()

# -----------------------------------------------------------------------------

inv_area1_ums = minv.MultipoleInversion(BASE_DIR / "AREA1_UMS_METADICT.json",
                                        BASE_DIR / 'Area1_UMS_NPZ_ARRAYS.npz',
                                        expansion_limit='quadrupole',
                                        sus_functions_module='spherical_harmonics_basis'
                                        )

inv_area1_ums.Bz_array = np.loadtxt(data_dir / 'Area1-90-fig2MMT.txt')
inv_area1_ums.sensor_dims = ()
inv_area1_ums.compute_inversion(rcond=1e-25, method='sp_pinv')

mag_area1_ums = inv_area1_ums.inv_multipole_moments / inv_area1_ums.volumes[:, None]
mag_area1_ums = np.sqrt(np.sum(mag_area1_ums[:, :3] ** 2, axis=1))

print('Magnetization using point source sensors')
for i, m in enumerate(mag_area1_ums):
    print(f'i = {i + 1}   '
          f'M_i = {m:>11.4f} A/m   '
          f'M_dGroot = {dG_Area1UMS[i]:>11.4f} A/m   '
          f'Volume: {inv_area1_ums.volumes[i] * 1e18:>9.4f} µm^3',
          )
