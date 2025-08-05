# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# from IPython.display import display, HTML
# display(HTML("<style>pre { white-space: pre !important; }</style>"))
# -

import PIL
import matplotlib.pyplot as plt
import numpy as np

with PIL.Image.open('drawing2.png') as img:
    img.load()
img

with PIL.Image.open('drawingRectWhite2.png') as im:
    # im.load()
    im = PIL.ImageChops.invert(im)
    im = im.convert(mode='P', palette=PIL.Image.Palette.ADAPTIVE, colors=2)
    # im = PIL.ImageChops.invert(im)

# +
# im = PIL.Image.open('drawing.png').convert(mode='1', dither=None)#.convert(mode='1')
# -

im

im_arr = np.asarray(im)[::-1, :]

plt.imshow(im_arr, origin='lower')
plt.show()

im_arr[-10, :]



# # Interpolating

im_arr

Sx, Sy = 15., 10.5
Sdx, Sdy = 0.5, 0.5

sx_range = np.arange(np.around(Sx / Sdx)) * Sdx
sy_range = np.arange(np.around(Sy / Sdy)) * Sdy

sy_range.shape

N_sensors = sx_range.shape[0] * sy_range.shape[0]

N_sensors

sx_range

sy_range

8 * 13

np.around(Sx / Sdx)

Sx / Sdx

scan_positions = np.ones((N_sensors, 3))
X_pos, Y_pos = np.meshgrid(sx_range, sy_range)
scan_positions[:, :2] = np.stack((X_pos, Y_pos), axis=2).reshape(-1, 2)
scan_positions[:, 2] *= 0.001
# mask_array = np.zeros(scan_positions.shape[])

scan_positions.shape

im_positions = np.ones((im_arr.shape[0] * im_arr.shape[1], 2))
# Note that the x range has the shape of columns!
im_xrange = np.linspace(sx_range[0], sx_range[-1], im_arr.shape[1])
im_yrange = np.linspace(sy_range[0], sy_range[-1], im_arr.shape[0])
imX_pos, imY_pos = np.meshgrid(im_xrange, im_yrange, indexing='xy')

np.stack((X_pos, Y_pos), axis=2).shape

im_positions = np.stack((imX_pos, imY_pos), axis=2).reshape(-1, 2)
# im_positions[:, 2] *= 0.001
# im.shape = (-1)

im_positions.shape

im_positions

im_arr.shape

import scipy.interpolate as si

np.meshgrid(im_xrange, im_yrange, indexing='ij')[0].shape

im_positions

im_arr[48]

plt.scatter(im_positions[:, 0], im_positions[:, 1], c=im_arr.reshape(-1), s=5)
plt.show()

# ## Regular Grid Interpolation

interp = si.RegularGridInterpolator((im_xrange, im_yrange), im_arr.T, method='nearest') 
idata = interp(scan_positions[:, [0, 1]])

plt.scatter(im_positions[:, 0], im_positions[:, 1], c=im_arr.reshape(-1), s=5, alpha=0.2, cmap='RdYlBu')
plt.scatter(scan_positions[:, 0], scan_positions[:, 1], c=idata.reshape(-1), s=25)
plt.show()

plt.imshow(idata.reshape(sy_range.shape[0], -1), origin='lower')
plt.show()

# ## RGI 2

# Notice that RegularGridInteprolator uses meshgrid with indexing='ij'
# So we must use the y array as the x input, to match the behavior of the library,
# where y varies across the rows and is constant across columns
# im_arr2 = im_arr[:, ::-1]
interp = si.RegularGridInterpolator((im_xrange, im_yrange), im_arr.T, method='nearest') 
# interp = si.LinearNDInterpolator(im_positions[:, [1, 0]], im_arr.reshape(-1))

interp((2., 4.8))

# +
# plt.imshow(im_arr.T)
# plt.show()
# -

scan_positions[:, [0, 1]]

idata = interp(scan_positions[:, [0, 1]])

plt.scatter(scan_positions[:, 0], scan_positions[:, 1], c=idata.reshape(-1), s=30)
plt.show()

plt.imshow(idata.astype(np.uint8).reshape(sy_range.shape[0], -1))
plt.show()

# ## Bivariate Spline

interp2 = si.RectBivariateSpline(im_xrange, im_yrange, im_arr.T, kx=1, ky=1)

idata2 = interp2(scan_positions[:, 0], scan_positions[:, 1], grid=False)

plt.scatter(scan_positions[:, 0], scan_positions[:, 1], c=idata2.reshape(-1), s=30)
plt.show()

# ## Using image resize

with PIL.Image.open('drawingRectWhite2.png') as imRes:
    imRes = PIL.ImageChops.invert(imRes)
    imRes = imRes.resize(size=(sx_range.shape[0], sy_range.shape[0]), resample=PIL.Image.Resampling.NEAREST)
    imRes = imRes.convert(mode='P', palette=PIL.Image.Palette.ADAPTIVE, colors=2)

imRes

# imRes = im.resize(size=(sy_range.shape[0], sx_range.shape[0]), resample=PIL.Image.Resampling.BILINEAR)
imRes_arr = np.asarray(imRes)[::-1, :]

plt.imshow(imRes_arr, origin='lower')
plt.show()

# +
plt.scatter(im_positions[:, 0], im_positions[:, 1], c=im_arr.reshape(-1), s=5, alpha=0.2, cmap='RdYlBu')

plt.scatter(scan_positions[:, 0], scan_positions[:, 1], 
            c=imRes_arr.reshape(-1), s=30)
plt.show()