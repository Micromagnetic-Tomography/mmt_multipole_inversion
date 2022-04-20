# Multipole Inversion

![](doc/images/multipoles.png)

---

Library for the numerical inversion of a scan grid detecting the magnetic
signal from magnetic sources by means of a multipole expansion of the
potential of the sources.

This library has two main modules

- `multipole_inversion/magnetic_sample.py` : contains the `MagneticSample`
  class to create a magnetic system with magnetic point sources (dipole or
  higher order multipole sources) and generate the scan signal. This class also
  has methods to save the scan signal data in `npz` format and the scan grid
  specifications in `json` format

- `multipole_inversion/multipole_inversion.py` : contains the
  `MultipoleInversion` class to perform a numerical inversion from a scan
  signal data. This class requires scan surface specifications, particle (point
  source) locations and the scan signal data. These can be passed from the
  `MagneticSample` output or be sp[ecified manually (useful for combining with
  other workflows such as micromagnetic simulations)

An additional module to plot results from the inversions is provided in
`multipole_inversion/plot_tools.py`. Magnetic susceptibility and magnetic field
functions can be found in the main library as well, although not all of them
are documented in the tutorial yet.

# Installation

Via `pip -e .` from the base directory or using `poetry` (recommended for
development).

# Tutorial

For now you can visualize the Jupyter notebooks from the `jupytext` scripts in
the `doc/tutorial/` folder. These notebooks can also be generated from their associated `jupytext` script, for example,

```
jupytext --to notebook multipoles_inversion_test.py 
```

The documentation of the classes can be generated using `sphinx` and the
scripts to do this are located in the `doc` folder. Future releases of this
library will include an online documentation of this code.
