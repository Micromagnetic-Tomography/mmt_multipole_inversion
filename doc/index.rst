.. Multipole Inversion documentation master file, created by
   sphinx-quickstart on Fri Nov  5 16:01:58 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Multipole Inversion documentation
=================================

.. image:: images/multipoles.png

----

The MMT numerical library **Multipole Inversion** is a Python library to
perform numerical inversions of magnetic scan signals on a surface into a
single or multiple magnetic sources, which are modelled as physical point
sources. Numerical inversions are based on a spherical harmonic expansion of
the magnetic scalar potential of every source, from which dipole and higher
order multipole moments can be obtained. For mathematical details of the
multipole inversion technique please refer to:

| D. Cortés-Ortuño, K. Fabian, L. V. de Groot
| `Single Particle Multipole Expansions From Micromagnetic Tomography <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021GC009663>`_.
| Geochemistry, Geophysics, Geosystems **22(4)**, e2021GC009663 (2021)
| DOI: `https://doi.org/10.1029/2021GC009663 <https://doi.org/10.1029/2021GC009663>`

If you find this library useful, please cite it referring to the paper as::

    @article{https://doi.org/10.1029/2021GC009663,
      author = {Cortés-Ortuño, David and Fabian, Karl and de Groot, Lennart V.},
      title = {Single Particle Multipole Expansions From Micromagnetic Tomography},
      journal = {Geochemistry, Geophysics, Geosystems},
      volume = {22},
      number = {4},
      pages = {e2021GC009663},
      keywords = {magnetism, micromagnetic tomography, multipole, paleomagnetism, rock magnetism},
      doi = {https://doi.org/10.1029/2021GC009663},
      url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021GC009663},
      eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021GC009663},
      note = {e2021GC009663 2021GC009663},
      year = {2021}
    }

You can also cite the library directly using::

    @Misc{Cortes2022,
      author       = {Cortés-Ortuño, David and Fabian, Karl and de Groot, Lennart V.},
      title        = {{MMT Numerical Libraries: Multipole Inversion}},
      publisher    = {Zenodo},
      note         = {Github: \url{https://github.com/Micromagnetic-Tomography/multipole_inversion}},
      year         = {2022},
      doi          = {10.5281/zenodo.6473257},
      url          = {https://doi.org/10.5281/zenodo.6473257},
    }

Usage
-----

Please refer to the Tutorial: Basics section.

Source code
-----------

Documentation of every function and class provides more details on the methods
and options of the library, these are specified in the API section. You can
also check the project's `Github`_ repository.

.. _Github: https://github.com/Micromagnetic-Tomography/mmt_multipole_inversion

.. toctree::
   :maxdepth: 2
   :hidden:

   Introduction <self>
   installation.rst
   tutorial/tutorial_basics.ipynb
   autoapi/index.rst


.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
