Installation
============

The `mmt_multipole_inversion` library can be installed directly via PyPI using 
`pip` (notice PyPI changed `_` to `-`)

.. code-block:: sh

    pip install mmt-multipole-inversion

Alternatively, you can clone the Github `repository`_ and install via
`pip`

.. code-block:: sh

    git clone https://github.com/Micromagnetic-Tomography/mmt_multipole_inversion.git
    cd multipole_inversion
    pip install .

Or using Poetry (recommended for development):

.. code-block:: sh

    git clone https://github.com/Micromagnetic-Tomography/mmt_multipole_inversion.git
    cd mmt_multipole_inversion
    poetry install

CUDA
----

To install the CUDA library you can clone the repository and define the
`CUDAHOME` or `CUDA_PATH` environment variables pointing to your CUDA
directory. For example

.. code-block:: sh

    git clone https://github.com/Micromagnetic-Tomography/mmt_multipole_inversion.git
    cd mmt_multipole_inversion
    export CUDA_PATH=/usr/local/cuda-11.7/
    poetry install

and you should see some compilation outputs from Cython. The build will also
work if the directory of the `nvcc` compiler is defined in your `PATH`
variable. 

Using Poetry will install the `mmt_multipole_inversion` in a new Python
environment. If you need it in your base environment, you can use `poetry
build` and then `pip install` the wheel (`.whl`) file that is generated in the
`dist` directory.

.. _repository: https://github.com/Micromagnetic-Tomography/multipole_inversion
