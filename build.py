# This build file for building cuda libraries using Cython is based on:
#   https://github.com/rmcgibbo/npcuda-example
# which holds a BSD2 license
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Robert T. McGibbon and the Authors
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
# IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -----------------------------------------------------------------------------

# This version has modifications to be used with the Poetry build system
# ans is licensed under the MIT license (see project's LICENSE file)
# Copyright (c) 2022, David Cortés-Ortuño and the Authors
# -----------------------------------------------------------------------------

import setuptools
from setuptools.extension import Extension
from setuptools.dist import Distribution
# setuptools contains the correct self.build_extensions function when
# writing our own custom_build_ext function:
# This might help: https://github.com/cython/cython/blob/master/docs/src/tutorial/appendix.rst
from setuptools.command.build_ext import build_ext
# cython and python dependency is handled by pyproject.toml
from Cython.Build import cythonize
import numpy
import os
from os.path import join as pjoin
from pathlib import Path


# -----------------------------------------------------------------------------
# CUDA SPECIFIC FUNCTIONS

def find_in_path(name, path):
    "Find a file in a search path"
    # adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME and CUDA_PATH env variable. If not
    found, everything is based on finding 'nvcc' in the PATH variable.
    """
    nvcc = None
    # First check if the CUDAHOME env variable is in use
    for cudaenv in ('CUDAHOME', 'CUDA_PATH'):
        if cudaenv in os.environ:
            home = os.environ[cudaenv]
            nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            # print(
            #     'The nvcc binary could not be located in'
            #     ' your $PATH. Either add it to your path, or set $CUDAHOME')
            return False

        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            # print('The CUDA %s path could not be located in %s' % (k, v))
            return False

    return cudaconfig


CUDA = locate_cuda()
# print(CUDA)


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's like a weird functional
    subclassing going on.
    """

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


# -----------------------------------------------------------------------------
# Compilation of CPP modules

# Define .cpp .c aguments passed to the compiler
# If using cuda, we set a dictionary to use different arguments for nvcc
# (see custom compiler)
if CUDA:
    com_args = dict(gcc=['-O3', '-fopenmp'])
else:
    com_args = ['-std=c99', '-O3', '-fopenmp']

link_args = ['-fopenmp']

extensions = []
# extensions = [
#     Extension("mmt_multipole_inversion.susceptibility_modules...",
#               ["",
#                ""],
#               extra_compile_args=com_args,
#               extra_link_args=link_args,
#               include_dirs=[numpy.get_include()]
#     )
# ]

if CUDA:
    # Add cuda options to the com_args dict and the extra library
    #
    # This syntax is specific to this build system
    # We're only going to use certain compiler args with nvcc and not with gcc
    # the implementation of this trick is in customize_compiler() below
    # For nvcc we use the Turing architecture: sm_75
    # See: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    # FMAD (floating-point multiply-add): turning off helps for numerical precission (useful
    #                                     for graphics) but this might slightly affect performance
    com_args['nvcc'] = ['-arch=sm_75', '--fmad=false', '--ptxas-options=-v',
                        '-c', '--compiler-options', "'-fPIC'"]
    extensions.append(
        Extension("mmt_multipole_inversion.susceptibility_modules.cuda.cudalib",
                  sources=["mmt_multipole_inversion/susceptibility_modules/cuda/cudalib.pyx",
                           "mmt_multipole_inversion/susceptibility_modules/cuda/spherical_harmonics_basis.cu"],
                  # library_dirs=[CUDA['lib64']],
                  libraries=['cudart'],
                  language='c++',
                  extra_compile_args=com_args,
                  include_dirs=[numpy.get_include(), CUDA['include'], '.'],
                  library_dirs=[CUDA['lib64']],
                  runtime_library_dirs=[CUDA['lib64']]
        )
    )

# -----------------------------------------------------------------------------

if CUDA is False:
    print("\nCUDAHOME or CUDA_PATH env variables not found: skipping cuda extensions")
    cmdclass = {'build_ext': build_ext}
else:
    cmdclass = {'build_ext': custom_build_ext}

# -----------------------------------------------------------------------------

# Source: https://stackoverflow.com/questions/60501869/poetry-cython-tests-nosetests
# distutils magic. This is essentially the same as calling
# python setup.py build_ext --inplace
dist = Distribution(attrs=dict(
            cmdclass=dict(build_ext=cmdclass['build_ext']),
            ext_modules=cythonize(extensions,
                                  language_level=3,
                                  ),
            zip_safe=False
        )
)
build_ext_cmd = dist.get_command_obj('build_ext')
build_ext_cmd.ensure_finalized()
build_ext_cmd.inplace = 1
build_ext_cmd.run()
