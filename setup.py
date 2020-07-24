import setuptools
# from setuptools.extension import Extension
import sys

with open('README.md') as f:
    long_description = f.read()

setuptools.setup(
    # setup_requires=['cython'],  # not working (see the link at top)
    name='multipole_inversion',
    version='0.1',
    description=('Library to generate scan grid measurements from multipole'
                 ' sources and perform numerical inversions'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='D. Cortes, F. Out, M. Kosters, K. Fabian, L. V. de Groot',
    author_email='d.i.cortes@uu.nl',
    packages=setuptools.find_packages(),
    install_requires=['matplotlib',
                      'numpy',
                      'scipy',
                      'pathlib',
                      ],
    # TODO: Update license
    classifiers=['License :: BSD2 License',
                 'Programming Language :: Python :: 3 :: Only',
                 ],
)
