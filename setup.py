from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import sys
import numpy


with open('README.md') as f:
    readme = f.read()


with open('LICENSE') as f:
    license = f.read()


extra_compile_args = [
    '-DHAVE_SSE2'
]
if 'win' in sys.platform:
    pass
else:
    extra_compile_args += [
        '-std=c99',
        '-ffast-math',
        '-ftree-vectorize',
        '-march=native']

extra_link_args = []

ion_trapping_lib = cythonize([Extension(
    'ion_trapping.ion_trapping',
    sources=[
        'ion_trapping/ion_trapping.pyx',
        'ion_trapping/ion_trapping_lib.c'
    ],
    include_dirs=['ion_trapping', numpy.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args
    )])

packages = ['ion_trapping']

setup(
    name='ion_trapping_lib',
    version='0.0.0',
    description='Utilities for Penning trap simulations',
    long_description=readme,
    author='Dominic Meiser',
    author_email='dmeiser79@gmail.com',
    url='https://github.com/d-meiser/ion-trapping-notes',
    license=license,
    packages=packages,
    ext_modules=ion_trapping_lib
)
