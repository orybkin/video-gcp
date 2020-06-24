from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='gcp',
    version='0.2dev',
    packages=['gcp', 'blox', 'gym-miniworld'],
    license='MIT License',
    ext_modules=cythonize(['gcp/evaluation/*.pyx']),
    include_dirs=[numpy.get_include(),],
)
