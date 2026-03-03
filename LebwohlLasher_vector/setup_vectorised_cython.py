
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name = "LebwohlLasher_vectorised_cython",
    ext_modules=cythonize("LebwohlLasher_vectorised_cython.pyx"), include_dirs=[numpy.get_include()])