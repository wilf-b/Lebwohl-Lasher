from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

compile_args = ["-O3", "-ffast-math", "-fopenmp"]
link_args = ["-fopenmp"]

extensions = [
    Extension(
        "LebwohlLasher_cython_parallel",
        ["LebwohlLasher_cython_parallel.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="LebwohlLasher_cython_parallel",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)