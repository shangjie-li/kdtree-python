from distutils.core import setup
from Cython.Build import cythonize

setup(name='kdtree_v5_cython',
      ext_modules=cythonize("kdtree_v5_cython.pyx"))
