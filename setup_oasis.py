#from distutils.core import setup
import numpy

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("oasis",
                     ["oasis.pyx"],
                     language='c++',
                     include_dirs=[numpy.get_include()]
                     )]

setup(
  name = 'oasis',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
