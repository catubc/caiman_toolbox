from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("oasis",
                     ["oasis.pyx"],
                     language='c++',
                     )]

setup(
  name = 'oasis',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
