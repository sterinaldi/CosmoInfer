import numpy
from setuptools import setup, find_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from codecs import open
from os import path
from distutils.extension import Extension
from Cython.Build import cythonize
import os
if not("LAL_PREFIX" in os.environ):
    print("No LAL installation found, please install LAL from source or source your LAL installation")
    exit()
else:
    lal_prefix = os.environ.get("LAL_PREFIX")

# see https://stackoverflow.com/a/21621689/1862861 for why this is here
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        self.include_dirs.append(numpy.get_include())


lal_includes = lal_prefix+"/include"
lal_libs = lal_prefix+"/lib"

ext_modules=[
             Extension("schechter",
                       sources=["schechter_function.pyx"],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include()]
                       )
             ]

setup(
      name = "schechter",
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=[numpy.get_include()]
      )

ext_modules=[
             Extension("likelihood",
                       sources=["likelihood.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = [lal_libs],
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include(),lal_includes,"./"]
                       )
             ]
setup(
      name = "likelihood",
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=[numpy.get_include(),lal_includes,"./"]
      )

ext_modules=[
             Extension("cosmology",
                       sources=["cosmology.pyx"],
                       libraries=["m","lal"], # Unix-like specific
                       library_dirs = [lal_libs],
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include(),lal_includes,"./"]
                       )
             ]
setup(
      name = "cosmology",
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=[numpy.get_include(),lal_includes,"./"]
      )

ext_modules=[
             Extension("galaxy",
                       sources=["galaxy.pyx"],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include(),"./"]
                       )
             ]
setup(
      name = "galaxy",
      ext_modules = cythonize(ext_modules, language_level = "3"),
      include_dirs=[numpy.get_include(),"./"]
      )


from Cython.Build import cythonize

ext_modules=[
             Extension("volume_reconstruction.utils.cumulative",
                       sources=["volume_reconstruction/utils/cumulative.pyx"],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include(),"volume_reconstruction"]
                       ),
             Extension("volume_reconstruction.utils.utils",
                       sources=["volume_reconstruction/utils/utils.pyx"],
                       libraries=["m"], # Unix-like specific
                       extra_compile_args=["-O3","-ffast-math"],
                       include_dirs=[numpy.get_include(),"volume_reconstruction"]
                       )
             ]
setup(
    name = "volume_reconstruction",
    version='0.5',
    description='gw volume reconstruction',
    classifiers=[
                 'Development Status :: 3 - Beta',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3.7',
                 'Topic :: Data Analysis :: Physics',
                 ],
    keywords='gravitational wave black holes neutron stars',
    author='Walter Del Pozzo, Archisman Ghosh',
    author_email='walter.delpozzo@ligo.org, archisman.ghosh@ligo.org',
    license='MIT',
    cmdclass={'build_ext': build_ext},
    packages = find_packages(),
    ext_modules = cythonize(ext_modules, language_level = "3"),
    include_dirs=[numpy.get_include()],
    install_requires=['numpy','cpnest','corner','matplotlib','setuptools-git'],
    scripts  = ['bin/VolumeReconstruction'],
    include_package_data=False,
    zip_safe=False
)
