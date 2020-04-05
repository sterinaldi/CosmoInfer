# CosmoInfer
Inference of cosmological parameters from gravitational wave observation using the galaxy catalog method.
This code is based on Del Pozzo's and Laghi's cosmolisa (https://github.com/wdpozzo/cosmolisa) and Del Pozzo's 3d_volume (https://github.com/wdpozzo/3d_volume).

Requirements:

* numpy
* scipy
* healpy
* lal
* cython

Installation

python setup.py build_ext --inplace
or
python setup.py install