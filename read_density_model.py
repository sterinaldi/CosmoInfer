from volume_reconstruction.dpgmm.dpgmm import *
from volume_reconstruction.utils.utils import *
import dill as pickle
from scipy.special import logsumexp
import numpy as np

def logPosterior(args):
    density,celestial_coordinates = args
    cartesian_vect = celestial_to_cartesian(celestial_coordinates)
    logPs = [np.log(density[0][ind])+prob.logProb(cartesian_vect) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)+np.log(Jacobian(cartesian_vect))


file_path = 'VolRec/dpgmm_density.p'
density_model = pickle.load(open(file_path, 'rb'))

gal_1 = np.array([50., 0., 0.])
gal_2 = np.array([30., 0.5, 3.])
gal_3 = np.array([70., -0.5, 1.])

print(logPosterior((density_model, gal_1)))
print(logPosterior((density_model, gal_2)))
print(logPosterior((density_model, gal_3)))
