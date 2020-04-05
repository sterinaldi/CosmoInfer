import numpy as np
import lal
from likelihood import *
from galaxies import *

mock = Galaxy(0, 0, 0, 0.1, 0, abs_magnitude = -10, app_magnitude = 4)

omega = lal.CreateCosmologicalParameters(0.7,0.7,0.3,0,0,0)

Int = ComputeLogLhNoPost(mock, omega, 0.05, 0.2)
print(Int)
