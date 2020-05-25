from __future__ import division
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.math cimport log,exp,sqrt,cos,fabs,sin,sinh,M_PI,pow,log10,INFINITY
cimport cython
from scipy.integrate import quad
from scipy.special.cython_special cimport erf
from scipy.special import logsumexp
from scipy.optimize import newton
from schechter import *
from scipy.integrate import quad
from galaxy cimport Galaxy
from cosmology cimport CosmologicalParameters
import itertools as it
import sys


cdef inline double log_add(double x, double y): return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))
cdef inline double linear_density(double x, double a, double b): return a+log(x)*b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cpdef double logLikelihood_single_event(list hosts, object event, CosmologicalParameters omega, double m_th, int Ntot, int N_sources, int EMcp = 0, object completeness_file = None):
    """
    Likelihood function for a single GW event.
    Loops over all possible hosts to accumulate the likelihood
    Parameters:
    ===============
    hosts: :obj: 'numpy.array' with shape Nx3. The columns are redshift, redshift_error, angular_weight
    event: :obj: 'something.Event'. Stores posterior distributions for GW event.
    x: :obj: 'numpy.double' cpnest sampling array.
    meandl: :obj: 'numpy.double': mean of the DL marginal likelihood
    sigma: :obj:'numpy.double': standard deviation of the DL marginal likelihood
    omega: :obj:'lal.CosmologicalParameter': cosmological parameter structure
    Ntot: :obj: 'numpy.int': Total number of galaxies in the considered volume (seen and unseen)
    event_redshift: :obj:'numpy.double': redshift for the the GW event
    em_selection :obj:'numpy.int': apply em selection function. optional. default = 0
    """
    cdef unsigned int i
    cdef unsigned int N = len(hosts)
    cdef unsigned int M = Ntot-N
    cdef double p_no_post_dark = 0.
    cdef double p_with_post_dark = 0.
    cdef double zmin, zmax
    cdef int N_back = Ntot - N_sources

    cdef Galaxy mockgalaxy = Galaxy(-1, 0,0,0,False, weight = 1./Ntot)

    cdef np.ndarray[double, ndim=1, mode="c"] p_with_post = np.zeros(N, dtype=np.float64)
    cdef double[::1] p_with_post_view = p_with_post
    cdef np.ndarray[double, ndim=1, mode="c"] p_no_post = np.zeros(N, dtype=np.float64)
    cdef double[::1] p_no_post_view = p_no_post

    zmin   = event.zmin
    zmax   = event.zmax

    if  Ntot < N:
        M = 0

    M = 0
    for i in range(N):

        # Voglio calcolare, per ogni galassia, le due
        # quantità rilevanti descritte in CosmoInfer.

        p_no_post_view[i]   = ComputeLogLhNoPost(hosts[i], event, omega, zmin, zmax, N_sources, N_back, m_th = m_th)
        p_with_post_view[i] = ComputeLogLhWithPost(hosts[i], event, omega, zmin, zmax, m_th = m_th)
        # print('no post: {0}'.format(p_no_post[i]))
        # print('with post: {0}'.format(p_with_post[i]))

    # Calcolo le likelihood anche per una singola dark galaxy
    if not (M == 0):
        p_no_post_dark   = ComputeLogLhNoPost(mockgalaxy, event, omega, zmin, zmax, N_sources, N_back, m_th = m_th)
        p_with_post_dark = ComputeLogLhWithPost(mockgalaxy, event, omega, zmin, zmax, m_th = m_th)
        # print('no post dark: {0}'.format(p_no_post_dark))
        # print('with post dark: {0}'.format(p_with_post_dark))

    # Calcolo i termini che andranno sommati tra loro (logaritmi)
    cdef np.ndarray[double, ndim=1, mode="c"] addends = np.zeros(N, dtype=np.float64)
    cdef double[::1] addends_view = addends
    cdef double sum = np.sum(p_no_post)

    for i in range(N):
        addends_view[i] = sum - p_no_post_view[i] + p_with_post_view[i] + M*p_no_post_dark
    cdef double dark_term = 0.
    if not (M == 0):
        dark_term = sum + (M-1)*p_no_post_dark + p_with_post_dark

    # Manca da fare la somma finale

    cdef double logL = -INFINITY
    for i in range(N):
        logL = log_add(addends_view[i], logL)

    if np.isfinite(dark_term):
        for i in range(M):
            logL = log_add(dark_term, logL)
    print('return: {0}'.format(logL))
    return logL

cdef inline double absM(double z, double m, CosmologicalParameters omega):
    '''
    Magnitudine assoluta
    '''
    return m - 5.0*log10(1e5*omega.LuminosityDistance(z)) + 5.*log10(omega.h) # promemoria: 10^5 è Mpc/10pc + K-correction

cdef inline double appM(double z, double M, CosmologicalParameters omega):
    '''
    Magnitudine apparente
    '''
    return M + 5.0*log10(1e5*omega.LuminosityDistance(z)) - 5.*log10(omega.h)


cdef inline double gaussian(double x, double x0, double sigma) nogil:
    return exp(-(x-x0)**2/(2*sigma**2))/(sigma*sqrt(2*M_PI))

cdef double integrate_magnitude_source(object e, double m_i, double dm_i, double z_cosmo, CosmologicalParameters omega, int visibility, float m_th=18., float M_max = 0., float M_min = -23.):

    cdef unsigned int i, n = 100
    cdef double I = 0.
    cdef np.ndarray[double, ndim = 1, mode = 'c'] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double m_min = appM(z_cosmo, M_min, omega)
    cdef double m_max = appM(z_cosmo, M_max, omega)

    cdef double m_app

    for i in range(n):
        m_app = appM(z_cosmo, M_view[i], omega)
        if visibility: # and m_app < m_th :
            I += gaussian(m_i, m_app, dm_i)*e.mag_dist(M_view[i])*dM
            # I += e.mag_dist(absM(z_cosmo, m_i, omega))*dM/(M_max-M_min)
        elif not visibility and m_app > m_th:
            I += e.mag_dist(M_view[i])*dM/(appM(z_cosmo, M_max, omega)-appM(z_cosmo, M_min, omega))
    return I

cdef double integrate_magnitude_background(object e, double m_i, double dm_i, double z_cosmo, CosmologicalParameters omega, double N_sources, double N_back, double visibility = 0, M_max = 0., double M_min = -23., double m_th = 18.):

    cdef unsigned int i, n = 100
    cdef double I = 0.
    cdef np.ndarray[double, ndim = 1, mode = 'c'] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double m_min = appM(z_cosmo, M_min, omega)
    cdef double m_max = appM(z_cosmo, M_max, omega)

    cdef object Schechter
    cdef double alpha, Mstar

    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h=omega.h)

    cdef double m_app

    for i in range(n):
        m_app = appM(z_cosmo, M_view[i], omega)
        if visibility :#and m_app < m_th :
            #I += dM*gaussian(m_i, m_app, dm_i)*(N_sources*e.mag_dist(M_view[i])+N_back*Schechter(M_view[i]))/(N_sources+N_back)
            I += (dM/(M_max-M_min))*(N_sources*e.mag_dist(absM(z_cosmo, m_i, omega))+N_back*Schechter(absM(z_cosmo, m_i, omega)))/float(N_sources+N_back)
        elif not visibility and m_app > m_th:
            I += dM*((N_sources*e.mag_dist(M_view[i])+N_back*Schechter(M_view[i]))/(N_sources+N_back))/(appM(z_cosmo, M_max, omega)-appM(z_cosmo, M_min, omega))
    return I

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef double ComputeLogLhWithPost(Galaxy gal, object event, CosmologicalParameters omega, double zmin, double zmax, double m_th = 18., double M_max = 0, double M_min = -23):

    cdef unsigned int i, n = 100
    cdef double mag_int
    cdef double LD_i

    cdef double I = 0.0
    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef double[::1] z_view = z

    cdef double dz = (zmax - zmin)/n

    cdef double CoVol
    cdef object Schechter
    cdef double alpha, Mstar, Mth
    cdef double exp_post, prop_motion, rel_sigma
    cdef double CoVolEl
    cdef double int_magnitude

    CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

    if gal.is_detected:
        for i in range(n):
            LD_i = omega.LuminosityDistance(z_view[i])
            exp_post = exp(event.logP([LD_i,gal.DEC,gal.RA]))
            # Controllare e modellizzare, eventualmente
            if(z_view[i] > 0.008):
                rel_sigma = 0.1*z_view[i]
            else:
                rel_sigma = 0.2*z_view[i]
            prop_motion = gaussian(gal.z,z_view[i], rel_sigma)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
            int_magnitude = integrate_magnitude_source(event, gal.app_magnitude, gal.dapp_magnitude, z_view[i], omega, visibility = 1, m_th= m_th)
            I += dz*exp_post*prop_motion*CoVolEl*int_magnitude*prop_motion*CoVolEl
        if I > 0:
            return log(I)
        else:
            return -np.inf
    else:
        for i in range(n):
            LD_i = omega.LuminosityDistance(z_view[i])
            exp_post = exp(event.marg_logP(LD_i))
            int_magnitude = integrate_magnitude_source(event, 0, 0, z_view[i], omega, 0, m_th = m_th )
            prop_motion = 1./(zmax-zmin)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
            I += dz*exp_post*prop_motion*CoVolEl*int_magnitude*prop_motion*CoVolEl*(4*np.pi)
        if I > 0:
            return log(I)
        else:
            return -np.inf

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ComputeLogLhNoPost(Galaxy gal, object event, CosmologicalParameters omega, double zmin, double zmax, double N_sources, double N_back, double m_th = 18, double M_max = 0, double M_min = -23):
    cdef unsigned int i, n = 100
    cdef double mag_int
    cdef double LD_i

    cdef double I = 0.0
    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef double[::1] z_view = z

    cdef double dz = (zmax - zmin)/n

    cdef double CoVol
    cdef object Schechter
    cdef double alpha, Mstar, Mth
    cdef double exp_post, prop_motion, rel_sigma
    cdef double CoVolEl
    cdef double int_magnitude

    CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

    if gal.is_detected:
        for i in range(n):
            LD_i = omega.LuminosityDistance(z_view[i])
            exp_post = exp(event.logP([LD_i,gal.DEC,gal.RA]))
            # Controllare e modellizzare, eventualmente
            if(z_view[i] > 0.008):
                rel_sigma = 0.1*z_view[i]
            else:
                rel_sigma = 0.2*z_view[i]
            prop_motion = gaussian(gal.z,z_view[i], rel_sigma)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
            int_magnitude = integrate_magnitude_background(event, gal.app_magnitude, gal.dapp_magnitude, z_view[i], omega, N_sources, N_back, visibility = 1)
            I += dz*exp_post*prop_motion*CoVolEl*int_magnitude*prop_motion*CoVolEl

        if I > 0:
            return log(I)
        else:
            return -np.inf

    else:
        for i in range(n):
            LD_i = omega.LuminosityDistance(z_view[i])
            exp_post = exp(event.marg_logP(LD_i))
            int_magnitude = integrate_magnitude_background(event, gal.app_magnitude, gal.dapp_magnitude, z_view[i], omega, N_sources, N_back, visibility = 0)
            prop_motion = 1./(zmax-zmin)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
            I += dz*exp_post*prop_motion*CoVolEl*int_magnitude*prop_motion*CoVolEl*(4*np.pi)
        if I > 0:
            return log(I)
        else:
            return -np.inf
