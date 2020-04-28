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

cpdef double logLikelihood_single_event(list hosts, object event, CosmologicalParameters omega, double m_th, int Ntot, EMcp = 0.):
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
    cdef unsigned int N_bright = len(hosts)
    cdef unsigned int N_dark = Ntot-N_bright
    cdef unsigned int N_dark_em, N_dark_no, N_em
    cdef double logTwoPiByTwo = 0.5*log(2.0*np.pi)
    cdef double logL_galaxy
    cdef double dl
    cdef double score_z, sigma_z
    cdef double logL_sum = -INFINITY
    cdef double logL_prod = 0.
    cdef double p_no_post_dark_em = 0.
    cdef double p_with_post_dark_em = 0.
    cdef double p_no_post_dark_no = 0.
    cdef double zmin, zmax, ramin, ramax, decmin, decmax

    cdef Galaxy mockgalaxy = Galaxy(-1, 0,0,0,False, weight = 1./Ntot)

    cdef object Schechter
    cdef double alpha, Mstar
    cdef double M_cutoff = -12.

    cdef np.ndarray[double, ndim=1, mode="c"] p_with_post = np.zeros(N_bright, dtype=np.float64)
    cdef double[::1] p_with_post_view = p_with_post
    cdef np.ndarray[double, ndim=1, mode="c"] p_no_post = np.zeros(N_bright, dtype=np.float64)
    cdef double[::1] p_no_post_view = p_no_post
    # Attenzione: non sono ancora stati sistemati i prior sulle posizioni per la dark galaxy
    # zmin   = RedshiftCalculation(event.LDmin, omega)
    # zmax   = RedshiftCalculation(event.LDmax, omega)
    zmin   = event.zmin
    zmax   = event.zmax
    ramin  = event.ramin
    ramax  = event.ramax
    decmin = event.decmin
    decmax = event.decmax

    Schechter, alpha, Mstar = SchechterMagFunction(-23., 0., h = omega.h)
    N_em = int(Integrate_Schechter_weighted(0., -23., -24., Schechter, M_cutoff)*Ntot)
    N_dark_em = N_em - N_bright
    N_dark_no = N_dark - N_dark_em

    if N_em <= N_bright:
        # If there are more galaxies than expected, set to 0 the number of unseen objects.
        N_dark_em = 0
        N_dark_no = Ntot - N_bright

    if EMcp:
        return ComputeLogLhWithPost(hosts[0], event, omega, zmin, zmax, ramin, ramax, decmin, decmax, m_th = m_th, M_cutoff = M_cutoff)

    print('Total number of galaxies: {0}'.format(Ntot))
    print('Bright galaxies: {0}'.format(N_bright))
    print('Dark galaxies: {0}'.format(N_dark))
    print('Emitting galaxies: {0}'.format(N_em))
    print('Emitting dark galaxies: {0}'.format(N_dark_em))
    print('Quiet dark galaxies: {0}'.format(N_dark_no))
    for i in range(N_bright):
        # Voglio calcolare, per ogni galassia, le due
        # quantità rilevanti descritte in CosmoInfer.
        p_no_post_view[i]   = ComputeLogLhNoPost(hosts[i], omega, zmin, zmax, m_th = m_th, M_cutoff = M_cutoff)
        p_with_post_view[i] = ComputeLogLhWithPost(hosts[i], event, omega, zmin, zmax, ramin, ramax, decmin, decmax, m_th = m_th, M_cutoff = M_cutoff)
        # print("{0}:\nwith:{1}\nno:{2}".format(i, p_with_post_view[i], p_no_post_view[i]))

    # Calcolo le likelihood anche per una singola dark galaxy
    if not (N_dark_em == 0):
        p_no_post_dark   = ComputeLogLhNoPost(mockgalaxy, omega, zmin, zmax, m_th = m_th, M_cutoff = M_cutoff)
        p_with_post_dark = ComputeLogLhWithPost(mockgalaxy, event, omega, zmin, zmax, ramin, ramax, decmin, decmax, m_th = m_th, M_cutoff = M_cutoff)
        # print("dark:\nwith:{1}\nno:{2}".format(i, p_with_post_dark, p_no_post_dark

    # Calcolo la probabilità di osservare una galassia che non possa emettere e che non riesco a vedere
    p_no_post_dark_no  = ComputeLogLhNoPost(mockgalaxy, omega, zmin, zmax, m_th = m_th, M_cutoff = M_cutoff, emission_flag = 0.)


    # Calcolo i termini che andranno sommati tra loro (logaritmi)
    cdef np.ndarray[double, ndim=1, mode="c"] addends = np.zeros(N_bright, dtype=np.float64)
    cdef double[::1] addends_view = addends
    cdef double sum = np.sum(p_no_post)

    for i in range(N_bright):
        addends_view[i] = sum - p_no_post_view[i] + p_with_post_view[i] + N_dark_em*p_no_post_dark + N_dark_no*p_no_post_dark_no
    cdef double dark_term = 0.
    if not (N_dark_em == 0):
        dark_term = sum + (N_dark_em-1)*p_no_post_dark + p_with_post_dark + N_dark_no*p_no_post_dark_no

    # Manca da fare la somma finale

    cdef double logL = -INFINITY
    for i in range(N_bright):
        logL = log_add(addends_view[i], logL)

    if np.isfinite(dark_term):
        for i in range(N_dark_em):
            logL = log_add(dark_term, logL)
    return logL

cdef LumDist(z, omega):
    return 3e3*(z + (1-omega.om +omega.ol)*z**2/2.)/omega.h

cdef dLumDist(z, omega):
    return 3e3*(1+(1-omega.om+omega.ol)*z)/omega.h

cdef RedshiftCalculation(LD, omega, zinit=0.3, limit = 0.001):
    '''
    Redshift given a certain luminosity, calculated by recursion.
    Limit is the less significative digit.
    '''
    LD_test = LumDist(zinit, omega)
    if abs(LD-LD_test) < limit :
        return zinit
    znew = zinit - (LD_test - LD)/dLumDist(zinit,omega)
    return RedshiftCalculation(LD, omega, zinit = znew)

cdef inline double absM(double z, double m, CosmologicalParameters omega):
    '''
    Magnitudine assoluta di soglia
    '''
    return m - 5.0*log10(1e5*omega.LuminosityDistance(z)) + 5.*log10(omega.h) # promemoria: 10^5 è Mpc/10pc + K-correction

cdef inline double appM(double z, double M, CosmologicalParameters omega):
    '''
    '''
    return M + 5.0*log10(1e5*omega.LuminosityDistance(z)) - 5.*log10(omega.h)

cdef inline double SchVar(double M, double Mstar) nogil:
    return pow(10,0.4*(Mstar-M))

cdef double myERF(double x) nogil:
    return (1+erf(x))/2.

cdef inline double gaussian(double x, double x0, double sigma) nogil:
    return exp(-(x-x0)**2/(2*sigma**2))/(sigma*sqrt(2*M_PI))

cdef double Integrand_dark(double z, CosmologicalParameters omega, double alpha, double Mstar, double Mmin, double Mmax, double CoVol):
    return -(gammainc(alpha+2,SchVar(Mmax, Mstar))-gammainc(alpha+2,SchVar(Mmin, Mstar)))*omega.ComovingVolumeElement(z)/CoVol

cdef double weighting_function(double M, double M_min, double M_cutoff = -10.):
    if (M > M_cutoff):
        return 0.
    else:
        return 1.


cdef double Integrate_Schechter_gaussian(double z_t, double m_i, double sigma_m, double M_min, double M_max, double M_th, CosmologicalParameters omega, object schechter):
    cdef unsigned int n = 100
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double I = 0.0
    cdef double m_app
    for i in range(n):
        if (M_view[i] < M_th):
            m_app = appM(z_t, M_view[i], omega)
            I += gaussian(m_i, m_app, sigma_m)*schechter(M_view[i])*dM*dM
    return I

cdef double Integrate_Schechter(double M_max, double M_min, double M_th, object schechter):
    cdef unsigned int n = 100
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double I = 0.0
    for i in range(n):
        if (M_view[i] > M_th):
            I += schechter(M_view[i])*dM
    return I

cdef double Integrate_Schechter_gaussian_weighted(double z_t, double m_i, double sigma_m, double M_min, double M_max, double M_th, CosmologicalParameters omega, object schechter, double M_cutoff):
    cdef unsigned int n = 100
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double I = 0.0
    cdef double m_app
    for i in range(n):
        if (M_view[i] < M_th):
            m_app = appM(z_t, M_view[i], omega)
            I += weighting_function(M_view[i], M_min, M_cutoff)*gaussian(m_i, m_app, sigma_m)*schechter(M_view[i])*dM*dM
    return I

cdef double Integrate_Schechter_weighted(double M_max, double M_min, double M_th, object schechter, double M_cutoff):
    cdef unsigned int n = 100
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double I = 0.0
    for i in range(n):
        if (M_view[i] > M_th):
            I += weighting_function(M_view[i], M_min, M_cutoff)*schechter(M_view[i])*dM
    return I

cdef double Integrate_Schechter_noemission_dark(double M_max, double M_min, double M_th, object Schechter, double M_cutoff):
    cdef unsigned int n = 100
    cdef np.ndarray[double, ndim=1, mode = 'c'] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max - M_min)/n
    cdef double I = 0.0
    for i in range(n):
        if (M_view[i] > M_th) and (M_view[i] > M_cutoff):
            I += Schechter(M_view[i])*dM
    return I

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cpdef double ComputeLogLhWithPost(Galaxy gal, object event, CosmologicalParameters omega, double zmin, double zmax, double ramin, double ramax, double decmin, double decmax, double M_cutoff, double m_th = 18., double M_max = 0, double M_min = -23):

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
    cdef double exp_post, prop_motion, rel_sigma, norm_post = 0.

    cdef double int_magnitude

    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h) # Modo semplice per tirare fuori i parametri di Schechter
    CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

    if gal.is_detected:
        LD = np.linspace(10,50,100)
        dLD = LD[2]-LD[1]
        for i in range(n):
            exp_post = exp(event.logP([LD[i],gal.DEC,gal.RA]))
            norm_post += LD[i]*dLD

        for i in range(n):
            prop_motion = 0.
            Mth = absM(z_view[i], m_th, omega)
            LD_i = omega.LuminosityDistance(z_view[i])
            int_magnitude = Integrate_Schechter_gaussian_weighted(z_view[i], gal.app_magnitude, gal.dapp_magnitude, M_min, M_max, Mth, omega, Schechter, M_cutoff)
            exp_post = exp(event.logP([LD_i,gal.DEC,gal.RA]))*LD_i**2.#/norm_post
            if(z_view[j] > 0.008):
                rel_sigma = 0.1*z_view[i]
            else:
                rel_sigma = 0.2*z_view[j]
            for j in range(n):
                # integrale in z_true
                prop_motion += gaussian(z_view[j],z_view[i], rel_sigma)*gaussian(gal.z,z_view[j], gal.dz)*dz
            # exp_post = exp(event.marg_logP(LD_i))
            # exp_post = gaussian(LD_i, 40.,5)*(LD_i**2.)*norm_post#exp(event.logP([40.,gal.DEC,gal.RA]))/norm_post
            I += dz*exp_post*int_magnitude*prop_motion*omega.ComovingVolumeElement(z_view[i])/(CoVol)
            # print('int_magnitude: {0}'.format(int_magnitude))
            # print('exp_post: {0}'.format(exp_post))
        # print('log int {0}: {1}'.format(gal.ID, log(I*gal.weight)))
        return log(I*gal.weight)
    else:
        for i in range(n):
            Mth = absM(z_view[i], m_th, omega)
            exp_post = exp(event.marg_logP(omega.LuminosityDistance(z_view[i])))
            int_magnitude = Integrate_Schechter_weighted(M_max, M_min, Mth, Schechter, M_cutoff)
            LD_i = omega.LuminosityDistance(z_view[i])
            # exp_post = gaussian(LD_i, 40.,5)*omega.LuminosityDistance(z_view[i])**2.
            I += dz*exp_post*int_magnitude*omega.ComovingVolumeElement(z_view[i])/CoVol
            # print('int_magnitude_dark: {0}'.format(int_magnitude))
        #     print('exp_post: {0}'.format(exp_post))
        # print('log int dark: {0}'.format(log(I*gal.weight)))
        return log(I*gal.weight)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double ComputeLogLhNoPost(Galaxy gal, CosmologicalParameters omega, double zmin, double zmax, M_cutoff, double m_th = 18, double M_max = 0, double M_min = -23, emission_flag = 1):
    '''
    Calcolo probabilità di osservare la galassia considerata.
    Si considera, nel caso di galassia osservata, la densità di probabilità dovuta alla misura (gaussiane con errore da determinarsi)
    Nel caso di galassia non osservata invece è necessario integrare sulle distribuzioni di probabilità (Schechter e dV/dz).

    m_th è la threshold dello strumento, Mmax è la massima magnitudine assoluta si ipotizza una galassia possa avere.

    Nota: Il prior in posizione per le galassie che non ho visto è 1/4pi (ovvero tutto il cielo) oppure una porzione corrispondente
    alla regione al 95%?
    '''
    cdef unsigned int i, n = 10
    cdef double mag_int
    cdef double LD_i

    cdef double I = 0.0
    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef double[::1] z_view = z

    cdef double dz = (zmax - zmin)/n

    cdef object Schechter
    cdef double alpha, Mstar, CoVol, Mth

    cdef double int_magnitude, prop_motion

    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h) # Modo semplice per tirare fuori i parametri di Schechter
    CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

    if gal.is_detected:
        for i in range(n):
            prop_motion = 0.
            Mth = absM(z_view[i], m_th, omega)
            int_magnitude = Integrate_Schechter_gaussian_weighted(z_view[i], gal.app_magnitude, gal.dapp_magnitude, M_min, M_max, Mth, omega, Schechter, M_cutoff)
            if(z_view[j] > 0.008):
                rel_sigma = 0.1*z_view[i]
            else:
                rel_sigma = 0.2*z_view[j]
            for j in range(n):
                # integrale in z_true
                prop_motion += gaussian(z_view[j],z_view[i], rel_sigma)*gaussian(gal.z,z_view[j], gal.dz)*dz
            I += dz*int_magnitude*prop_motion*omega.ComovingVolumeElement(z_view[i])/CoVol
            # print(gaussian(gal.z, z_view[i], gal.dz)*omega.ComovingVolumeElement(z_view[i])/CoVol)
        return log(I)

    else:
        if emission_flag:
            for i in range(n):
                Mth = absM(z_view[i], m_th, omega)
                I += dz*Integrate_Schechter_weighted(M_max, M_min, Mth, Schechter, M_cutoff)*omega.ComovingVolumeElement(z_view[i])/CoVol
        else:
            for i in range(n):
                Mth = absM(z_view[i], m_th, omega)
                I += dz*Integrate_Schechter_noemission_dark(M_max, M_min, Mth, Schechter, M_cutoff)*omega.ComovingVolumeElement(z_view[i])/CoVol
        return log(I)
