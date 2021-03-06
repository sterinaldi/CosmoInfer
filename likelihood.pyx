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
from scipy.stats import poisson
from galaxy cimport Galaxy
from cosmology cimport CosmologicalParameters
import itertools as it
import sys
import ray



def logLikelihood_single_event(list hosts, object event, CosmologicalParameters omega, double m_th, int EMcp = 0, str completeness_file = None):
    return _logLikelihood_single_event(hosts, event, omega, m_th, EMcp = EMcp, completeness_file = completeness_file)


cdef inline double log_add(double x, double y): return x+log(1.0+exp(y-x)) if x >= y else y+log(1.0+exp(x-y))
cdef inline double linear_density(double x, double a, double b): return a+log(x)*b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double _logLikelihood_single_event(list hosts, object event, CosmologicalParameters omega, double m_th, int EMcp = 0, str completeness_file = None):

    cdef unsigned int i
    cdef unsigned int N = len(hosts)
    cdef int M
    cdef double logTwoPiByTwo = 0.5*log(2.0*np.pi)
    cdef double logL_galaxy
    cdef double dl
    cdef double score_z, sigma_z
    cdef double logL_sum = -INFINITY
    cdef double logL_prod = 0.
    cdef double p_no_post_dark = 0.
    cdef double p_with_post_dark = 0.
    cdef double p_noemission
    cdef double zmin, zmax, ramin, ramax, decmin, decmax
    cdef double M_min, M_max, M_cutoff = -15.
    cdef object schechter
    cdef double alpha, Mstar
    cdef int avg_N_bright, avg_N_em, avg_N_tot, N_em, Ntot
    cdef int N_noem
    cdef object file_comp

    cdef np.ndarray[double, ndim=1, mode="c"] addends = np.zeros(N, dtype=np.float64)
    cdef double[::1] addends_view = addends
    cdef double sum = 0.
    cdef double fixed_Nem = -INFINITY
    cdef double I_Ntot, I_Nem
    cdef double logL = -INFINITY
    cdef double dark_term = 0.
    cdef object pNtot, pNem, pNbright

    cdef Galaxy mockgalaxy = Galaxy(-1, 0,0,0,False, weight = 1.)
    cdef np.ndarray[double, ndim=1, mode="c"] p_with_post = np.zeros(N, dtype=np.float64)
    cdef double[::1] p_with_post_view = p_with_post
    cdef np.ndarray[double, ndim=1, mode="c"] p_no_post = np.zeros(N, dtype=np.float64)
    cdef double[::1] p_no_post_view = p_no_post
    zmin   = event.zmin
    zmax   = event.zmax
    ramin  = event.ramin
    ramax  = event.ramax
    decmin = event.decmin
    decmax = event.decmax

    # if EMcp:
    #     N_em   = 1
    #     M      = 0
    #     N_noem = 0
    # if N_em <= N:
    #     M = 0
    #     N_noem = avg_Ntot - N
    #     if avg_Ntot < N:
    #         N_noem = 0

    # if completeness_file is not None:
    #     file_comp = open(completeness_file, 'a')
    #     file_comp.write('\n{0}\t{1}\t{2}\t{3}'.format(omega.h, N_em, N, M))
    #     file_comp.close()

    for i in range(N):
        # Voglio calcolare, per ogni galassia, le due
        # quantità rilevanti descritte in CosmoInfer.
        # print('Galaxy %d of %d\r' % (i+1, N))
        p_no_post_view[i]   = ComputeLogLhNoPost(hosts[i], omega, zmin, zmax, m_th = m_th, M_cutoff = M_cutoff)
        p_with_post_view[i] = ComputeLogLhWithPost(hosts[i], event, omega, zmin, zmax, ramin, ramax, decmin, decmax, m_th = m_th, M_cutoff = M_cutoff)
    # Calcolo le likelihood anche per una singola dark galaxy
    p_no_post_dark   = ComputeLogLhNoPost(mockgalaxy, omega, zmin, zmax, m_th = m_th, M_cutoff = M_cutoff)
    p_with_post_dark = ComputeLogLhWithPost(mockgalaxy, event, omega, zmin, zmax, ramin, ramax, decmin, decmax, m_th = m_th, M_cutoff = M_cutoff)
    p_noemission     = ComputeLogLhNoEmission(mockgalaxy, omega, zmin, zmax, m_th = m_th, M_cutoff = M_cutoff)

    # file_comp = open(completeness_file, 'a')
    # file_comp.write('\n{0}\t{1}\t{2}\t{3}'.format(omega.h, p_no_post_dark, p_with_post_dark, p_noemission))

    M_max = -6.
    M_min = -23.

    avg_N_tot = ComputeAverageTotal(event.area, zmin, zmax, omega, 0.66)

    pNtot = poisson(avg_N_tot).pmf
    schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h)
    Ntot_array = np.arange(int(avg_N_tot-3*np.sqrt(avg_N_tot)), int(avg_N_tot+3*np.sqrt(avg_N_tot)))

    I_Ntot = -INFINITY
    for Ntot in Ntot_array:
        avg_N_em     = int(Integrate_Schechter(M_max, M_min, M_min-3, schechter,M_cutoff)*Ntot)
        pNem         = poisson(avg_N_em).pmf
        avg_N_bright = ComputeAverageBright(event.area, M_min, M_max, zmin, zmax, m_th, M_cutoff, Ntot, omega, schechter)
        pNbright     = poisson(avg_N_bright).pmf
        Nem_array    = np.arange(int(avg_N_em-3*np.sqrt(avg_N_em)), int(avg_N_em+3*np.sqrt(avg_N_em)))
        I_Nem = -INFINITY
        for N_em in Nem_array:
            M = N_em-N
            if M < 0:
                M = 0
                N_em = N
            N_noem = Ntot-N_em
            if N_noem < 0:
                N_noem = 0

            # print(avg_Ntot, avg_N_em, avg_N_bright, N, M, N_em, N_noem)
    # Calcolo i termini che andranno sommati tra loro (logaritmi)


            sum = np.sum(p_no_post)
            for i in range(N):
                addends_view[i] = sum - p_no_post_view[i] + p_with_post_view[i] + M*p_no_post_dark + N_noem*p_noemission


            if not (M == 0):
                dark_term = sum + (M-1)*p_no_post_dark + p_with_post_dark + N_noem*p_noemission

            fixed_Nem = -INFINITY
            for i in range(N):
                fixed_Nem = log_add(addends_view[i], fixed_Nem)
            if np.isfinite(dark_term):
                for i in range(M):
                    fixed_Nem = log_add(dark_term, fixed_Nem)

            fixed_Nem += log(pNem(N_em))
            if N <= N_em:
                I_Nem = log_add(fixed_Nem, I_Nem)

        I_Ntot = log_add(I_Nem + log(pNtot(Ntot)) + log(pNbright(N)), I_Ntot)

    # print('avg_Ntot, avg_N_em, avg_N_bright, N, M, N_em, N_noem')
    logL = I_Ntot

    print('h=%.3f done' %(omega.h))

    if np.isfinite(logL):
        return logL
    else:
        return -INFINITY



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
    return m - 5.0*log10(1e5*omega.LuminosityDistance(z)) + 5.*log10(omega.h) # promemoria: 10^5 è Mpc/10pc + K-correction

cdef inline double appM(double z, double M, CosmologicalParameters omega):
    return M + 5.0*log10(1e5*omega.LuminosityDistance(z)) - 5.*log10(omega.h)

cdef inline double gaussian(double x, double x0, double sigma) nogil:
    return exp(-(x-x0)**2/(2*sigma**2))/(sigma*sqrt(2*M_PI))

cdef int ComputeAverageTotal(object area, double z_min, double z_max, CosmologicalParameters omega, double density):

    z_array = np.linspace(z_min, z_max, 1000)
    dz = z_array[2]-z_array[1]
    I_z = 0.

    for z in z_array:
        I_z += density*omega.ComovingVolumeElement(z)*(area(omega.LuminosityDistance(z))/(4*np.pi))*dz

    return int(I_z)


cdef int ComputeAverageBright(object area, double M_min, double M_max, double z_min, double z_max, double mth, double M_cutoff, int Ntot, CosmologicalParameters omega, object Schechter):

    M_array = np.linspace(M_min, M_max, 1000)
    dM = M_array[2]-M_array[1]
    z_array = np.linspace(z_min, z_max,1000)
    dz = z_array[2]-z_array[1]

    norm = 0.
    for z in z_array:
        norm += omega.ComovingVolumeElement(z)*(area(omega.LuminosityDistance(z))/(4*np.pi))*dz


    I_z = 0.
    for z in z_array:
        Mth = absM(z, mth, omega)
        I_M = 0.
        for M in M_array:
            if M < Mth and M < M_cutoff:
                I_M += Schechter(M)*dM
        I_z += omega.ComovingVolumeElement(z)*(area(omega.LuminosityDistance(z))/(4*np.pi))*dz*I_M/norm

    return int(Ntot*I_z)



cdef double Integrate_Schechter_gaussian(double z_t, double m_i, double sigma_m, double M_min, double M_max, double m_th, CosmologicalParameters omega, object schechter, double M_cutoff):
    cdef unsigned int n = 1000
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double I = 0.0
    cdef double M_app
    for i in range(n):
        M_app = appM(z_t, M_view[i], omega)
        if (M_app < m_th) and (M_view[i] < M_cutoff):
            I += gaussian(m_i, M_app, sigma_m)*dM*schechter(M_view[i])
            #I += schechter(absM(z_t, m_i, omega))*dM
    return I

cdef double Integrate_Schechter(double M_max, double M_min, double M_th, object schechter, double M_cutoff):
    cdef unsigned int n = 1000
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double I = 0.0
    for i in range(n):
        if (M_view[i] > M_th) and (M_view[i] < M_cutoff):
            I += schechter(M_view[i])*dM
    return I

cdef double Integrate_Schechter_above(double M_max, double M_min, double M_th, object schechter, double M_cutoff):
    cdef unsigned int n = 1000
    cdef np.ndarray[double, ndim=1, mode = "c"] M = np.linspace(M_min, M_max, n, dtype = np.float64)
    cdef double[::1] M_view = M
    cdef double dM = (M_max-M_min)/n
    cdef double I = 0.0
    for i in range(n):
        if (M_view[i] > M_th) and (M_view[i] > M_cutoff):
            I += schechter(M_view[i])*dM
    return I

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

cdef double ComputeLogLhWithPost(Galaxy gal, object event, CosmologicalParameters omega, double zmin, double zmax, double ramin, double ramax, double decmin, double decmax, double M_cutoff, double m_th = 18., double M_max = -6., double M_min = -23):

    cdef unsigned int i, n = 1000
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
    cdef double CoVolEl
    cdef double int_magnitude

    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h) # Modo semplice per tirare fuori i parametri di Schechter
    CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

    if gal.is_detected:
        for i in range(n):
            Mth = absM(z_view[i], m_th, omega)
            LD_i = omega.LuminosityDistance(z_view[i])
            exp_post = exp(event.logP([LD_i,gal.DEC,gal.RA]))
            # exp_post = gaussian(LD_i, 40., 5.)
            if(z_view[i] > 0.008):
                rel_sigma = 0.1*z_view[i]
            else:
                rel_sigma = 0.2*z_view[i]
            prop_motion = gaussian(gal.z,z_view[i], rel_sigma)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
            int_magnitude = Integrate_Schechter_gaussian(z_view[i], gal.app_magnitude, gal.dapp_magnitude, M_min, M_max, m_th, omega, Schechter, M_cutoff)
            I += dz*exp_post*prop_motion*CoVolEl*int_magnitude*prop_motion*CoVolEl
        return log(I)
    else:
        for i in range(n):
            Mth = absM(z_view[i], m_th, omega)
            LD_i = omega.LuminosityDistance(z_view[i])
            exp_post = exp(event.marg_logP(LD_i))
            int_magnitude = Integrate_Schechter(M_max, M_min, Mth, Schechter, M_cutoff)
            prop_motion = 1./(zmax-zmin)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)

            I += dz*exp_post*prop_motion*CoVolEl*int_magnitude*prop_motion*CoVolEl*4*np.pi
        return log(I)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double ComputeLogLhNoPost(Galaxy gal, CosmologicalParameters omega, double zmin, double zmax, double M_cutoff, double m_th = 18., double M_max = -6., double M_min = -23.):
    cdef unsigned int i, n = 1000
    cdef double mag_int
    cdef double LD_i

    cdef double I = 0.0
    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef double[::1] z_view = z

    cdef double dz = (zmax - zmin)/n

    cdef object Schechter
    cdef double alpha, Mstar, CoVol, Mth

    cdef double int_magnitude, prop_motion, CoVolEl

    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h) # Modo semplice per tirare fuori i parametri di Schechter
    CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

    if gal.is_detected:
        for i in range(n):
            Mth = absM(z_view[i], m_th, omega)
            if(z_view[i] > 0.008):
                rel_sigma = 0.1*z_view[i]
            else:
                rel_sigma = 0.2*z_view[i]
            prop_motion = gaussian(gal.z,z_view[i], rel_sigma)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
            int_magnitude = Integrate_Schechter_gaussian(z_view[i], gal.app_magnitude, gal.dapp_magnitude, M_min, M_cutoff, m_th, omega, Schechter, M_cutoff)
            I += dz*int_magnitude*prop_motion*CoVolEl
        return log(I)

    else:
        for i in range(n):
            Mth = absM(z_view[i], m_th, omega)
            LD_i = omega.LuminosityDistance(z_view[i])
            int_magnitude = Integrate_Schechter(M_max, M_min, Mth, Schechter, M_cutoff)
            prop_motion = 1./(zmax-zmin)
            CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
            I += dz*int_magnitude*prop_motion*CoVolEl*(4*np.pi)
        return log(I)

cdef double ComputeLogLhNoEmission(Galaxy gal, CosmologicalParameters omega, double zmin, double zmax, double M_cutoff, double m_th = 18, double M_max = -6., double M_min = -23):

    cdef unsigned int i, n = 1000
    cdef double mag_int
    cdef double LD_i

    cdef double I = 0.0
    cdef np.ndarray[double, ndim=1, mode = "c"] z = np.linspace(zmin, zmax, n, dtype = np.float64)
    cdef double[::1] z_view = z

    cdef double dz = (zmax - zmin)/n

    cdef object Schechter
    cdef double alpha, Mstar, CoVol, Mth

    cdef double int_magnitude, prop_motion, CoVolEl

    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, h = omega.h) # Modo semplice per tirare fuori i parametri di Schechter
    CoVol = (omega.ComovingVolume(zmax)-omega.ComovingVolume(zmin))

    for i in range(n):
        Mth = absM(z_view[i], m_th, omega)
        LD_i = omega.LuminosityDistance(z_view[i])
        int_magnitude = Integrate_Schechter_above(M_max, M_min, Mth, Schechter, M_cutoff)
        prop_motion = 1./(zmax-zmin)
        CoVolEl = omega.ComovingVolumeElement(z_view[i])/(CoVol)
        I += dz*int_magnitude*prop_motion*CoVolEl
    return log(I)
