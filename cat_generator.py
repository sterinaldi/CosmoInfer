#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from optparse import OptionParser
from scipy.special import logsumexp
from functools import reduce
from scipy.stats import norm
import unittest
import lal
import cpnest.model
import sys
import os
import readdata
import matplotlib
import corner
import itertools as it
import cosmology as cs
import numpy as np
import likelihood as lk
import matplotlib.pyplot as plt
from displaypost import plot_post
import math
from schechter import *
import random as rd
import lal
import os
import sys

def appM(z, M, omega):
    return M + 5.0*np.log10(1e5*lal.LuminosityDistance(omega,z)) - 5.*np.log10(omega.h)

def gaussian(x, x0, sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))


if __name__ == '__main__':

    n_ev = 25
    omega = lal.CreateCosmologicalParameters(0.7, 0.3, 0.7, -1, 0, 0)
    M_max    = -4.
    M_min    = -23.
    M_mean = -15.
    sigma  = 0.5
    Schechter, alpha, Mstar = SchechterMagFunction(M_min, M_max, omega.h)
    output = 'upsidedown3/'
    if not os.path.exists(output):
        os.mkdir(output)
    numberdensity = 0.066

    z_min = 0.002
    z_max = 0.04
    dCoVolMax = lal.ComovingVolumeElement(z_max,omega)
    pM_max    = Schechter(M_max)
    CoVol = lal.ComovingVolume(omega, z_max) - lal.ComovingVolume(omega, z_min)
    ev_density = n_ev/CoVol
    np.savetxt(output+'evdensity.txt', np.array([ev_density]).T, header = 'evdensity')
    N_tot = int(CoVol*numberdensity)

    ID      = []
    ra      = []
    dec     = []
    z_cosmo = []
    z       = []
    appB    = []
    absB    = []
    dB      = []
    DL      = []
    host    = []

    ID_h      = []
    ra_h      = []
    dec_h     = []
    z_cosmo_h = []
    z_h       = []
    appB_h    = []
    absB_h    = []
    dB_h      = []
    DL_h      = []
    host_h    = []


    for i in range(N_tot):

        sys.stdout.write('{0} out of {1}\r'.format(i+1, N_tot))
        sys.stdout.flush()
        ID.append(i)
        ra.append(rd.uniform(0,2*np.pi))
        dec.append(rd.uniform(-np.pi/2.,np.pi/2.))
        while 1:
            z_temp = rd.uniform(z_min,z_max)
            if rd.random()*dCoVolMax < lal.ComovingVolumeElement(z_temp,omega):
                z_c = z_temp
                z_cosmo.append(z_c)
                break
        z_pec = rd.gauss(0, 0.001)
        z.append(z_c+z_pec)
        DL.append(lal.LuminosityDistance(omega,z_c))
        while 1:
            B_temp = rd.uniform(M_min, M_max)
            if rd.random()*pM_max < Schechter(B_temp):
                B = B_temp
                absB.append(B)
                break
        dB.append(0.5)
        appB.append(appM(z_c, B, omega))
        host.append(0)

    for i in range(n_ev):
        index = rd.randint(0,N_tot-1)
        new_B = rd.gauss(M_mean,sigma)
        new_Bapp = appM(z_cosmo[index], new_B, omega)
        host[index] = 1
        absB[index] = new_B
        appB[index] = new_Bapp
        ID_h.append(ID[index])
        ra_h.append(ra[index])
        dec_h.append(dec[index])
        z_cosmo_h.append(z_cosmo[index])
        z_h.append(z[index])
        appB_h.append(appB[index])
        absB_h.append(absB[index])
        dB_h.append(dB[index])
        DL_h.append(DL[index])
        host_h.append(host[index])

    header = 'ID\tra\t\tdec\t\tz\t\tz_cosmo\t\tDL\t\tB_abs\t\tB\t\tB_err\t\thost'
    fmt = '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d'
    np.savetxt(output+'mockcatalog.txt', np.array([ID, ra, dec, z, z_cosmo, DL, absB, appB, dB, host]).T, fmt = fmt, header = header)
    np.savetxt(output+'hosts.txt', np.array([ID_h, ra_h, dec_h, z_h, z_cosmo_h, DL_h, absB_h, appB_h, dB_h, host_h]).T, fmt = fmt, header = header)

    fig_z_cosmo = plt.figure()
    fig_z_pm    = plt.figure()
    fig_M       = plt.figure()
    fig_M_hosts = plt.figure()

    ax_z_cosmo  = fig_z_cosmo.add_subplot(111)
    ax_z_pm     = fig_z_pm.add_subplot(111)
    ax_M        = fig_M.add_subplot(111)
    ax_M_hosts  = fig_M_hosts.add_subplot(111)


    app_z = np.linspace(z_min, z_max, 1000)
    app_CoVol = []


    for zi in app_z:
        app_CoVol.append(lal.ComovingVolumeElement(zi, omega)/CoVol)


    app_z_pm    = np.linspace(-5*0.001, 5*0.001, 1000)
    app_M       = np.linspace(M_min, M_max, 1000)
    app_M_hosts = np.linspace(M_mean-4*sigma, M_mean+4*sigma, 1000)
    app_pM      = []

    for Mi in app_M:
        ratio = float(n_ev)/float(N_tot)
        app_pM.append((1-ratio)*Schechter(Mi)+ratio*gaussian(Mi, M_mean, sigma))

    ax_z_cosmo.hist(z_cosmo, bins = int(np.sqrt(len(z_cosmo))), density = True)
    ax_z_cosmo.plot(app_z, app_CoVol)
    ax_z_cosmo.set_xlabel('$z_{cosmo}$')
    ax_z_cosmo.set_ylabel('$p(z_{cosmo})$')
    fig_z_cosmo.savefig(output+'z_cosmo.pdf', bbox_inches='tight')

    ax_z_pm.hist(np.array(z)-np.array(z_cosmo), bins = int(np.sqrt(len(z_cosmo))), density = True)
    ax_z_pm.plot(app_z_pm, gaussian(app_z_pm, 0, 0.001))
    ax_z_pm.set_xlabel('$z_{pm}$')
    ax_z_pm.set_ylabel('$p(z_{pm})$')
    fig_z_pm.savefig(output+'z_pm.pdf', bbox_inches='tight')

    ax_M.hist(absB, bins = int(np.sqrt(len(absB))), density = True)
    ax_M.plot(app_M, app_pM)
    ax_M.set_xlabel('$M\ (B\ band)$')
    ax_M.set_ylabel('$p(M)$')
    fig_M.savefig(output+'M.pdf', bbox_inches='tight')

    ax_M_hosts.hist(absB_h, bins = int(np.sqrt(len(absB_h))), density = True, color = 'lightblue', label = '$M$')
    ax_M_hosts.plot(app_M_hosts, gaussian(app_M_hosts, M_mean, sigma), color = 'red', linewidth = 0.5, label = '$\\propto exp(-(M-\\mu)^2/2\\sigma^2)$')
    ax_M_hosts.set_xlabel('$M\ (B\ band, hosts)$')
    ax_M_hosts.set_ylabel('$p(M)$')
    ax_M_hosts.legend(loc=0)
    fig_M_hosts.savefig(output+'M_hosts.pdf', bbox_inches='tight')
