#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random as rd
import lal
import os
import sys

def LumDist(z, omega):
    return 3e3*(z + (1-omega.om +omega.ol)*z**2/2.)/omega.h

def dLumDist(z, omega):
    return 3e3*(1+(1-omega.om+omega.ol)*z)/omega.h

def RedshiftCalculation(LD, omega, zinit=0.3, limit = 0.001):
    '''
    Redshift given a certain luminosity, calculated by recursion.
    Limit is the less significative digit.
    '''
    LD_test = LumDist(zinit, omega)
    if abs(LD-LD_test) < limit :
        return zinit
    znew = zinit - (LD_test - LD)/dLumDist(zinit,omega)
    return RedshiftCalculation(LD, omega, zinit = znew)

file_hosts = './mockcatalog_H150/hosts.txt'

hosts = np.genfromtxt(file_hosts, names = True)
omega = lal.CreateCosmologicalParameters(1.5,0.3,0.7,-1,0,0)

omegamin = lal.CreateCosmologicalParameters(0.3,0.3,0.7,-1,0,0)
omegamax = lal.CreateCosmologicalParameters(2,0.3,0.7,-1,0,0)


m_th  = 18.

counter = 1

full_catalog = np.genfromtxt('./mockcatalog_H150/mockcatalog.txt', names = True)

for gal in hosts:
    sys.stdout.write('Event {0} of {1}\r'.format(counter, len(hosts)))
    folder = './mockcatalog_H150/event_'+str(counter)+'/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    fmt = '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d'
    np.savetxt(folder+'host.txt', np.column_stack(list(gal)), fmt=fmt, header = '\t\t'.join(hosts.dtype.names))
    dra_w  = 0.01
    ddec_w = 0.01
    ra_w   = rd.gauss(gal['ra'], dra_w)
    dec_w  = rd.gauss(gal['dec'], ddec_w)
    dLD_w  = rd.uniform(0.1,0.3)*gal['DL']
    LD_w   = rd.gauss(gal['DL'], dLD_w)

    np.savetxt(folder+'posterior.txt', np.column_stack([ra_w, dra_w, dec_w, ddec_w, LD_w, dLD_w]), fmt = '%f\t%f\t%f\t%f\t%f\t%f', header = 'ra\tdra\tdec\tddec\tLD\tdLD')

    LD_max  = LD_w + 2*dLD_w
    LD_min  = LD_w - 2*dLD_w
    ra_min  = ra_w - 2*dra_w
    ra_max  = ra_w + 2*dra_w
    dec_min = dec_w - 2*ddec_w
    dec_max = dec_w + 2*ddec_w
    area = np.pi*dra_w**2
    volume = (4./3.)*np.pi*(LD_max**3-LD_min**3)*(area/(4*np.pi))

    zmin = RedshiftCalculation(LD_min, omega)
    zmax = RedshiftCalculation(LD_max, omega)

    np.savetxt(folder+'confidence_region.txt', np.column_stack([ra_min,ra_max,dec_min,dec_max,LD_min,LD_max, zmin, zmax, area*360*180,volume]), fmt = '%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f\t\t%f', header = 'ra_min\tra_max\tdec_min\tdec_max\tLD_min\tLD_max\tz_min\tz_max\tarea\tvolume')

    ID      = []
    ra      = []
    ra_rad  = []
    dec     = []
    dec_rad = []
    z_cosmo = []
    z       = []
    appB    = []
    absB    = []
    dB      = []
    LD      = []
    host    = []

    for pot_host in full_catalog:
        if (zmin < pot_host['z'] < zmax) and (np.sqrt((pot_host['ra']-ra_w)**2+(pot_host['dec']-dec_w)**2) < 2*dra_w) and (pot_host['B']<m_th):
            ID.append(pot_host['ID'])
            ra.append(np.rad2deg(pot_host['ra']))
            ra_rad.append(pot_host['ra'])
            dec.append(np.rad2deg(pot_host['dec']))
            dec_rad.append(pot_host['dec'])
            z_cosmo.append(pot_host['z_cosmo'])
            z.append(pot_host['z'])
            appB.append(pot_host['B'])
            absB.append(pot_host['B_abs'])
            dB.append(pot_host['B_err'])
            LD.append(pot_host['DL'])
            host.append(pot_host['host'])

    header = 'ID\tra\t\tra_rad\t\tdec\t\tdec_rad\t\tz\t\tz_cosmo\t\tLD\t\tB_abs\t\tB\t\tB_err\t\thost'
    fmt = '%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d'
    np.savetxt(folder+'galaxy_0.9.txt', np.array([ID, ra, ra_rad, dec, dec_rad, z, z_cosmo, LD, absB, appB, dB, host]).T, fmt = fmt, header = header)
    counter += 1
