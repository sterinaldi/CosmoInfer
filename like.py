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

usage=""" %prog (options)"""

parser = OptionParser(usage)
parser.add_option('-d', '--data',        default=None, type='string', metavar='data', help='Galaxy data location')
parser.add_option('-o', '--out',         default=None, type='string', metavar='out', help='Directory for output')
parser.add_option('-c', '--event-class', default=None, type='string', metavar='event_class', help='Class of the event(s) [MBH, EMRI, sBH]')
parser.add_option('-e', '--event',       default=None, type='int', metavar='event', help='Event number')
parser.add_option('-m', '--model',       default='LambdaCDM', type='string', metavar='model', help='Cosmological model to assume for the analysis (default LambdaCDM). Supports LambdaCDM, CLambdaCDM, LambdaCDMDE, and DE.')
parser.add_option('-j', '--joint',       default=0, type='int', metavar='joint', help='Run a joint analysis for N events, randomly selected (EMRI only).')
parser.add_option('-z', '--zhorizon',    default=1000.0, type='float', metavar='zhorizon', help='Horizon redshift corresponding to the SNR threshold')
parser.add_option('--snr_threshold',     default=0.0, type='float', metavar='snr_threshold', help='SNR detection threshold')
parser.add_option('--em_selection',      default=0, type='int', metavar='em_selection', help='Use EM selection function')
parser.add_option('--reduced_catalog',   default=0, type='int', metavar='reduced_catalog', help='Select randomly only a fraction of the catalog')
parser.add_option('-t', '--threads',     default=None, type='int', metavar='threads', help='Number of threads (default = 1/core)')
parser.add_option('-s', '--seed',        default=0, type='int', metavar='seed', help='Random seed initialisation')
parser.add_option('--nlive',             default=1000, type='int', metavar='nlive', help='Number of live points')
parser.add_option('--poolsize',          default=100, type='int', metavar='poolsize', help='Poolsize for the samplers')
parser.add_option('--maxmcmc',           default=1000, type='int', metavar='maxmcmc', help='Maximum number of mcmc steps')
parser.add_option('--postprocess',       default=0, type='int', metavar='postprocess', help='Run only the postprocessing')
parser.add_option('-n', '--nevmax',      default=None, type='int', metavar='nevmax', help='Maximum number of considered events')
parser.add_option('-u', '--uncert',      default='0.1', type='float', metavar='uncert', help='Relative uncertainty on z of each galaxy (peculiar motion)')
parser.add_option('-a', '--hosts',       default=None, type='int', metavar='hosts', help='Total number of galaxies in considered volume')
parser.add_option('--EMcp',              default=0, type='int', metavar='EMcp', help='Electromagnetic counterpart')
(opts,args)=parser.parse_args()


if opts.event_class == 'TEST' or opts.event_class == 'CBC':
    events = readdata.read_event(opts.event_class, input_folder = opts.data, emcp = opts.EMcp, nevmax = opts.nevmax)
else:
    print('I do not know the class {0}, exit...'.format(opts.event_class))

if opts.out == None:
    opts.out = opts.data + 'output/'
    if not os.path.exists(opts.out):
        os.mkdir(opts.out)

h = np.linspace(0.3,2,100)
dh = (h.max()-h.min())/len(h)
# h = [0.7]
evcounter = 0
lhs = []

for e in events:
    I = 0.
    likelihood = []
    evcounter += 1
    for hi in h:
        omega = cs.CosmologicalParameters(hi, 0.3,0.7,-1,0)
        logL = 0.
        sys.stdout.write('ev {0} of {1}, h = {2}\r'.format(evcounter, len(events), hi))
        logL += lk.logLikelihood_single_event(e.potential_galaxy_hosts, e, omega, 18., Ntot = e.n_tot, completeness_file = opts.out+'completeness_fraction_'+str(e.ID)+'.txt')
        omega.DestroyCosmologicalParameters()
        likelihood.append(logL)

    likelihood = np.array(likelihood)
    likelihood_app = np.exp(likelihood - likelihood.max())
    for li in likelihood_app:
        I += li*dh
    likelihood = likelihood - np.log(I) - likelihood.max()
    lhs.append(np.array(likelihood))
    np.savetxt(opts.out+'likelihood_'+str(e.ID)+'.txt', np.array([h, likelihood]).T, header = 'h\t\tlogL')

joint = np.zeros(len(likelihood))
for like in lhs:
    if np.isfinite(like[10]):
        joint += like

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for l in lhs:
    ax1.plot(h*100,np.exp(l), linewidth = 0.3)
ax1.axvline(70, linewidth = 0.5, color = 'r')
ax2.plot(h*100, np.exp(joint), label ='Joint posterior')
ax2.axvline(70, color = 'r')
ax2.legend(loc=0)
ax2.set_xlabel('$H_0\ [km\\cdot s^{-1}\\cdot Mpc^{-1}]$')
ax2.set_ylabel('$p(H_0)$')
ax1.set_ylabel('$p(H_0)$')
fig.savefig(opts.out+'h_posterior.pdf', bbox_inches='tight')
# plt.show()
