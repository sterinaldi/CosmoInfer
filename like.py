#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from optparse import OptionParser
from scipy.special import logsumexp
from functools import reduce
from scipy.stats import norm
import unittest
import lal
# import cpnest.model
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
import ray

def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

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

@ray.remote
def calculatelikelihood(args):
        hi = args[0]
        print(hi)
        e  = args[1]
        completeness_file = args[2]
        omega = cs.CosmologicalParameters(hi, 0.3,0.7,-1,0)
        logL = 0.
        #sys.stdout.write('Event %d of %d, h = %.3f, hmax = %.3f\n' % (evcounter, len(events), hi, h.max()))
        logL += lk.logLikelihood_single_event(e.potential_galaxy_hosts, e, omega, 20., Ntot = e.n_tot, completeness_file = completeness_file)
        omega.DestroyCosmologicalParameters()
        return logL


usage=""" %prog (options)"""

if __name__ == '__main__':
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

    ray.init()
    events = readdata.read_event(opts.event_class, input_folder = opts.data, emcp = opts.EMcp, nevmax = opts.nevmax)

    if opts.out == None:
        opts.out = opts.data + 'output/'
        if not os.path.exists(opts.out):
            os.mkdir(opts.out)

    h  = np.linspace(0.67, 0.8, 20, endpoint=False)
    dh = (h.max()-h.min())/len(h)

    evcounter    = 0
    lhs          = []
    lhs_unnormed = []

    # import multiprocessing as mp
    # pool = mp.Pool(4)


    for e in events:
        I = 0.
        likelihood = []
        evcounter += 1
        completeness_file = opts.out+'completeness_fraction_'+str(e.ID)+'.txt'
        f=open(opts.out+'completeness_fraction_'+str(e.ID)+'.txt', 'w')
        f.write('h Nem N M')
        f.close()
        # for hi in h:
        #     omega = cs.CosmologicalParameters(hi, 0.3,0.7,-1,0)
        #     logL = 0.
        #     sys.stdout.write('Event %d of %d, h = %.3f, hmax = %.3f\n' % (evcounter, len(events), hi, h.max()))
        #     logL += lk.logLikelihood_single_event(e.potential_galaxy_hosts, e, omega, 20., Ntot = e.n_tot, completeness_file = opts.out+'completeness_fraction_'+str(e.ID)+'.txt')
        #     omega.DestroyCosmologicalParameters()
        #     likelihood.append(logL)
        args = [(hi, e, completeness_file) for hi in h]
        # results = pool.map(calculatelikelihood, args)

        futures = [calculatelikelihood.remote((hi, e, completeness_file)) for hi in h]

        results = ray.get(futures)

        likelihood = np.array(results)
        lhs_unnormed.append(np.array(likelihood))
        likelihood_app = np.exp(likelihood - likelihood.max())
        for i in range(len(likelihood_app)):
            li = likelihood_app[i]
            I += li*dh
        likelihood = likelihood - np.log(I) - likelihood.max()
        lhs.append(np.array(likelihood))
        np.savetxt(opts.out+'likelihood_'+str(e.ID)+'.txt', np.array([h, likelihood]).T, header = 'h\t\tlogL')

    joint = np.zeros(len(likelihood))
    for like in lhs_unnormed:
        if np.isfinite(like[10]):
            joint += like
    I = 0.
    joint_app = np.exp(joint - joint.max())
    for i in range(len(joint_app)):
        ji = joint_app[i]
        I += ji*dh
    joint = joint - np.log(I) - joint.max()

    percentiles = weighted_quantile(h*100, [0.05, 0.16, 0.50, 0.84, 0.95], sample_weight = np.exp(joint))
    thickness   = [0.4,0.5,1,0.5,0.4]

    styles      = ['dotted', 'dashed', 'solid', 'dashed','dotted']

    hmax = 100*h[np.where(joint == joint.max())]
    results = ' %.0f^{+%.0f}_{-%.0f}' % (hmax, percentiles[3]-hmax, hmax-percentiles[1])

    percentiles[2] = hmax
    title = '$H_0 = '+results+'\ km\\cdot s^{-1}\\cdot Mpc^{-1}$'
    fig = plt.figure()
    fig.suptitle(title)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    for l in lhs:
       ax1.plot(h*100,np.exp(l)/100., linewidth = 0.3)
    ax1.axvline(70, linewidth = 0.5, color = 'r')
    ax2.plot(h*100, np.exp(joint)/100, label ='Joint posterior')
    ax2.legend(loc=0)
    ax2.set_xlabel('$H_0\ [km\\cdot s^{-1}\\cdot Mpc^{-1}]$')
    ax2.set_ylabel('$p(H_0)$')
    ax1.set_ylabel('$p(H_0)$')
    fig.savefig(opts.out+'h_posterior.pdf', bbox_inches='tight')

    fig2 = plt.figure()
    fig2.suptitle(title)
    ax = fig2.add_subplot(111)
    ax.plot(h*100, np.exp(joint)/100.)
    # ax.axvline(70, color = 'r')
    for value, thick, style in  zip(percentiles, thickness, styles):
        ax.axvline(value, ls = style, linewidth = thick, color = 'darkblue')
    #ax.set_xlim(55,80)
    ax.set_xlabel('$H_0\ [km\\cdot s^{-1}\\cdot Mpc^{-1}]$')
    ax.set_ylabel('$p(H_0)$')
    fig2.savefig(opts.out+'h_posterior_tight.pdf', bbox_inches='tight')


    completeness = np.genfromtxt(opts.out+'completeness_fraction_1.0.txt', names = True)
    Nem   = completeness['Nem']
    N     = completeness['N']
    gamma = N/Nem
    h = completeness['h']
    fig2 = plt.figure()
    ax1  = fig2.add_subplot(211)
    ax2  = fig2.add_subplot(212)
    gamma = np.array([x for _,x in sorted(zip(h,gamma))])
    gammamax = gamma[np.where(joint == joint.max())]
    h.sort()

    ax1.plot(h*100, gamma)
    ax1.set_ylabel('$\\gamma(H_0)$')
    ax1.set_xlabel('$H_0$')
    ax2.plot(gamma, np.exp(joint)/100.)
    ax2.axvline(gammamax, ls = '--', color = 'r', label = '$\\gamma = %.2f$'%(gammamax))
    ax2.set_ylabel('$p(\\gamma)$')
    ax2.set_xlabel('$\\gamma = N/N_{tot}$')
    ax2.legend(loc=0)
    fig2.tight_layout()
    fig2.savefig(opts.out+'completeness.pdf', bbox_inches = 'tight')
