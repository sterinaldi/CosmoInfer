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

class catalog(cpnest.model.Model):

    def __init__(self,omega, schechter):
        self.names     = ['B', 'z_cosmo', 'ra', 'dec']
        self.bounds    = [[-23,0], [0.0002,0.2], [0,2*np.pi], [-np.pi,np.pi]]
        self.omega     = omega
        self.schechter = schechter

    def log_prior(self,x):

        logP = super(CosmologicalModel,self).log_prior(x)
        return logP

    def log_likelihood(self,x):

        logL = np.log(lal.ComovingVolumeElement(omega, x['z_cosmo'])))+np.log(schechter(x['B']))
        return logL


if __name__ == '__main__':

    omega = lal.CreateCosmologicalParameters(0.7, 0.3, 0.7, -1, 0, 0)
    Schechter, alpha, Mstar = SchechterMagFunction(-23,0,omega.h)
    output = './mockcatalog'

    C = catalog(omega, Schechter)

    work=cpnest.CPNest(C, verbose = 3, output = output)
    work.run()
    print('log Evidence {0}'.format(work.NS.logZ))
    x = work.posterior_samples.ravel()
    samps = np.column_stack((x['h'],x['om']))
    fig = corner.corner(samps,
           labels= [r'$B$',
                    r'$z_{cosmo}$', r'$ra$', r'$dec$'],
           quantiles=[0.05, 0.5, 0.95],
           show_titles=True, title_kwargs={"fontsize": 12},
           use_math_text=True,
           filename=os.path.join(output,'joint_posterior.pdf'))
           
