#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

filename = 'MDC/h_sigma.txt'

data = np.genfromtxt(filename, names = True)

def func(x, a):
    return a/np.sqrt(x)

sigma0, dsigma0 = curve_fit(func, xdata = data['N_ev'], ydata = data['sigma'], sigma = data['sigma']*0.1)
dsigma0 = np.sqrt(dsigma0[0][0])
app = np.linspace(data['N_ev'].min(), data['N_ev'].max(),1000)

plt.figure(1)
plt.subplot(211)
plt.suptitle('$\\sigma (n) = a/n^{1/2},\ a=%.3f \\pm %.3f$' %(sigma0, dsigma0))
plt.errorbar(data['N_ev'], data['sigma'], yerr = data['sigma']*0.1, marker = '+', ls = '', label = '$Data$')
plt.plot(app, func(app, sigma0), label = '$Best-fit$')
plt.ylabel('$\\sigma (n)$')
plt.legend(loc=0)
plt.subplot(212)
plt.plot(data['N_ev'], (data['sigma']-func(data['N_ev'],sigma0))/(data['sigma']*0.1), marker = '+', ls = '')
plt.ylabel('$Normalized\ Residuals$')
plt.xlabel('$n\ [Events]$')
plt.savefig('sigmavsevents.pdf')
