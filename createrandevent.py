import os
import numpy as np
import random as rd
import lal
from optparse import OptionParser

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
parser.add_option('-p', '--data_path', default='./', type='string', metavar='data', help='Events destination folder')
parser.add_option('-n', '--n_ev',      default=1, type='int', metavar='n', help='Number of events')
parser.add_option('-e', '--emcp',      default=0, type='int', metavar='emcp', help='Electromagnetic counterpart')
parser.add_option('-s', '--sigmald',   default=0.1, type='float', metavar='sigmald', help='Relative uncertainty on GW LD')

(opts,args)=parser.parse_args()
path = opts.data_path
omega = lal.CreateCosmologicalParameters(0.7,0.3,0.7,-1.,0.,0.)

N_events = opts.n_ev

for i in range(N_events):
    z_gal    = rd.uniform(0.005, 0.03)
    RA_gal   = rd.uniform(0,360)
    DEC_gal  = rd.uniform(-90,90)
    LD_gal   = lal.LuminosityDistance(omega,z_gal)
    dLD_gal  = LD_gal*opts.sigmald

    if opts.emcp:
        file_cat = open(path+'catalog_'+str(i)+'.txt', 'w')
        file_cat.write('null null null 18294475-4245118 null G '+str(RA_gal)+' '+str(DEC_gal)+' '+str(LD_gal)+' null '+str(z_gal)+' 13.284 null -21.558 10.708 0.024 10.056 0.030 9.766 0.041 3 1')
        file_cat.close()
    else:
        print('Remember to provide a galaxy catalog for event {0}'.format(i))


    LD  = rd.uniform(LD_gal-2*dLD_gal, LD_gal+2*dLD_gal)
    dLD = LD*opts.sigmald

    if not opts.emcp:
        dRA = 5.
        dDEC= 5.
        RA  = rd.uniform(RA_gal-2*dRA, RA_gal+2*dRA)
        DEC = rd.uniform(DEC_gal-2*dDEC, DEC_gal+2*dDEC)


    if opts.emcp:
        dRA  = 0.1
        dDEC = 0.1
        RA  = RA_gal
        DEC = DEC_gal


    file_event = open(path+'event_'+str(i)+'.txt', 'w')
    file_event.write('LD dLD RA dRA DEC dDEC\n')
    tobeprinted = str(LD)+' '+str(dLD)+' '+str(RA)+' '+str(dRA)+' '+str(DEC)+' '+str(dDEC)
    file_event.write(tobeprinted)
    file_event.close()
