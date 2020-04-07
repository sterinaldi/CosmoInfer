import numpy as np
import sys
import os
from galaxies import *
import lal
from volume_reconstruction.dpgmm.dpgmm import *
from volume_reconstruction.utils.utils import *
import dill as pickle
from scipy.special import logsumexp

def logPosterior(args):
    density,celestial_coordinates = args
    cartesian_vect = celestial_to_cartesian(celestial_coordinates)
    logPs = [np.log(density[0][ind])+prob.logProb(cartesian_vect) for ind,prob in enumerate(density[1])]
    return logsumexp(logPs)+np.log(Jacobian(cartesian_vect))

def gaussian(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

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



class Event_test(object):
    """
    Event class:
    initialise a GW event based on its distance and potential
    galaxy hosts
    """
    def __init__(self,
                 ID,
                 dLD,
                 dRA,
                 dDEC,
                 LD_true,
                 RA_true,
                 DEC_true,
                 omega,
                 rel_z_error  = 0.1,
                 catalog_file = None,
                 catalog_data = None,
                 n_tot        = None):


        if catalog_file is None and catalog_data is None:
            raise SystemExit('No catalog provided')

        self.ID                     = ID
        self.LD                     = LD_true
        self.dLD                    = dLD
        self.z_true                 = RedshiftCalculation(self.LD, omega)
        self.dz                     = self.z_true-RedshiftCalculation(self.LD-self.dLD, omega)
        self.dRA                    = dRA
        self.dDEC                   = dDEC
        self.RA_true                = RA_true
        self.DEC_true               = DEC_true

        '''
        ATTENZIONE: PER CLASSI NON-TEST È NECESSARIO RIPENSARE I BOUNDARIES
        '''

        self.ramin   = RA_true-3*dRA
        self.ramax   = RA_true+3*dRA
        self.decmin  = DEC_true-3*dDEC
        self.decmax  = DEC_true+3*dDEC
        self.zmin    = self.z_true-3*self.dz
        self.zmax    = self.z_true+3*self.dz

        self.potential_galaxy_hosts = read_galaxy_catalog({'RA':[self.ramin, self.ramax], 'DEC':[self.decmin, self.decmax], 'z':[self.zmin, self.zmax]}, rel_z_error = rel_z_error, catalog_data = catalog_data, catalog_file = catalog_file, n_tot = n_tot)
        self.n_hosts                = len(self.potential_galaxy_hosts)
        if n_tot is not None:
            self.n_tot = n_tot
        else:
            self.n_tot = self.n_hosts

    def post_LD(self, LD):
        app = gaussian(LD, self.LD, self.dLD)
        return app

    def post_RA(self, RA):
        app = gaussian(RA, self.RA_true, self.dRA)
        return app

    def post_DEC(self, DEC):
        app = gaussian(DEC, self.DEC_true, self.dDEC)
        return app

class Event_CBC(object):

    def __init__(self,
                 ID,
                 catalog_file,
                 density,
                 levels_file,
                 n_tot        = None,
                 gal_density  = 0.6675): # galaxies/Mpc^3 (from Conselice et al., 2016)

        if catalog_file is None:
            raise SystemExit('No catalog provided')

        self.ID                     = ID
        self.potential_galaxy_hosts = read_galaxy_catalog({'RA':[0., 360.], 'DEC':[-90., 90.], 'z':[0., 4.]}, catalog_file = catalog_file, n_tot = None)
        self.n_hosts                = len(self.potential_galaxy_hosts)
        self.density_model          = pickle.load(open(density, 'rb'))

        self.cl      = np.genfromtxt(levels_file, names = ['CL','vol','area','LD', 'ramin', 'ramax', 'decmin', 'decmax'])
        self.vol_90  = self.cl['vol'][np.where(self.cl['CL']==0.95)[0][0]]-self.cl['vol'][np.where(self.cl['CL']==0.05)[0][0]]
        self.area_90 = self.cl['area'][np.where(self.cl['CL']==0.95)[0][0]]-self.cl['area'][np.where(self.cl['CL']==0.05)[0][0]]
        self.LDmin   = self.cl['LD'][np.where(self.cl['CL']==0.05)[0][0]]
        self.LDmax   = self.cl['LD'][np.where(self.cl['CL']==0.95)[0][0]]
        self.LDmean  = self.cl['LD'][np.where(self.cl['CL']==0.5)[0][0]]
        self.ramin   = self.cl['ramin'][np.where(self.cl['CL']==0.9)[0][0]]
        self.ramax   = self.cl['ramax'][np.where(self.cl['CL']==0.9)[0][0]]
        self.decmin  = self.cl['decmin'][np.where(self.cl['CL']==0.9)[0][0]]
        self.decmax  = self.cl['decmax'][np.where(self.cl['CL']==0.9)[0][0]]

        if n_tot is not None:
            self.n_tot = n_tot
        elif gal_density is not None:
            self.n_tot = gal_density*self.vol_90
            print('Total number of galaxies in the considered volume ({0} Mpc^3): {1}'.format(self.vol_90, self.n_tot))
            self.potential_galaxy_hosts = catalog_weight(self.potential_galaxy_hosts, weight = 'uniform', ngal = self.n_tot)
        else:
            self.n_tot = self.n_hosts

    def logP(self, galaxy):
        '''
        galaxy must be a list with [LD, dec, ra]
        '''
        return logPosterior((self.density_model, np.array(galaxy)))

def read_TEST_event(errors = None, omega = None, input_folder = None, catalog_data = None, N_ev_max = None, rel_z_error = 0.1, n_tot = None):
    '''
    Classe di evento costruita per finalità di test. Le distribuzioni di probabilità sono gaussiane e centrate su una galassia a scelta.
    '''
    all_files    = os.listdir(input_folder)
    events_list  = [f for f in all_files if 'event' in f]
    catalog_list = [f for f in all_files if 'catalog' in f]
    events_list.sort()
    catalog_list.sort()
    print(catalog_list)
    events = []

    if N_ev_max is not None:
        events_list = events_list[N_ev_max:N_ev_max+1:]
        catalog_list = catalog_list[N_ev_max:N_ev_max+1:]

    for ev, cat in zip(events_list, catalog_list):
        catalog_file        = input_folder+"/"+cat
        event_file          = open(input_folder+'/'+ev,"r")
        data                = np.genfromtxt(event_file, names = True)
        events.append(Event_test(N_ev_max, data['dLD'],np.deg2rad(data['dRA']), np.deg2rad(data['dDEC']), data['LD'], np.deg2rad(data['RA']), np.deg2rad(data['DEC']), omega, rel_z_error, catalog_file, catalog_data, n_tot))
        event_file.close()
    return np.array(events)

def read_CBC_event(input_folder, n_tot = None, gal_density = 0.6675):
    all_files     = os.listdir(input_folder)
    print(all_files)
    event_folders = []
    for file in all_files:
        if not '.' in file and 'event' in file:
            event_folders.append(file)
    events = []
    ID = 0.
    for evfold in event_folders:
        ID +=1
        catalog_file = input_folder+evfold+'/galaxy_0.9.txt'
        event_file   = input_folder+evfold+'/dpgmm_density.p'
        levels_file  = input_folder+evfold+'/confidence_levels.txt'
        events.append(Event_CBC(ID, catalog_file, event_file, levels_file))
    return np.array(events)





def read_event(event_class,*args,**kwargs):

    if event_class == "TEST": return read_TEST_event(*args, **kwargs)
    if event_class == "CBC": return read_CBC_event(*args, **kwargs)
    else:
        print("I do not know the class %s, exiting\n"%event_class)
        exit(-1)

if __name__=="__main__":
    input_folder = '/Users/wdp/repositories/LISA/LISA_BHB/errorbox_data/EMRI_data/EMRI_M1_GAUSS'
    event_number = None
    e = read_event("EMRI",input_folder, event_number)
    print(e)
