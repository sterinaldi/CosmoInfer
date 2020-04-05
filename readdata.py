import numpy as np
import sys
import os
from galaxies import *
import lal
from volume_reconstruction import VolumeReconstruction



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

def get_samples(file, names = ['ra','dec','luminosity_distance']):
    filename, ext = splitext(file)
    samples = {}

    if ext == '.json':
        with open(file, 'r') as f:
            data = json.load(f)

        post = np.array(data['posterior_samples']['SEOBNRv4pHM']['samples'])
        keys = data['posterior_samples']['SEOBNRv4pHM']['parameter_names']

        for name in names:
            index  = keys.index(name)
            samples[name] = post[:,index]

        return samples

    if ext == '.hdf5':
        f = h5py.File(file, 'r')
        dati = f[list(f.keys())[2]] # 2 low spin posteriors, 0 high spin posteriors
        h5names = ['right_ascension','declination','luminosity_distance_Mpc']

        for name, nameh5 in zip(names, h5names):
            samples[name] = dati[nameh5]

        return samples

    if ext == '.dat':
        data = np.genfromtxt(file, names = True)
        dat_names = ['ra','dec','distance']

        for name, name_dat in zip(names, dat_names):
            samples[name] = data[dat_names]

        return samples


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

def read_CB_event():
     all_files    = os.listdir(input_folder)
     events_list  = [f for f in all_files if 'event' in f]
     catalog_list = [f for f in all_files if 'catalog' in f]
     events_list.sort()
     catalog_list.sort()
     print(catalog_list)
     events = []

    for ev, cat in zip(events_list, catalog_list):
        catalog_file        = input_folder+"/"+cat
        event_file          = open(input_folder+'/'+ev,"r")
        data                = np.genfromtxt(event_file, names = True)
        events.append(Event_test(N_ev_max, data['dLD'],np.deg2rad(data['dRA']), np.deg2rad(data['dDEC']), data['LD'], np.deg2rad(data['RA']), np.deg2rad(data['DEC']), omega, rel_z_error, catalog_file, catalog_data, n_tot))
        event_file.close()
    return np.array(events)





def read_event(event_class,*args,**kwargs):

    if event_class == "TEST": return read_TEST_event(*args, **kwargs)
    if event_class == "CB": return read_CB_event(*args, **kwargs)
    else:
        print("I do not know the class %s, exiting\n"%event_class)
        exit(-1)

if __name__=="__main__":
    input_folder = '/Users/wdp/repositories/LISA/LISA_BHB/errorbox_data/EMRI_data/EMRI_M1_GAUSS'
    event_number = None
    e = read_event("EMRI",input_folder, event_number)
    print(e)
