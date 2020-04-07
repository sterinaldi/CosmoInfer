#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pickle
from galaxy import Galaxy

def read_galaxy_catalog(limits, catalog_file = None, n_tot = None):
    '''
    The catalog can be passed either as a path or, if precedently loaded, as np.array.
    In case both data and path are provided, already loaded data are used.

    GLADE flag description:

    flag1:  Q: the source is from the SDSS-DR12 QSO catalog
            C: the source is a globular cluster
            G: the source is from another catalog and not identified as a globular cluster

    flag2:  0: no z nor LD
            1: measured z
            2: measured LD
            3: measured spectroscopic z

    flag3:  0: velocity field correction has not been applied to the object
            1: we have subtracted the radial velocity of the object
    '''

    if catalog_data is None and catalog_file is None:
        raise SystemExit('No catalog data nor file provided.')

    if catalog_data is not None and catalog_file is not None:
        print('Both data and path provided. Loaded data will be used.')

    if catalog_file is not None and catalog_data is None:
        glade_names = "PGCname, GWGCname, HyperLedaname, 2MASSname, SDSS-DR12name,\
                    flag1, RA, DEC, dist, dist_err, z, B, B_err, B_abs, J, J_err,\
                    H, H_err, K, K_err, flag2, flag3"
        catalog_data = np.atleast_1d(np.genfromtxt(catalog_file, names=True)) # Troubles with single row files.

    catalog = []

    for i in range(catalog_data.shape[0]):
        # Check the entries: B-band mag (abs and apparent), redshift and proximity to GW position posteriors
        if isinbound(catalog_data[i], limits):
            if catalog_data['pec.mot.corr']==1.:
                z_error = 0.01 # errore dovuto alla misura
            else
                z_error = 0.1  # assumo un 10% conservativo oltre i 300 Mpc
            catalog.append(Galaxy(i, np.deg2rad(catalog_data['ra'][i]), np.deg2rad(catalog_data['dec'][i]), catalog_data['z'][i], True, z_error = z_error, app_magnitude = catalog_data['B'][i], dapp_magnitude = catalog_data['B_err'][i] , abs_magnitude = catalog_data['B_abs'][i])) # Controlla nomi con catalogo!
            # Warning: GLADE stores no information on dz. 2B corrected.

    catalog = catalog_weight(catalog, ngal = n_tot) # Implementare meglio la selezione del peso delle galassie.

    return catalog

def isinbound(galaxy, limits):
    if (limits['RA'][0] <= np.deg2rad(galaxy['ra']) <= limits['RA'][1]) and (limits['DEC'][0] <= np.deg2rad(galaxy['dec']) <= limits['DEC'][1]) and (limits['z'][0] <= galaxy['z'] <= limits['z'][1]) :
        return True
    return False


def catalog_weight(catalog, weight = 'uniform', ngal = None):
    '''
    Method:
    Assign a weight for each galaxy in catalog according to the emission probability
    of the galaxy.
    Please note that this has to be a relative probability rather than an absolute one.

        - Uniform weighting: 1/N ('uniform')
    '''
    if weight == 'uniform':
        for galaxy in catalog:
            if ngal is None:
                galaxy.weight = 1./len(catalog)
            else:
                galaxy.weight = 1./ngal

    return catalog
