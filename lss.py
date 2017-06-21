"""
LSS computations from pycamb/EH transfer function --- needs the cosmology class
"""
from os.path import isfile, dirname, abspath
import numpy as np
from cosmology.cosmoparams import Planck2015
from scipy import integrate
import pkg_resources
from astropy.io import fits

from scipy.special import spherical_jn


default_cosmo = Planck2015()
LOCATION = dirname(abspath(__file__))

class LSS(object):

    def __init__(self, cosmology=default_cosmo, camb_init=False):
        self.cosmology = cosmology
        
    def dndz(self, z, z0=0.7, nbar=1E9):
        return 1.5*z*z*np.exp(-np.power(z/z0, -1.5))
    
    def angular_power_spectrum(self, z=0.7, lmax=191):
        klist = np.arange(5E-5, 0.5, 5E-5)
        Plist = self.cosmology.power_spectrumz(klist, z=z)
            # power_spectrumz uses the EH transfer function
        dndz = self.dndz(z=z)
        rz = self.cosmology.comoving_distance(z)
        
        Cllist = np.zeros(lmax+1)
        
        for l in range(2, lmax+1):
            Cllist[l] = np.power(dndz, 2.0)*integrate.trapz(
                klist*klist*Plist*np.power(spherical_jn(l, klist*rz),2.0), klist)
        
        return Cllist
