"""
cosmology.cosmo class with Planck 2013 and 2015 cosmological parameters
"""

from cosmology import cosmo

class Planck2013(cosmo):
    def set_parameters(self):
        self.name="Planck2013"
        self.Ob0=0.048252; self.Om0=0.30712
        self.H0=67.77
        self.n=0.9611; self.r=0.
        self.sigma8=0.8288
        self.tau=0.0952; self.z_reion=11.52
        self.t0=13.7965; self.Tcmb0=2.7255
        self.Neff=3.046
        self.flat=True
        self.m_nu=[0., 0., 0.06]
        self.f_baryon=self.Ob0/self.Om0
        self.h=self.H0/100.

class Planck2015(cosmo):
    def set_parameters(self):
        self.name="Planck2015"
        self.Ob0=0.048252; self.Om0=0.308
        self.H0=67.8
        self.n=0.968
        self.r=0.
        self.sigma8=0.815
        self.tau=0.066
        self.z_reion=11.52
        self.t0=13.7965
        self.Tcmb0=2.7255
        self.Neff=3.046
        self.flat=True
        self.m_nu=[0., 0., 0.06]
        self.f_baryon=self.Ob0/self.Om0
        self.h=self.H0/100.
