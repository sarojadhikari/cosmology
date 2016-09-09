"""
cosmology.cosmo class with Planck 2013 and 2015 cosmological parameters
"""

from cosmology import cosmo

class Planck2013(cosmo):
    def set_parameters(self):
        self.name="Planck2013"
        self.Ob0=0.048252; self.Om0=0.30712; self.Oc0=self.Om0-self.Ob0
        self.H0=67.77; self.sigma8=0.8288
        self.n=0.9611; self.r=0.; self.ns = self.n
        self.As=2.2E-9
        self.tau=0.0952; self.z_reion=11.52; self.zeq = 3365
        self.t0=13.7965; self.Tcmb0=2.7255
        self.Neff=3.046; self.m_nu=[0., 0., 0.06]
        self.flat=True

class Planck2015(cosmo):
    def set_parameters(self):
        """
        the parameters are taken from Planck temperature data combined with
        Planck lensing, taken from the second column [TT+lowP+lensing] of
        Table 4 of Planck 2015 Cosmology paper arXiv:1502.01589
        """
        self.name="Planck2015"
        self.Ob0=0.04841; self.Om0=0.308; self.Oc0=self.Om0-self.Ob0
        self.H0=67.8; self.sigma8 = 0.815
        self.n=0.968; self.r=0.; self.ns = self.n
        self.As=2.139E-9
        self.tau=0.0660; self.z_reion=8.8; self.zeq = 3365
        self.t0=13.799; self.Tcmb0=2.7255
        self.Neff=3.046 # this is the standard model N_eff Planck measures: 3.15\pm0.23
        self.m_nu=[0., 0., 0.06]
        self.flat=True
