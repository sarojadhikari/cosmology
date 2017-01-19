"""
CMB computations from pycamb --- needs the cosmology class
"""
from os.path import isfile
import numpy as np
from cosmology.cosmoparams import Planck2015
import camb
default_cosmo = Planck2015()

class CMB(object):

    def __init__(self, cosmology=default_cosmo, camb_init=True):
        """
        """
        self.cosmology = cosmology
        self.klist = []
        self.glk = []
        if (camb_init):
            self.init_camb()

    def init_camb(self, aboost=4, LMAX=3500):
        self.set_camb_parameters(aboost=aboost, LMAX=LMAX)
        #self.get_nonlin_power()
        self.camb_init = True

    def set_camb_parameters(self, LMAX=2000, Omk=0.0, aboost=1, metak=0.):
        """
        """
        self.camblmax = LMAX
        self.cambparams = camb.CAMBparams()
        self.camb = camb.camb
        self.cambtransferparams = camb.model.TransferParams()
        self.cambparams.set_cosmology(H0=self.cosmology.H0,
                                      ombh2=self.cosmology.Ob0*self.cosmology.h**2.0,
                                      omch2=self.cosmology.Oc0*self.cosmology.h**2.0,
                                      omk=Omk,
                                      tau=self.cosmology.tau,
                                      mnu=self.cosmology.m_nu[-1])
        self.cambparams.set_dark_energy()
        self.cambparams.InitPower.set_params(As=self.cosmology.As,
                                             ns=self.cosmology.n,
                                             pivot_scalar=self.cosmology.k0,
                                             r=self.cosmology.r)
        if (metak>0.):
            self.cambparams.set_for_lmax(LMAX, max_eta_k=22000.)
        else:
            self.cambparams.set_for_lmax(LMAX)
        self.cambtransferparams.high_precision = 1 # set high precison to True
        self.cambparams.set_accuracy(AccuracyBoost=aboost,
                                     lSampleBoost=50)
        self.camb_aboost = aboost

    def init_camb_transfer(self, SAVE=True):
        """
        do not call this directly--call get_cmb_transfer_l() which looks for a
        saved file and calls this function only if there is no saved file
        """
        if not(self.camb_transfer_init):
            """
            get the transfer data if it is the first time or if the specified
            accuracy aboost is greater than the one that is saved
            """
            if not(self.camb_init):
                self.init_camb()

            self.cambdata = camb.get_transfer_functions(self.cambparams)
            self.cambtransfer = self.cambdata.get_cmb_transfer_data()
            self.camb_transfer_init = True
            # save the current transfer data
            if (SAVE):
                fname = "glk_"+self.cosmology.name+"_"+str(self.camb_aboost)+"_"+str(self.camblmax)+".npy"
                np.save(fname, np.array([self.cambtransfer.q, (5./3)*self.cambtransfer.delta_p_l_k]))
            self.klist = self.cambtransfer.q[0:-15]
            self.glk = (5./3.)*self.cambtransfer.delta_p_l_k[:,:,0:-15]
            """
            the factor of (5./3) is necessary as the code uses Bardeen potential but the glk_data
            returned assumes curvature perturbations

            one could have rather done the translation of Phi to zeta later when computing alms or Cls
            -- but it is already done here so that there is no need to keep track of
            (3./5) factors anywhere else

            That this factor is necessary can be checked by tallying the results from
            get_camb_results and get_Cls_from_glk
            """


    def init_camb_tensor_transfer(self):
        """
        """
        if not(self.camb_tensor_transfer_init):
            if not(self.camb_init):
                self.init_camb()

    def get_camb_results(self):
        self.cambresults = camb.get_results(self.cambparams)
        self.totalCl = self.cambresults.get_cmb_power_spectra(self.cambparams)['total']
        return self.cambresults

    def get_Cls_from_glk(self, TEB=0, LMAX=100):
        """
        this will work as a check for the transfer function glk normalization

        C_l = 4 pi int_0^infty dlnk glk^2 A_phi (k/k0)^{ns-1}
        """
        Cls=[0., 0.]

        for l in range(2, LMAX+1):
            integrand = 4.*np.pi*np.power(self.get_cmb_transfer_l(TEB, l), 2.0)*(
                        self.primordial_power(self.A, self.klist, self.k0)/self.klist)
            Cl = integrate.trapz(integrand, self.klist)

            Cls.append(Cl)

        return np.array(Cls)

    def get_cmb_transfer_l(self, TEB=0, l=2):
        """
        """
        # check if klist and glk are already loaded
        if (len(self.klist) > 0 and len(self.glk) > 0):
            if (TEB == 0):
                return self.glk[0, l - 2, :]
            else:
                return l * l * self.glk[TEB, l - 2, :]
        else:
            # first check if there is a file saved for the current AccuracyBoost and LMAX
            fname = "glk_" + self.cosmology.name + "_" + str(self.camb_aboost) + "_" + str(self.camblmax) + ".npy"
            if (isfile(fname)):
                print("cmb transfer saved file found!")
                self.klist, self.glk = np.load(fname)
                self.klist = self.klist[self.klist < 0.515]
                # for the glk_*_4_3500.npy, k>~0.515 are sparse (large dk) and
                # produce unwanted oscillations in alhpa(r)
                self.glk = self.glk[:, :, 0:len(self.klist)]
                if (TEB == 0):
                    return self.glk[0, l - 2, :]
                else:
                    return l * l * self.glk[TEB, l - 2, :]
            else:
                # try if the file is in ./datafiles/
                if isfile("datafiles/" + fname):
                    print("cmb transfer saved file found!")
                    self.klist, self.glk = np.load("datafiles/" + fname)
                    self.klist = self.klist[self.klist < 0.515]
                    self.glk = self.glk[:, :, 0:len(self.klist)]
                    if (TEB == 0):
                        return self.glk[0, l - 2, :]
                    else:
                        return l * l * self.glk[TEB, l - 2, :]

                print("cmb transfer file not found...generating")
                self.init_camb_transfer()
                if (TEB == 0):
                    return self.glk[0, l - 2, :]
                else:
                    return l * l * self.glk[TEB, l - 2, :]