"""
    low-l Planck TT likelihood approximation based on 1608.02487 and
    https://github.com/marius311/lsplitsims/
"""

import numpy as np
from cosmology.cmb import CMB
from cosmology.cosmoparams import Planck2015
from scipy.stats import chi2

from os.path import dirname, abspath
LOCATION = dirname(abspath(__file__))

cmb = CMB(camb_init=False)
cmb.init_camb(aboost=1, lboost=1, LMAX=100)
cmb.cambparams.DoLensing = 1

p15 = Planck2015()

lmin = 2; lmax=29
fl = np.loadtxt(LOCATION+"/commander_dx11d2_mask_temp_n0016_likelihood_v1_f.dat", skiprows=lmin)
fsky = 0.9362
mufac = (2.7255E6)**2.0 # conversion to muK^2

cls_meas_low, cls_err_low = p15.get_Planck_lowL_data(lmin=lmin)
cls_meas_low = cls_meas_low * mufac

chi2s = [chi2((2*l+1)*fsky*fl[l-lmin]) for l in range(lmin, lmax+1)]

def lowllike(x):
    logA, ns, H0, Oc, Ob, tau = x
    
    cmb.cosmology.set_H0(H0)
    cmb.cosmology.set_n(ns)
    cmb.cosmology.set_A((9./25)*np.exp(logA)*1.e-10)
    cmb.cosmology.set_Oc0(Oc)
    cmb.cosmology.set_Ob0(Ob)
    cmb.cosmology.set_tau(tau)
    
    cmb.set_camb_cosmology()
    cmb.get_camb_results()
    
    cls_lowl = cmb.cambTCls[lmin:30]*mufac
    
    lowlkl = np.log([chi2s[l-lmin].pdf(
             (2*l+1)*fsky*fl[l-lmin]*cls_meas_low[l-lmin]/cls_lowl[l-lmin])/cls_lowl[l-lmin] 
             for l in range(lmin, lmax+1)]).sum()
             
    return lowlkl
