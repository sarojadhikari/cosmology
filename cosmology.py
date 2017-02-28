"""
.. module:: cosmology
   :synopsis: cosmology definitions and basic computations

.. moduleauthor:: Saroj Adhikari <adh.saroj@gmail.com>

"""

import numpy as np
from scipy import integrate

class cosmo(object):
    """
    define a cosmology and provide methods to compute basic cosmological
    quantities using the currently set cosmological parameters
    """
    def __init__(self):
        self.set_parameters()
        self.A = self.As*9./25.
        self.h=self.H0/100.
        #self.k0=0.05/self.h # if using h/Mpc as the units
        self.k0=0.05 # if using 1/Mpc as the units (CAMB)
        self.f_baryon=self.Ob0/self.Om0
        self.gf0=self.growth_factor(0.)
        self.klist = []

        ## some other useful derived variables
        self.omhh = self.Om0*self.h**2.0
        self.obhh = self.omhh * self.f_baryon

    def set_parameters(self):
        self.name = "default"   # default means 2013 here
        self.Ob0=0.048252; self.Om0=0.30712; self.Oc0=self.Om0-self.Ob0
        self.H0=67.77; self.sigma8=0.8288
        self.n=0.9611; self.r=0.; self.ns = self.n
        self.As=2.2E-9; self.alphafac = 1.
        self.tau=0.0952; self.t0=13.7965
        self.z_reion=11.52; self.Tcmb0=2.7255
        self.Neff=3.046; self.m_nu=[0., 0., 0.06]
        self.flat = True

    def set_r(self, r):
        self.r=r

    def set_sigma8(self, s8):
        self.sigma8=s8
        self.normalize()

    def set_Ob0(self, Ob):
        self.Ob0=Ob

    def set_Om0(self, Om):
        self.Om0=Om

    def set_n(self, ns):
        self.n=ns
        self.ns=ns

    def set_h(self, h):
        self.h=h

    def set_tau(self, tau):
        self.tau=tau

    def rhom(self):
        """
        return the matter density for the current cosmology
        """
        mpc_to_cm=3.0856e24
        crit_dens=1.8791e-29*self.h*self.h*pow(mpc_to_cm,3.0) # in grams Mpc^{-3}
        M_sun=1.989e33 # in grams
        return crit_dens*self.Om0/(M_sun/self.h) # in M_sun/h Mpc^{-3}

    def pps(self, k):
        """
        should use self.A (A_Phi, self.As is A_zeta);
        the old version of glk_data files use As convention
        if there is discrepency in Cls computation CHECK that
        the CONVENTION are consistently followed!
        """
        return self.primordial_power(self.A, k, self.k0)

    def primordial_power(self, A, k, k0):
        """
        return the dimensionless primordial power spectrum value at a wave number k,
        given amplitude A and the current cosmology, using pivot wavenumber k0

        .. math::

            \\mathcal{P}(k) = A \\left( \\frac{k}{k_0} \\right)^{n_s-1}

        """
        return  A*(k/k0)**(self.n-1)

    def power_spectrumz(self, k, z=0):
        """
        returns the matter power spectrum at wave number k at a redshift z
        """
        return self.power_spectrum0(self.A, k)*np.power(self.gfratio(z), 2.0)

    def power_spectrum0(self, A, k):
        """
        returns the matter power spectrum value at wave number k, given A at z=0
        """
        if A==0.:
            A=self.A
        return A*np.power(self.alpha(k,z=0), 2.0)*2.*np.power(np.pi, 2.0)*np.power(k/self.k0, self.n-1.0)/np.power(k, 3.0) # alternatively one can directly implement alpha(k,z=0) here are cancel some powers of k

    def power_spectrum_bbks(self, A, k):
        """
        return the matter power spectrum using bbks transfer function
        """
        return A*(2.*np.pi**2.0)*k**self.n * (2998./self.h)**(3.+self.n)*(self.transfer_function_bbks(k)*self.growth_factor(0.0))**2.0

    def growth_rate_f(self, z, factor=1.0001):
        """
        return the growth rate f = dlnD/dlna at the given redshift
        the code limits the smallest z input
        """
        if (z<=10*(factor-1)):
            z=10*(factor-1)
        a = 1.0/(1.0+z)
        aplus = factor*a
        zplus = 1/aplus - 1
        Dplus = np.log(self.growth_factor(zplus))
        D0 = np.log(self.growth_factor(z))
        return (Dplus - D0)/(np.log(aplus)-np.log(a))

    def sigma_sq_integrand(self, k, R):
        return k*k/(2.0*np.pi**2.0)*self.power_spectrum0(self.A, k)*top_hat(k,R)**2.0

    def RtoM(self, R):
        """
        convert R in :math:`h^{-1}` Mpc to the corresponding M in :math:`h^{-1}M_\odot`
        """
        return 4.0*self.rhom()*np.pi*R*R*R/3.0

    def MtoR(self, M):
        """
        convert M in M_sun/h to the corresponding R in Mpc
        """
        return (3.0*M/4.0/self.rhom()/np.pi)**(1.0/3.0)

    def sigmaM(self, M):
        """
        """
        R=self.MtoR(M)
        return self.sigmaR(R)

    def sigmaM_M(self, M):
        """
        return the derivative sigmaM,M
        """
        plus_value=self.sigmaM(M*1.01)
        minus_value=self.sigmaM(M*0.99)
        finite_diff=plus_value-minus_value
        delta_M=0.02*M
        return (finite_diff/delta_M)

    def sigmaR(self, R):
        """
        compute sigma_R by integrating;
        now use the CAMB power spectrum
        """
        #fac = 1./(2.*np.power(np.pi, 2.0))
        from utilities.functions import top_hat
        integrand = lambda q: np.power(top_hat(q, R)*self.alpha(q), 2.0)*self.primordial_power(self.A, q, self.k0)/q
        #integrand = lambda q: q*q*np.power(top_hat(q, R), 2.0)*self.camb_power_lin(q)
        results = integrate.quad(integrand, 0.0, 20./R, limit=80)
        return np.sqrt(results[0])

    def xi(self, r=0., z1=0.0, R1=8., z2=0.0, R2=8.):
        """return the smoothed two-point correlation function (of two subvolumes of size R1 and R2), r apart
        """
        from utilities.functions import BesselJ, top_hat
        fac = self.gfratio(z1)*self.gfratio(z2)/(2.*np.power(np.pi, 2.0))
        integrand = lambda q: q*q*top_hat(q, R1)*top_hat(q, R2)*BesselJ(0, q*r)*self.power_spectrumz(q, z=0)
        results = integrate.quad(integrand, 0.0, 20./min(R1, R2))
        return fac*results[0]

    def xi_cube(self, r=0., z1=0.0, R1=8., z2=0.0, R2=8.0):
        """return the smoothed two-point correlation function for the case of
        a cubic volume
        """
        #integrand = lambda qx, qy, qz: cubic_top_hat(2*R1, qx, qy, qz)*cubic_top_hat(2*R2, qx, qy, qz)*self.power_spectrumz(np.sqrt(qx*qx+qy*qy+qz*qz), z=0.)*BesselJ(0, np.sqrt(qx*qx+qy*qy))


    def normalize(self):
        """
        nomrmalize the amplitude of primordial fluctuations A so as to produce
        the sigma8 of the current cosmology

        * A  : amplitude for the Bardeen potential :math:`\Phi` and

        * As : amplitude for the scalar curvature perturbation :math:`\zeta`
        require As to be the input and calculate multiplicative factor for
        alpha(k)
        """
        self.A = self.As*(3./5)**2.0
        s8computed = self.sigmaR(8.)
        self.alphafac = np.sqrt(self.sigma8/s8computed)

    def alpha(self, k,z=0):
        """
        return the product of transfer function and the growth factor at a wavenumber k and redshift
        z with other appropriate factors; alpha relates the primordial gravitational potential to the overdensity
        """
        c=299792.458 # speed of light in km/s
        if (z==0):
            return self.alphafac*2.0*k*k*self.transfer_function(k)*self.gf0*c*c/(3.0*self.Om0*np.power(self.H0/self.h, 2.0))
        else:
            return self.alphafac*2.0*k*k*self.transfer_function(k)*self.growth_factor(z)*(c*c/(3.0*self.Om0*np.power(self.H0/self.h, 2.0)))

    def growth_factor_integrand(self, z):
        """
        return the growth factor integrand
        """
        hubblez=100*self.h*np.sqrt(self.Om0*pow(1+z,3.0)+(1-self.Om0))
        return (1+z)/pow(hubblez/(100.0*self.h),3.0)

    def gfratio(self, z):
        """
        return the ratio of the growth factor at redshift z to that
        of the growth factor at redshift 0
        """
        return self.growth_factor(z)/self.gf0

    def growth_factor(self, z):
        """
        return the growth factor D(z)
        """
        result=integrate.quad(self.growth_factor_integrand, z, 1000) # therefore only valid at z<<1000.
        hubblez=100*self.h*np.sqrt(self.Om0*pow(1+z,3.0)+(1-self.Om0))
        gf = (5.0*self.Om0*hubblez/(2*100.*self.h))*result[0]
        return gf

    def E4(self, z, Omk = 0.):
        """
        return the function
        :math:`E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda+\Omega_k (1+z)^2+\Omega_r (1+z)^4}`

        Use:
        :math:`1+z_{eq} = \Omega_m/\Omega_r`
        """
        Omr = self.Om0/(1+self.zeq)
        return np.sqrt(self.Om0*(1+z)**3.0+(1-self.Om0)+Omk*(1+z)**2.0+Omr*(1+z)**4.0)

    def scale_factor_time(self, a, Omk=0.):
        """
        return the loopback time for the scale factor a
        """
        redshift = (1./a)-1.
        zmax = np.infty
        integrand = lambda z: 1./((1+z)*self.E4(z, Omk))
        result = (1./self.H0)*integrate.quad(integrand, redshift, zmax, limit=1000)[0]
        return result

    def E(self, z):
        """
        return the function :math:`E(z)=\sqrt{\Omega_m(1+z)^3+\Omega_\Lambda}`
        """
        return np.sqrt(self.Om0*(1+z)**3.0+(1-self.Om0))

    def comoving_distance_integrand(self, z):
        return 1./self.E(z)

    def comoving_distance(self, z):
        """
        return the comoving distance :math:`D(z) = \\int \\frac{c dt}{a}`
        as a function of the redshift
        """
        c=299792.458 # speed of light in km/s
        result=(c/self.H0)*integrate.quad(self.comoving_distance_integrand, 0.0, z)[0]
        return result

    def Hubble(self,z):
        return self.H0*np.sqrt(self.Om0*np.power(1+z, 3.0)+1-self.Om0)

    def cmb_lensing_kernel(self, z):
        c=299792.458 # speed of light in km/s
        com_dist=self.comoving_distance(z)
        com_dist_lss=self.comoving_distance(1100.)
        result = 3.*self.Om0*self.H0**2.0*(1+z)*com_dist*(com_dist_lss-com_dist)/(2.*c*com_dist_lss*self.Hubble(z))
        return result

    def volume_factor_integrand(self, z):
        """
        return the volume factor integrand
        """
        DH=3033 # in Mpc/h
        DC=DH*integrate.quad(self.E, 0.0, z)[0]
        DA=DC/(1+z)
        return DH*(1+z)**2.0*DA**2.0/self.E(z)

    def volume_factor(self, z, dz):
        """
        return the volume factor V_i (in (Mpc/h)^3); the
        units is set by the units of DH in volume_factor_integrand
        """
        result=integrate.quad(self.volume_factor_integrand, z-dz/2.0, z+dz/2.0)
        return result[0]

    def comoving_volume(self, z):
        """
        return the comoving (spherical) volume enclosed upto redshift z
        """
        r = self.comoving_distance(z)
        return 4.*np.pi*np.power(r, 3.0)/3

    def comoving_volume_shell(self, z1, z2):
        """
        return the comoving volume of a spherical shell enclosed by two
        redshifts
        """
        return self.comoving_volume(z2)-self.comoving_volume(z1)

    def transfer_function_bbks(self, k):
        """
        return the fitting formula for transfer function by
        Bardeen, Bond, Kaiser, and Szalay (1986, BBKS)

        Eq 7.71 of Dodelson
        """
        q=k/self.Om0/self.h
        return np.log(1+2.34*q)/(2.34*q)*(1+3.89*q+(16.2*q)**2.0+(5.47*q)**3.0+(6.71*q)**4.0)**(-0.25)

    def zdrag(self):
        """ use the EH fitting function
        """
        z_drag_b1=0.313*self.omhh**(-0.419)*(1.+0.607*self.omhh**0.674)
        z_drag_b2=0.238*self.omhh**0.223
        z_drag=1291*(self.omhh**0.251)/(1+0.659*self.omhh**0.828)*(1+z_drag_b1*self.obhh**z_drag_b2)
        return z_drag


    def transfer_function(self, k, hMpc=True):
        """
        return the transfer function T(k) for the current cosmology
        this python version was simply taken from the C code (tf_fit.c) for EH transfer function
        by just making the formulae compatible with python/numpy.

        note that units are in 1/Mpc in the tf_fit.c code; so,
        make sure to look at the additional code right after this comment
        below to change to 1/Mpc if the input is intended to be h/Mpc
        """
        # make sure to change the input k in units of 1/Mpc is your input
        # is in h/Mpc
        #================================================================
        if (hMpc==True):
            # convert the units of k from h/Mpc to 1/Mpc
            k = k * self.h

        # first set parameters as in TFset_parameters
        #=============================================
        Tcmb=self.Tcmb0
        theta_cmb=Tcmb/2.7
        f_baryon=self.f_baryon
        omhh=self.Om0*self.h**2.0
        obhh=omhh*f_baryon
        #h=self.h

        z_equality=2.50e4*omhh/theta_cmb**4.0
        k_equality=0.0746*omhh/theta_cmb**2.0

        z_drag_b1=0.313*omhh**(-0.419)*(1.+0.607*omhh**0.674)
        z_drag_b2=0.238*omhh**0.223
        z_drag=1291*(omhh**0.251)/(1+0.659*omhh**0.828)*(1+z_drag_b1*obhh**z_drag_b2)

        R_drag=31.5*obhh/theta_cmb**4.0*(1000/(1+z_drag))
        R_equality=31.5*obhh/theta_cmb**4.0*(1000/z_equality)

        sound_horizon=2./3./k_equality*np.sqrt(6./R_equality)*np.log((np.sqrt(1+R_drag)+np.sqrt(R_drag+R_equality))/(1+np.sqrt(R_equality)))

        k_silk = 1.6*pow(obhh,0.52)*pow(omhh,0.73)*(1+pow(10.4*omhh,-0.95))

        alpha_c_a1=(46.9*omhh)**0.670*(1+(32.1*omhh)**(-0.532))
        alpha_c_a2=(12.0*omhh)**0.424*(1+(45.0*omhh)**(-0.582))
        alpha_c=alpha_c_a1**(-f_baryon)*(alpha_c_a2)**(-f_baryon**3.0)

        beta_c_b1 = 0.944/(1+pow(458*omhh,-0.708))
        beta_c_b2 = pow(0.395*omhh, -0.0266)
        beta_c = 1.0/(1+beta_c_b1*(pow(1-f_baryon, beta_c_b2)-1))

        y = z_equality/(1+z_drag);
        alpha_b_G = y*(-6.*np.sqrt(1+y)+(2.+3.*y)*np.log((np.sqrt(1+y)+1)/(np.sqrt(1+y)-1)))
        alpha_b = 2.07*k_equality*sound_horizon*pow(1+R_drag,-0.75)*alpha_b_G

        beta_node = 8.41*pow(omhh, 0.435)
        beta_b = 0.5+f_baryon+(3.-2.*f_baryon)*np.sqrt(pow(17.2*omhh,2.0)+1)

        #k_peak = 2.5*3.14159*(1+0.217*omhh)/sound_horizon
        #sound_horizon_fit = 44.5*np.log(9.83/omhh)/np.sqrt(1+10.0*pow(obhh,0.75))

        #alpha_gamma = 1-0.328*np.log(431.0*omhh)*f_baryon + 0.38*np.log(22.3*omhh)*f_baryon**2.0;

        # the TFfit_onek code starts from here
        # ====================================
        q = k/13.41/k_equality
        xx = k*sound_horizon

        T_c_ln_beta = np.log(2.718282+1.8*beta_c*q)
        T_c_ln_nobeta = np.log(2.718282+1.8*q)
        T_c_C_alpha = 14.2/alpha_c + 386.0/(1+69.9*pow(q,1.08))
        T_c_C_noalpha = 14.2 + 386.0/(1+69.9*pow(q,1.08))

        T_c_f = 1.0/(1.0+pow(xx/5.4, 4.0))
        T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*(q*q)) + (1-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*(q*q))

        s_tilde = sound_horizon*pow(1+pow(beta_node/xx, 3.0),-1./3.)
        xx_tilde = k*s_tilde

        T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*(q*q));
        T_b = np.sin(xx_tilde)/(xx_tilde)*(T_b_T0/(1+pow(xx/5.2,2.0))+
		alpha_b/(1+pow(beta_b/xx,3.0))*np.exp(-pow(k/k_silk,1.4)));

        f_baryon = obhh/omhh;
        T_full = f_baryon*T_b + (1-f_baryon)*T_c;
        return T_full
