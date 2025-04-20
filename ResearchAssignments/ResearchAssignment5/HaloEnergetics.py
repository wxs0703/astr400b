# import modules
from os.path import join
import numpy as np
import astropy.units as u
from astropy.constants import G
import scipy.optimize as so
from scipy.integrate import simps
from colossus.cosmology import cosmology
# import plotting modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from MassProfile import MassProfile

m_halo = {'MW': 1.975, 'M31': 1.921, 'M33': 0.187}
cosmo = cosmology.setCosmology('planck18')

class HaloEnergetics:
    '''
    Class that manages calculations of the energies of a halo at a certain snapshot,
    along with relevant methods.
    '''
    
    def __init__(self, galaxy, snap, r_enc=None):
        '''
        Arguments
        ---------
        gal: str, string that represents the halo, e.g. "MW", "M31", or "M33"
        snap: int, snapshot number
        '''
        self.galaxy = galaxy
        self.snap = snap
        self.MP = MassProfile(galaxy, snap)
        self.M = m_halo[galaxy]*1e12
        self.G = self.MP.G.value
        
        if r_enc is not None:
            self.full_compute(r_enc)
    
    def full_compute(self, r_enc):
        '''
        Run all the necessary methods for energy analysis
        
        Arguments
        ---------
        r_enc: np.array, array of radii where enclosed mass will be calculated (not the density profile radii)
        '''
        self.density_profile(r_enc)
        self.fit_hernquist()
        self.hernquist_pot_energy(self.r)
        self.virial_radius()
    
    def save_density_profile(self):
        '''
        Save density profile into npy file
        '''
        np.save(f'data/rho_{self.galaxy}_{self.snap}.npy', np.vstack((self.r, self.rho)))
        
    def load_density_profile(self):
        '''
        Loads saved density profile data into object.
        '''
        profile = np.load(f'data/rho_{self.galaxy}_{self.snap}.npy')
        self.r = profile[0]
        self.rho = profile[1]
        
    def density_profile(self, r_enc):
        '''
        Calculates the density profile of a halo.

        Arguments
        ---------
        r_enc: np.array, array of radii where enclosed mass will be calculated (not the density profile radii)
        '''
        m_enc = self.MP.massEnclosed(1, r_enc).value  # calculate mass enclosed for halo particles
        V_enc = (4/3)*np.pi*r_enc**3  # calculate volume enclosed at each radii
        self.r = (r_enc[1:] + r_enc[:-1]) / 2  # the radii of each density will be in between two m_enc radii
        V = (V_enc[1:] - V_enc[:-1])
        self.rho = (m_enc[1:] - m_enc[:-1]) / V  # calculate density by subtracting m_enc of adjacent radii then dividing by volume
    
    def fit_hernquist(self, a_low=10, a_high=100):
        '''
        Fits the numerical halo density profile using the Hernquist profile.
        
        Arguments
        ---------
        a_low: float, scale radius lower bound for the fitting algorithm (kpc)
        a_high: float, scale radius upper bound for the fitting algorithm (kpc)
        '''
        popt, _ = so.curve_fit(self.hernquist, self.r, self.rho, sigma=1/self.r**3, 
                               bounds=([self.M-0.1, a_low], [self.M+0.1, a_high]), 
                               method='trf', p0=[self.M, 50], max_nfev=1000)
        self.a = popt[1]
    
    def hernquist_pot_energy(self, r):
        '''
        Calculates the gravitational potential energy of the halo based on a hernquist density profile.
        
        Arguments
        ---------
        r: np.array, array of radii to calculate the potential.
        '''
        self.Phi = -self.G*self.M/(r+self.a)
        Phi_0 = -self.G*self.M/(self.a)
        self.U = self.Phi-Phi_0
    
    def virial_radius(self):
        '''
        Calculates the virial radius of the halo, defined as 360 times the dark matter density
        of the universe.
        '''
        rho_dm = cosmo.rho_c(0)*(cosmo.Om(0)-cosmo.Ob(0))
        hern = lambda r: np.abs(self.hernquist(r, self.M, self.a)-rho_dm)
        res = so.minimize(hern, 1000)
        self.r_vir = res.x[0]
    
    @staticmethod
    def hernquist(r, M, a):
        '''
        Computes the Hernquist density profile at a single point or an array of radii.

        Arguments:
        r: float or np.array, radii at which density will be computed
        rho_0: float, central density
        '''
        return (M/(2*np.pi)) * a/(r*(r+a)**3)