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

from ReadFile import Read
from CenterOfMass import CenterOfMass
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
        # Determine Filename
        # add a string of the filenumber to the value "000"
        ilbl = '000' + str(snap)
        # remove all but the last 3 digits
        ilbl = ilbl[-3:]
        # create filenames
        self.filename = f'/home/astr400b/HighRes/{galaxy}/{galaxy}_{ilbl}.txt'
        
        # Define some basic galaxy properties
        self.galaxy = galaxy
        self.snap = snap
        self.MP = MassProfile(galaxy, snap)
        self.M = m_halo[galaxy]*1e12
        
        self.G = self.MP.G.value # big G constant
        
        # Store mass, position, velocity data for later kinetic energy calculation
        isHalo = np.where(self.MP.data['type'] == 1)
        data = self.MP.data[isHalo]
        self.m = data['m']
        self.x = data['x']
        self.y = data['y']
        self.z = data['z']
        self.vx = data['vx']
        self.vy = data['vy']
        self.vz = data['vz']
        del data
        
        # Calculate center of mass
        com = CenterOfMass(self.filename,2)
        self.com_p = com.COM_P(0.1).value
        
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
        self.virial_radius()
        self.hernquist_pot_energy()
        self.kinetic_energy()
    
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
    
    def virial_radius(self):
        '''
        Calculates the virial radius of the halo, defined as 200 times the critical density
        of the universe.
        '''
        rho_vir = cosmo.rho_c(0)*200 # reference density based on vir radius definition
        
        # functions to be minimized
        rho_enc = lambda r: self.MP.hernquistMass(r, self.a, self.M).value/((4/3)*np.pi*r**3)
        overdensity = lambda r: np.abs(rho_enc(r)-rho_vir)
        
        # minimize and extract result
        res = so.minimize(overdensity, 100, bounds=[(10, 1000)])
        self.r_vir = res.x[0]
        
    def hernquist_pot_energy(self):
        '''
        Calculates the gravitational potential energy of the halo based on a hernquist density profile.
        Units: Msun km^2 s^-2
        '''
        # Calculate components of potential energy
        Uinf = -self.G*self.M**2/(6*self.a)
        U1 = lambda r: self.G*self.M**2*self.a/(2*(r+self.a)**2)
        U2 = lambda r: -self.G*self.M**2*self.a**2/(3*(r+self.a)**3)
        
        # Calculate potential energy profile an total potential energy
        self.U = Uinf+U1(self.r)+U2(self.r)
        self.U_tot = Uinf+U1(self.r_vir)+U2(self.r_vir)
    
    def kinetic_energy(self):
        '''
        Calculates the kinetic energy of the halo with respect to r as well as total kinetic energy.
        Units: Msun km^2 s^-2
        '''
        # shift particle coordinates relative to center of mass
        xdist = self.x - self.com_p[0]
        ydist = self.y - self.com_p[1]
        zdist = self.z - self.com_p[2]
        rdist = np.sqrt(xdist**2 + ydist**2 + zdist**2)
        v2 = self.vx**2 + self.vy**2 + self.vz**2
        m = self.m*1e10
        
        # calculate kinetic energy enclosed within each radii
        K = []
        for R in self.r:
            mask = rdist < R
            K.append(np.sum(0.5*m[mask]*v2[mask]))
        self.K = np.array(K)
        
        # calculate total kinetic energy
        mask = rdist < self.r_vir
        self.K_tot = np.sum(0.5*m[mask]*v2[mask])
    
    @staticmethod
    def hernquist(r, M, a):
        '''
        Computes the Hernquist density profile at a single point or an array of radii.

        Arguments:
        r: float or np.array, radii at which density will be computed
        rho_0: float, central density
        '''
        return (M/(2*np.pi)) * a/(r*(r+a)**3)

    
class MergedHaloEnergetics(HaloEnergetics):
    '''
    A child class to HaloEnergetics specifically for analyzing the MW-M31 merged remnant as a whole.
    '''
    
    def __init__(snap, r_enc=None):
        pass