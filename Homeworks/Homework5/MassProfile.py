import numpy as np
import astropy.units as u
from astropy.constants import G

from ReadFile import read
from CenterOfMass import CenterOfMass

# Set data directory and constant
data_dir = '/home/wxs0703/astr400b/'
G = G.to(u.kpc*u.km**2/u.s**2/u.Msun)

class MassProfile:
    
    def __init__(self, galaxy, snap):
        """ 
        Class that computes mass profiles for a galaxy

        Arguments:
            galaxy: str, string representing which galaxy to compute
            snap: int, snapshot number
        """
        
        # add a string of the filenumber to the value “000”
        ilbl = '000' + str(snap)
        # remove all but the last 3 digits
        ilbl = ilbl[-3:]
        self.filename = data_dir + '%s_'%(galaxy) + ilbl + '.txt'
        
        self.gname = galaxy # store galaxy name
        
        # read x, y, z positions and mass and particle type
        _, _, data = read(self.filename)
        self.x = data['x']*u.kpc
        self.y = data['y']*u.kpc
        self.z = data['z']*u.kpc
        self.m = data['m']
        self.ptype = data['type']
        
    def mass_enclosed(self, ptype, r_array):
        """
        Computes the mass enclosed within an array of radii for a particle type
        
        Arguments:
        ptype: int, particle type
        r_array: np.array, array of radii
        
        Returns:
        m_array: np.array, array of enclosed masses within each radii, in Msun.
        """
        # Check if particle type exists, if not, return null array
        if len(self.ptype[self.ptype == ptype]) == 0:
            return np.full_like(r_array, 0)
        
        # Compute center of mass
        COM = CenterOfMass(self.filename, ptype)
        x_COM, y_COM, z_COM = COM.COM_P()
        
        # Select by particle type and compute distance relative to COM
        ptype_mask = np.where(self.ptype == ptype)
        x_new, y_new, z_new = self.x[ptype_mask], self.y[ptype_mask], self.z[ptype_mask]
        m_new = self.m[ptype_mask]
        r_new = np.sqrt((x_new-x_COM)**2 + (y_new-y_COM)**2 + (z_new-z_COM)**2)
        
        # Compute enclosed mass for each radii
        m_array = []
        for r_enc in r_array:
            r_mask = np.where(r_new < r_enc*u.kpc)
            m_array.append(np.sum(m_new[r_mask]))
        m_array = np.array(m_array)*1e10*u.Msun
        
        return m_array
    
    def mass_enclosed_total(self, r_array):
        """
        Computes the total mass enclosed within an array of radii
        
        Arguments:
        r_array: np.array, array of radii
        
        Returns:
        m_array: np.array, array of total enclosed mass within each radii, in Msun.
        """
        
        # Call mass_enclosed for all particle types and sum
        m_arrays = []
        for ptype in range(1, 4):
            m_arrays.append(self.mass_enclosed(ptype, r_array))
        m_array = np.sum(np.vstack(m_arrays), axis=0)
        
        return m_array
    
    def total_mass(self, ptype):
        """
        Utility function that returns total mass of a particle type
        """
        
        ptype_mask = np.where(self.ptype == ptype)
        m_tot = np.sum(self.m[ptype_mask])*1e10*u.Msun
        
        return m_tot
    
    def hernquist_mass(self, r_array, a):
        """ 
        Function that defines the Hernquist 1990 DM mass profile 

        Arguments:
            r_array: np.array, array of galactocentric radii in kpc
            a: float, scale radius of the Hernquist profile in kpc
            m_halo: float, total halo mass in units of 1e12 Msun 

        Returns:
            mass: astropy quantity, total mass within the input radius r in Msun
        """
        
        m_halo = self.total_mass(1)
        c = r_array**2/(a + r_array)**2
        mass = m_halo*c

        return mass
    
    def circular_velocity(self, ptype, r_array):
        """
        Calculates the circular velocity at each radii for a particle type based on enclosed mass
        
        Arguments:
        ptype: int, particle type
        r_array: np.array, array of radii
        
        Returns:
        vcirc: np.array, array of circular velocities for each radii
        """
        
        # compute mass enclosed within each radii and use centripetal force
        # to calculate vcirc
        m_array = self.mass_enclosed(ptype, r_array)
        vcirc = np.sqrt(G*m_array/r_array)
        
        return vcirc
    
    def circular_velocity_total(self, r_array):
        """
        Calculates the total circular velocity at each radii based on enclosed mass
        
        Arguments:
        r_array: np.array, array of radii
        
        Returns:
        vcirc: np.array, array of total circular velocity for each radii
        """
        
        # compute total mass enclosed within each radii and use centripetal force
        # to calculate vcirc
        m_array = self.mass_enclosed_total(r_array)
        vcirc = np.sqrt(G*m_array/r_array)
        
        return vcirc
    
    def hernquist_vcirc(self, r_array, a):
        """
        Calculates the total circular velocity at each radii using Hernquist enclosed mass
        
        Arguments:
        r_array: np.array, array of radii
        a: float, scale radius of the Hernquist profile in kpc
        
        Returns:
        vcirc: np.array, array of total circular velocity for each radii
        """
        
        # compute hernquist mass enclosed for each radii and use centripetal force
        # to calculate vcirc
        m_array = self.hernquist_mass(r_array, a)
        vcirc = np.sqrt(G*m_array/r_array)
        
        return vcirc