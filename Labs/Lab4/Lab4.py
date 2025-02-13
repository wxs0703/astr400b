

# In Class Lab 4
# G. Besla 

# import relevant modules 
import astropy.units as u
import numpy as np
from astropy import constants as const # import astropy constants


# The Large Magellanic Cloud is at a distance of 50 kpc from the Galactic Center. 
# It is observed to have a stellar disk that extends to a radius of at least 18.5 kpc.
# 
# ![LMC](./tidal.png)
# Deep photometric imaging reveals the faint stellar outskirts of the LMC. 
# Outskirts: DECam data Mackey+2016 MNRAS 459, 239. 
# Inner: shallower imaging from robotic telescopes Besla+2016 APJ 825.
# 
# In this lab we will determine
# the minimum mass required for the LMC so that it maintains the observed radius 
# in the face of the Milky Way's tidal field. 

# # Part A
# 
# We define the mass profile of the Milky Way using a Hernquist profile.
# 
# 
# $\rho(r) =  \frac{M_{halo}}{2\pi} \frac{h_a}{r(r+h_a)^3} \qquad M(r) =  \frac{M_{halo} r^2}{(h_a+r)^2}$ 
# 
# 

# ## #1
# 
# Create a function `hernquist_mass` that returns the dark matter halo mass at a given radius in units of solar mass.
# This function should take as input:  the distance from the Galactic center $r$, the scale radius $h_a$, and the halo mass $M_{halo}$.
# 
# 
# For the Hernquist scale radius for the Milky Way, use the default value of $h_a=60$ kpc. 
# 
# For $M_{halo}$ use your answer for the total mass of the simulated Milky Way you computed in Homework 3 as the default value (in units of 1e12). 



def hernquist_mass(r,h_a=60*u.kpc, m_halo=0): # ADD m_halo=??
    """ Function that defines the Hernquist 1990 mass profile 
    Inputs:
        r: astropy quantity
            Galactocentric distance in kpc
        a: astropy quantity
            scale radius of the Hernquist profile in kpc
        m_halo: float
            total halo mass in units of 1e12 Msun 
        
    Ouputs:
        mass:  astropy quantity
            total mass within the input radius r in Msun
    """
    
    
    
    mass = 0 # TEMPLATE Add   
    
    return mass


# ## #2
# 
# Compute the total mass of the Milky Way within 50 kpc, including its baryonic components (Dark Matter + Bulge + Disk). Use your answers from Homework 3 for the Bulge and Disk Masses. 
# Store this as a variable called `mass_MW50`.
# 






# # Part B
# 
# The Jacobi Radius for a satellite on a circular orbit about an extended host, where 
# the host is assumed to be well modeled as an isothermal sphere halo:
# 
# 
# $R_j = r  \bigg( \frac{M_{sat}}{2 M_{host}(<r)} \bigg)^{1/3}$
# 
# 
# The Isothermal Sphere approximation is not a bad one within 50 kpc.
# 
# Note also that the LMC is not on a circular orbit, but it is very close to its pericentric approach, where the velocity is all in the tangential component. So this isn't a terrible approximation either. 
# 
# ## #1
# Create a function called `jacobi_mass` that returns the total mass of a satellite galaxy in units of Msun, 
# such that it has a given size 
# 
# Do this by rearranging the Jacobi Radius equation to solve for the satellite mass. 
# 




def jacobi_mass(rj,r,m_host):
    """ Function that determines the minimum satellite
    mass needed to maintain a the size of a given 
    satellite using the Jacobi Radius
    
    Inputs:
        rj : astropy quantity
            Jacobi Radius or the stellar radius of the 
            satellite in kpc
        r : astropy quantity 
            Distance of the satellite from the host in kpc
        m_host: astropy quantity 
            Mass of the host galaxy in Msun within r in Msun
        
    Outputs:
        m_min: astropy quantity
            Minimum satellite mass in Msun
    """
    
    m_min = 0 # TEMPLATE ADD HERE
    
    return m_min
    


# ## #2 
# 
# Determine the minimum total mass of the LMC needed to maintain its radius of 18.5 kpc in the face of the Milky Way's tidal 
# field at its current distance of 50 kpc. Store this as a variable called `LMC_jacobiM`.






# Recall that, ignoring centrifugal forces and assuming the host is a point mass, the tidal radius is given as :
# 
# $r_{tide} = r\left (\frac{m_{sat}}{4M_{host} } \right)^{1/3} $
# 
# Since we have a factor of 4 in the denominator instead of 2, the required LMC mass to maintain a radius of 18.5 kpc would be a factor of 2 larger under the point mass assumption.

# ## #3
# 
# How does the total mass of the LMC compare to its stellar mass (M$_\ast = 3 \times 10^9$ M$_\odot$)? 
# 





