import numpy as np
import astropy.units as u

from ReadFile import read

def particle_info(filename, ptype, pid):
    '''
    Returns key properties of a simulation object.

    Arguments:
    filename: str, data file directory
    ptype: int, index representing the type of particle
    pid: int, index of particle within a type
    
    Returns:
    dist: float, 3D distance to particle in kpc
    v: float, 3D velocity of particle in km/s
    m: float, mass of particle in M_sun
    '''
    time, N, data = read(filename)                               #read out simulation data
    data_typed = data[np.where(data['type']==ptype)]             #filter only data of given type
    data_obj = data_typed[pid]                                   #pick out specified object

    _, m, x, y, z, vx, vy, vz = data_obj                         #extract mass, positions and velocities
    dist = np.round(np.sqrt(x**2 + y**2 + z**2), 3)*u.kpc        #calculate distance
    v = np.round(np.sqrt(vx**2 + vy**2 + vz**2), 3)*u.km/u.s     #calculate velocity
    m = np.round(m*1e10, 3)*u.Msun                               #extract mass

    return dist, v, m