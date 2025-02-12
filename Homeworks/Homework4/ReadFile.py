import numpy as np
import astropy.units as u

def read(filename):
    '''
    Reads a simulation data file and outputs info.
    
    Arguments:
    filename: str, data file directory
    
    Returns:
    time: float, simulation time
    N: int, number of particles in simulation
    data: np.array, data for all the particles in file
    '''
    with open(filename, 'r') as file:
        line1 = file.readline()        #read 1st line
        _, value = line1.split()       #extract value from 1st line
        time = float(value)*u.Myr      #set value as time
        line2 = file.readline()        #read 2nd line
        _, value = line2.split()       #extract value from 1st line
        N = int(value)                 #set value as number of particles
    
    #read rest of data file into array
    data = np.genfromtxt(filename, dtype=None, names=True, skip_header=3)

    return time, N, data