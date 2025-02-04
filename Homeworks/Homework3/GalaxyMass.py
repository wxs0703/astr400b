import numpy as np

from ReadFile import read

def component_mass(filename, ptype):
    '''
    Returns the total mass of all particles of a type.
    
    Arguments:
    filename: str, directory to simulation file
    ptype: int, integer indicating particle type
    
    Returns:
    m_comp: float, total mass of galaxy component in 1e12 M_sun.
    '''
    _, _, data = read(filename)
    data_typed = data[np.where(data['type']==ptype)]
    m_comp = np.round(np.sum(data_typed['m'])/100, 3)
    
    return m_comp