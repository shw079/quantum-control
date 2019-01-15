'''Functions used in solver class

'''

import numpy as np
from state import State

def cosphi(m):
    """Get operator for x-component of dipole moment projection.

    Parameters
    ----------
    m : int
        Maxium energy quantum number.

    Returns
    -------
    cosphi : numpy.array, shape=(2m+1,2m+1)
        Matrix representation of operator for x-component of dipole 
        moment projection.

    """
    cosphi_input=np.full((2*m),0.5)
    cosphi=np.diag(cosphi_input,k=1)+np.diag(cosphi_input,k=-1)
    return cosphi

def sinphi(m):
    """Get operator for y-component of dipole moment projection.
    
    Parameters
    ----------
    m : int
        Maxium energy quantum number.

    Returns
    -------
    sinphi : numpy.array, shape=(2m+1,2m+1)
        Matrix representation of operator for y-component of dipole 
        moment projection.

    """
    sinphi_input1=np.full((2*m),0.5j)
    sinphi_input2=np.full((2*m),-0.5j)
    sinphi=np.diag(sinphi_input1,k=1)+np.diag(sinphi_input2,k=-1)
    return sinphi

def ddphi(m):
    """Calculates the operator to get the first derivative of phi, 
    used for solving b-vector in solvers.PathToField._get_b method.

    Parameters
    ----------
    m : int
        Maxium energy quantum number.

    Returns
    -------
    ddphi : numpy.array, shape=(2m+1,2m+1)
        Matrix representation of the operator to get the first 
        derivative of phi.

    """
    ddphi_input = np.arange(-m,m+1)
    ddphi = 1j*np.diag(ddphi_input,k=0)
    return ddphi

def d2dphi2(m):
    """Calculates the operator to get the second derivative of phi, 
    used for solving b-vector in solvers.PathToField._get_b method.

    Parameters
    ----------
    m : int
        Maxium energy quantum number.

    Returns
    -------
    d2dphi2 : numpy.array, shape=(2m+1,2m+1)
        Matrix representation of the operator to get the second 
        derivative of phi.

    """
    d2dphi2_input = np.arange(-m,m+1)**2
    d2dphi2 = -1*np.diag(d2dphi2_input,k=0)
    return d2dphi2

def d2dt2(x,dt):
    """Calculate second derivative of a sequence of scalar numbers. 
    Assume the time differecne between two adjacent points is the 
    same throughout the entire sequence. 

    This is used for solving b-vector in solvers.PathToField._get_b 
    method.

    Parameters
    ----------

    x : numpy.array, shape=(n,)
        An 1-D sequence of numbers.

    dt : float
        Step size of time, i.e. the time difference between two 
        adjacent numbers in `x`.

    Returns
    -------
    d2x : numpy.array, shape=(n,)
        The second derivative of the original sequence `x`.

    """
    # initial step (finite differences method)
    n=len(x)
    d2x = np.zeros(n,dtype=float)
    d2x[0] = (x[2]-2*x[1]+x[0])/(dt**2)
    for i in range(1,n-1):
        d2x[i] = ((x[i+1]-x[i])-(x[i]-x[i-1]))/(dt**2)
    d2x[n-1] = (x[n-3]-2*x[n-2]+x[n-1])/(dt**2)
    return d2x

