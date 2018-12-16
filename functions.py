''' functions.py
'''

import numpy as np

def cosphi(m):
    """Calculates for x component of path """
    cosphi_input=np.full((2*m),0.5)
    cosphi=np.diag(cosphi_input,k=1)+np.diag(cosphi_input,k=-1)
    return cosphi

def sinphi(m):
    """Calculates for y component of path """
    sinphi_input1=np.full((2*m),0.5j)
    sinphi_input2=np.full((2*m),-0.5j)
    sinphi=np.diag(sinphi_input1,k=1)+np.diag(sinphi_input2,k=-1)
    return sinphi

def ddphi(m):
    """Calculates first derivative of phi, used for solving b-vector in pathtofield)"""
    ddphi_input = np.arange(-m,m+1)
    ddphi = 1j*np.diag(ddphi_input,k=0)
    return ddphi

def d2dphi2(m):
    """Calculates first derivative of phi, used for solving b-vector in pathtofield)"""
    d2dphi2_input = np.arange(-m,m+1)**2
    d2dphi2 = -1*np.diag(d2dphi2_input,k=0)
    return d2dphi2

def d2dt2(x,dt): # pass in a 1d np.array
    """Calculate second derivative of path using centered finite differences, used for solving b-vector in pathtofield"""
    # initial step (finite differences method)
    n=len(x)
    d2x = np.zeros(n,dtype=complex)
    d2x[0] = (x[2]-2*x[1]+x[0])/(dt**2)
    for i in range(1,n-1):
        d2x[i] = ((x[i+1]-x[i])-(x[i]-x[i-1]))/(dt**2)
    d2x[n-1] = (x[n-3]-2*x[n-2]+x[n-1])/(dt**2)
    return d2x
