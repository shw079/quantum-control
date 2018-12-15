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
