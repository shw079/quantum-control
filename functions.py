'''functions.py
'''

import numpy as np
from dataStore import DataStore
import constants as const
import math
from scipy import linalg

def calcOperators(m):
    """Calculate operator representations of cos(phi), sin(phi), d/dphi, d2/dphi2"""

    cosphi_input=np.full((2*m),0.5)
    cosphi=np.diag(cosphi_input,k=1)+np.diag(cosphi_input,k=-1)     # matrix operator representation of cos(phi)

    sinphi_input1=np.full((2*m),0.5j)
    sinphi_input2=np.full((2*m),-0.5j)
    sinphi=np.diag(sinphi_input1,k=1)+np.diag(sinphi_input2,k=-1)     # matrix operator representation of sin(phi)

    ddphi_input = np.arange(-m,m+1)
    ddphi = 1j*np.diag(ddphi_input,k=0)

    d2dphi2 = -1*np.diag(ddphi_input**2,k=0)  # matrix operator representation of d^2/dphi^2

    return cosphi,sinphi,ddphi,d2dphi2

def integrate(sinphi,cosphi,field,state,dt):
    """Integrate Schroedinger equation one time step forward.

    """
    H = const.B*np.diag((np.arange(-const.m,const.m+1))**2,k=0)-const.mu*cosphi*field[0]-const.mu*sinphi*field[1]
    U=linalg.expm((-1j/const.hbar)*H*dt)
    state[:] = U@state[:]
    return state


def pathFromField(self):
    """Calculate predicted path from a given field.

    Note: state can be provided optionally to skip integration.

    """
    pass

def d2dt2(path,dt):
    """Calculate second derivative using centered finite differences."""
    # initial step (finite differences method)
    # add checks for if path length too short
    n=len(path)
    d2x = np.zeros(n,dtype=complex)
    d2y = np.zeros(n,dtype=complex)
    d2x[0] = (path[2,0]-2*path[1,0]+path[0,0])/(dt**2)
    d2y[0] = (path[2,1]-2*path[1,1]+path[0,1])/(dt**2)
    for i in range(1,n-1):
        d2x[i] = ((path[i+1,0]-path[i,0])-(path[i,0]-path[i-1,0]))/(dt**2)
        d2y[i] = ((path[i+1,1]-path[i,1])-(path[i,1]-path[i-1,1]))/(dt**2)
    d2x[n-1] = (path[n-3,0]-2*path[n-2,0]+path[n-1,0])/(dt**2)
    d2y[n-1] = (path[n-3,1]-2*path[n-2,1]+path[n-1,1])/(dt**2)
    return d2x,d2y

def calcDandA(sinphi,cosphi,state):
    sinphi2 = sinphi@sinphi
    cosphi2 = cosphi@cosphi
    sinphi_cosphi = sinphi@cosphi
    cosphi_sinphi = cosphi@sinphi
    state_conj = np.conj(state.transpose())
    D_consts = (4*const.B**2*const.mu**2/const.hbar**4)
    D_term1 = state_conj@sinphi2@state
    D_term2 = state_conj@cosphi2@state
    D_term3 = state_conj@sinphi_cosphi@state
    D_together = D_consts*(D_term1*D_term2-D_term3**2)
    D = D_consts*(D_term1*D_term2-D_term3**2)
    a1x = (2*const.B*const.mu/const.hbar**2)*state_conj@sinphi2@state
    var1 = state_conj@sinphi2
    a2x = (-2*const.B*const.mu/const.hbar**2)*state_conj@sinphi_cosphi@state
    a1y = (-2*const.B*const.mu/const.hbar**2)*state_conj@cosphi_sinphi@state
    a2y = (2*const.B*const.mu/const.hbar**2)*state_conj@cosphi2@state
    return D,a1x,a2x,a1y,a2y

