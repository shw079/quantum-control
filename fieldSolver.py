'''fieldSolver.py

'''

import numpy as np
from dataStore import DataStore
import constants as const
import functions as f
from collections import namedtuple
import math

class FieldSolver(object):
    """Workhouse class FieldSolver.

    """
    def __init__(self,DataStore=None):
        """
            Point to arrays in DataStore:
        """

        if DataStore is not None: # this is for if no object is passed, but maybe not necessary? datastore object always passed?
            self.path_desired = DataStore.path_desired # initial path given by user
            self.state = DataStore.state 
            self.field = DataStore.field
            self.path_obs = DataStore.path_obs # path calculated
            self.n = DataStore.n
            self.tf = (2*math.pi*const.hbar/const.B)*100/(2*math.pi) #not sure how we are deciding on timesteps, here using fixed final time to calculate time step 
            self.dt = self.tf/self.n


    def PathToField(self):
        """Calculate field and save it within DataStore."""
        
        # initial step calculations
        t = 0

        # generate operators
        cosphi,sinphi,ddphi,d2dphi2 = f.calcOperators(const.m)
        sinphi2 = sinphi@sinphi
        cosphi2 = cosphi@cosphi
        sinphi_cosphi = sinphi@cosphi
        cosphi_sinphi = cosphi@sinphi

        # initialize state
        self.state[const.m,t] = 1 # ground state initial condition
        state_conj = np.conj(self.state[:,t].T) # transposed state (psi_conj)

        # determine actual path at initial time t=0
        self.path_obs[t,0] = np.conj(self.state[:,t].T)@cosphi@self.state[:,t] # x path
        self.path_obs[t,1] = np.conj(self.state[:,t].T)@sinphi@self.state[:,t] # y path

        # solve system of equations
        D,a1x,a2x,a1y,a2y = f.calcDandA(sinphi,cosphi,self.state[:,t])
        
        # A^-1
        A_inv = (1/D)*np.array([[a2y, -a1y],[-a2x, a1x]])

        # Calculate B vector
        d2x,d2y = f.d2dt2(self.path_desired,self.dt) # 2nd derivative calculation
        b1 = d2x[0]+np.real((const.B**2/const.hbar**2)*(state_conj@(cosphi+4*sinphi@ddphi-4*cosphi@d2dphi2)@self.state[:,t]))
        b2 = d2y[0]+np.real((const.B**2/const.hbar**2)*(state_conj@(sinphi-4*cosphi@ddphi-4*sinphi@d2dphi2)@self.state[:,t]))

        # initial step epsilon/field calculated
        self.field[t,:] = np.real(A_inv@[b1, b2])

        # loop through timesteps
        for i in range(1,self.n):

            # integrate
            self.state[:,i] = f.integrate(sinphi,cosphi,self.field[i-1,:],self.state[:,i-1],self.dt)
            state_conj = np.conj(self.state[:,i].transpose())

            D,a1x,a2x,a1y,a2y = f.calcDandA(sinphi,cosphi,self.state[:,i])
            A_inv = (1/D)*np.array([[a2y, -a1y],[-a2x, a1x]])
            b1 = d2x[i]+np.real((const.B**2/const.hbar**2)*(state_conj@(cosphi+4*sinphi@ddphi-4*cosphi@d2dphi2)@self.state[:,i]))
            b2 = d2y[i]+np.real((const.B**2/const.hbar**2)*(state_conj@(sinphi-4*cosphi@ddphi-4*sinphi@d2dphi2)@self.state[:,i]))
            self.field[i,:] = A_inv@[b1, b2]

            # recalculate path as a check, does this need to go in another function (verify_path)?
            self.path_obs[i,0] = state_conj@cosphi@self.state[:,i]
            self.path_obs[i,1] = state_conj@sinphi@self.state[:,i]

        # convert field to appropriate units (V/angstrom)    
        field_const = 5.142*(10**11)*(10**-10)
        self.field = self.field*field_const

        # should this be returning entire datastore object instead?
        return self.field,self.state[:,i],self.path_obs
    
    def _fieldFromPath(self):
        """Calculate field from desired path."""
        pass

    def _verify_path(self):
        """Compare desired path and path resulted from calculated field."""
        pass



