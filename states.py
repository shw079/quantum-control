''' states.py
'''

import numpy as np
from state import State

class States(object):
    """Collection of state array at every time point ((2m+1) x n time points)
    States is built with n column vectors with each vector a 'state' at a single time instance
    """
    def __init__(self, n, m): 
        """ Initializes a Field instance with appropriate length (n number of time points) """
        self.allStates = np.zeros((2*m+1,n))

    def set_value(self, i, State):
        """ Allow for user to set the corresponding time column vector of States with the calculated 
        column vector state (single time instance) 
        """
        # there use to be varible 'list' here, but I am not sure what list is, don't think it is necessary...
        #self.list = State
        self.allStates[:,i] = State.get_value()

    def get_value(self):
        """ Allow for user to return entire States array (state vector at all time points) """
        return self.allStates
