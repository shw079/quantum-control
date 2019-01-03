'''state.py
'''

import numpy as np

class State(object):
    """ A State is a (2m+1)x1 array where m is the quantum number
    Class State provides a container for calculated state at a single time point
    and functions to access the state
    """

    def __init__(self, m, value=None):
        """Initializes a State instance with quantum number m and 
        optionally probablility of each quantum state.
        """

        self.m = m

        #with no probability specified
        if value is None:
            self.value = np.zeros(int(2*m+1))
        #check format of input probability
        #maybe in the future can accept array-like
        elif not isinstance(value, np.ndarray):
            errmsg = "Expect np.array to be input."
            raise TypeError(errmsg)
        elif value.size is not (2*m+1):
            errmsg = "Expect input to have " + str(2*m+1) + " elements."
            raise ValueError(errmsg)
        else: 
            self.value = value.reshape((2*m+1,))
        
    # def get_value(self):
    #     """ Allow for user to return values of state """
    #     return self.value

    def as_bra(self):
        """ calculates the complex conjugate of the input state, known as 'bra' """
        return np.conj(self.value).reshape((1,2*self.m+1))

    def as_ket(self):
        """ returns the original input state, known as 'ket' """
        return self.value.reshape((2*self.m+1,1))

    def get_expt(self, operator):
        """ calculates the expectation value of the operator acting on the state
        < bra | operator | ket >
        """
        expt = self.as_bra() @ operator @ self.as_ket()
        return np.asscalar(expt)