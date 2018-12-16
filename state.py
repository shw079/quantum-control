'''state.py
'''

import numpy as np

class State(object):
    """ A State is a (2m+1)x1 array where m is the quantum number
    Class State provides a container for calculated state at a single time point
    and functions to access the state
    """

    def __init__(self, m, value=None):
        """ Initializes a State instance with appropriate length m """
        self.m = m
        if value is not None:
            self.value = value
        else:
            self.value = np.zeros(m)
        

    def get_value(self):
        """ Allow for user to return values of state """
        return self.value

    def as_bra(self):
        """ calculates the complex conjugate of the input state, known as 'bra' """
        return np.conj(self.value).reshape((1,2*self.m+1))

    def as_ket(self):
        """ returns the original input state, known as 'ket' """
        return self.value.reshape((2*self.m+1,1))

    def get_expectation(self, operator):
        """ calculates the expectation value of the operator acting on the state
        < bra | operator | ket >
        """
        return self.as_bra() @ operator @ self.as_ket()