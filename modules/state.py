'''!@namespace state

@brief Implementation of class State to hold weights of basis 
wavefunctions describing the system.
'''

import numpy as np

class State(object):
    """!@brief Contains weights of basis wavefunctions and to derive 
    properies of the system.

    Class State provides the following functionalities:

    1. As a container for calculated state at a single time point,
    which is a (2m+1)x1 array where m is the quantum number; and

    2. Functions to access the state and, in conjunction with 
    other objects, to derive other properties of the system.

     """

    def __init__(self, m, value=None):
        """!@brief Initializes a State instance with quantum number 
        `m` and optionally weights of each quantum state.

        @param m: Maximum energy quantum number. Guaranteed to be
         `int` (not yet implemented)

        @param value: (Optional) Wight of each basis wavefunction. 
        Guaranteed to be a np.ndarray containing 2m+1 numbers 
        corresponding to the 2m+1 basis wavefunctions describing 
        the system. If not provided, this will be weights 
        corresponding to the ground state.

        """

        ## Maximun energy quantum number
        self.m = m

        # Generate weights for ground state if weights not provided
        if value is None:
            ## Weights for 2m+1 basis; np.ndarray with shape (2m+1,)
            self.value = np.zeros(int(2*m+1))

        #check format of input probability
        #maybe in the future can accept array-like?
        elif not isinstance(value, np.ndarray):
            errmsg = "Expect np.array to be input."
            raise TypeError(errmsg)
        elif value.size is not (2*m+1):
            errmsg = "Expect input to have " + str(2*m+1) + " elements."
            raise ValueError(errmsg)
        else: 
            self.value = value.reshape((2*m+1,))

    def as_bra(self):
        """!@brief Calculates the complex conjugate (known as `bra`) of 
        the weights (self.value).

        @return A numpy.ndarray with shape (1,2m+1)
        """

        return np.conj(self.value).reshape((1,2*self.m+1))

    def as_ket(self):
        """!@brief Returns the weights array (self.value) as a column 
        vector, known as `ket`.

        @return A numpy.ndarray with shape (2m+1,1)
        """

        return self.value.reshape((2*self.m+1,1))

    def get_expt(self, operator):
        """!@brief Calculates the expectation value of an observable 
        for the state.

        This is same as the following expression: < bra | operator | ket >
        where the `operator` is the matrix representation of a 
        corresponding observable.

        @param operator: The matrix representation of a specific 
        observable of interest. Guaranteed to be a numpy.ndarray of
        shape (2m+1,2m+1) where m is the maximum energy quantum number
        the state used (i.e., m = self.state.value.size) (Not 
            implemented yet.)

        @return A scalar of the calculated expectation value.

        """

        expt = self.as_bra() @ operator @ self.as_ket()
        return np.asscalar(expt)