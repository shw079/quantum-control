'''Implementation of class State to hold amplitutes of basic wave 
functions describing the system and member methods to use.

'''

import numpy as np

class State(object):
    """Contains amplitudes of basis wave functions and to derive 
    properies of the system.

    Class State provides the following functionalities:

    1. As a container for calculated state at a single time point,
    which is a (2m+1,) array where m is the maximum energy quantum 
    number.

    2. Functions to access the state and, in conjunction with 
    other objects (e.g., operators), to derive other properties of 
    the system.

    Parameters
    ----------
    m: int
        Maximim energy quantum number.

    value: numpy.array, shape=(2m+1,)
        Amplitude of each basic wave function. Default to None which 
        will generate an amplitude-vector for the ground state.

    Attributes
    ----------
    m: int
        Maximim energy quantum number.

    value: numpy.array, shape=(2m+1,), optional (default=None)
        Amplitude of each basic wave function. Default to None which 
        will generate an amplitude-vector for the ground state.

    """

    def __init__(self, m, value=None):
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
        """Calculates the complex conjugate (known as `bra`) of 
        the weights (self.value).

        Returns
        -------
        numpy.array, shape=(1,2m+1)
            Conjugate transpose of the state amplitudes.

        """

        return np.conj(self.value).reshape((1,2*self.m+1))

    def as_ket(self):
        """Returns the state amplitudes vector, known as `ket`.

        Returns
        -------
        numpy.array, shape=(2m+1,1)
            Vector of state amplitudes.

        """

        return self.value.reshape((2*self.m+1,1))

    def get_expt(self, operator):
        """Calculates the expectation value of an observable for the 
        state.

        This is same as the following expression: 
        < bra | operator | ket >
        where the `operator` is the matrix representation of a 
        corresponding observable.

        Parameters
        ----------
        operator: numpy.array, shape=(2m+1,2m+1)
            The matrix representation of a specific observable of 
            interest.

        Returns
        -------
        expt: float
            A scalar of the calculated expectation value for the 
            observable described by the input operator.

        """

        expt = self.as_bra() @ operator @ self.as_ket()
        expt = np.asscalar(expt)
        return expt
