'''dataStore class
'''

import numpy as np
from collections import namedtuple
import constants as const

class DataStore(object):
    """A DataStore contains all data associated with a single path.

    Class DataStore provides a container to store data associated with a 
    single path specified by user. A DataStore object can be instantiated
    either with or without input argument. In the latter case, the object
    contains only predefined constants. Otherwise, a desired path processed
    by GUI_Interface can be used as input to instantiate a DataStore, and
    classes FieldSolver, NoiseAnalyzer, and Visualization can then access/
    save data within the DataStore for the desired path.
  
    Attributes:

        Const: A named tuple contains predefined constants in our quantum 
        system. Each constant can be accessed by its name as an attribute 
        of object Const. These include:
            -- m:       maximum energy quantum number
            -- B:       rotational constant in atomic units
            -- mu:      dipole moment in atomic units
            -- hbar:    reduced planck's constant
            -- K:       4pi * epsilon0
            -- w1:      first energy level spacing

        n: Number of time points. n is guaranteed to be >= 2.

        t: Time at each time points guaranteed to be strictly increasing.
        A numpy ndarray of shape (n,).

        path_desired: User-specified desired path of dipole projection 
        described by x and y (in this order). A numpy ndarray of shape 
        (n,2).

        field: Control fields e_x and e_y at each time point calculated by 
        FieldSolver. A numpy ndarray of shape of (n,2).

        path_actual: Path of dipole projection described by x and y that is
        resulted from the control field. A numpy ndarray of shape (n,2).

        state: State described by (2m+1) basis at each time point. A numpy 
        ndarray of shape ((2m+1),n).

        noise_stat: A dictionary composed of the following two objects:
            -- "mean": mean x and y at each time point. A numpy ndarray of
            shape (n,2).
            -- "sd": standard deviation of x and y at each time point. A
            numpy ndarray of shape (n,2).

    """

    def __init__(self, txy_desired=None):
        """Initialize a DataStore instance.
        
        DataStore.Const is always initialized by class method init_const()
        with the predefined constants. Other class attributes are ini-
        tialized only when parameter txy_desired is not None.

        Parameters:

            txy_desired: Optional. A numpy ndarray of shape (n,2) where 
            n >= 2 and each row contains (in this order) time, x projection
            and y projection of path defined by user. 

        Raises:

            TypeError: if input is not an instance of numpy.ndarray

            ValueError: if one of the following:
                -- txy_desired is not of shape (n,2)
                -- txy_desired contains nan or inf
                -- time in txy_desired (i.e. txy_desired[:,0]) is not 
                strictly increasing.

        """
        
        #check type and shape of input txy_desired
        if not isinstance(txy_desired, np.ndarray):
            errmsg = ("DataStore can only be instantiated with n-by-2 "
                      "numpy ndarray as input argument if any.")
            raise TypeError(errmsg)
        isnot2dim = txy_desired.ndim is not 2
        isnot2col = txy_desired.ndim is 2 and txy_desired.shape[1] is not 2
        if isnot2dim or isnot2col:
            errmsg = ("DataStore can only be instantiated with n-by-2 "
                      "numpy ndarray as input argument if any.")
            raise ValueError(errmsg)
        #check nan or inf in input txy_desired
        if np.isnan(txy_desired).any() or np.isinf(txy_desired).any():
            errmsg = ("DataStore can only be instantiated with n-by-2 "
                      "numpy ndarray as input argument if any.")
            raise ValueError(errmsg)

        #attr calculated from input txy_desired
        self.n = txy_desired.shape[0] #number of time points
        self.t = txy_desired[:,0].astype(float)
        self.path_desired = txy_desired[:,0:].astype(complex)
        #attr initialized with all entries being zeros but of correct
        #shapes
        self.path_obs = np.zeros_like(self.path_desired)
        self.field = np.zeros((self.n, 2),dtype=complex)
        self.state = np.zeros((2*const.m+1, self.n),dtype=complex)
        self.noise_stat = dict({
            "mean": np.zeros((self.n, 2)),
            "sd": np.zeros((self.n, 2))
            })

        #raise error if time not strictly increasing
        #if not np.all( self.t[1:] > self.t[:-1] ):
        #    errmsg = ("Time series provided is not strictly increasing.")
        #    raise ValueError(errmsg)


    #def get_state(self):
    #    return self.state





