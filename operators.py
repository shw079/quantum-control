'''operators.py
'''

import numpy as np
# from dataStore import DataStore
import constants as const
import functions as f
import math
from scipy import linalg
import abc
from state import State


#abstract base classes
class Observable(abc.ABC):
    """
    An Operator is a variable that can be represented as an expectation 
    value < variable|operator|variable >. Class Operator provides types 
    of operators and methods to utilize them.  This class is an 
    abstract base class.

    """

    @abc.abstractmethod
    def __init__(self):
        """Initialize with matrix repr of operator"""
        pass

    def act_on_a_state(self, State):
        """Apply operator from the left to a state ket-vector.

        Takes a State object as input arg. Returns a ket-vector of shape 
        same as the one of State.as_ket().

        """
        
        return self.operator @ State.as_ket()

    def get_expt(self, State):
        """Calculate the expectation value of a observable given a state.

        Takes a State object as input arg. Returns a scalar.

        """
        
        expt = State.as_bra() @ self.operator @ State.as_ket()
        return np.asscalar(expt)

class Hamiltonian(Observable): # subclass of operator (generation 2)
    """A Hamiltonian is an operator corresponding with observable energy.

    A Hamiltonian can is also used to derive a unitary operator to evolve 
    the system in time. Class Operator provides methods to operate on the 
    state and evolve the system.
    """

    @abc.abstractmethod
    def evolve(self, state_i,dt):
        """Return state_f"""
        pass

# class Path(Operator): # subclass of operator (generation 2)
#     """ A Path is a 2x1 array that represents the x and y projection of a molecular dipole at a single time point. 
#     Class Path provides functions to operate on states to return a path. Path is a subclass of the Operabor abstract base class.
#     """
#     def __init__(self):
#         self.operator = [f.cosphi(const.m),f.sinphi(const.m)]
#         self.value = None # value is 1 single time point of state (dimention: 2*m+1 x 1)

#     def act_on_state(self, State): # state object is passed here, returns expectation value
#         n = len(State.get_value())
#         xy = np.zeros(2,dtype=complex) # path input
#         xy[0] = State.get_expectation(self.operator[0]).item(0)
#         xy[1] = State.get_expectation(self.operator[1]).item(0)
#         self.value = xy

#     def set_value(self, xy):
#         self.value = xy # xy is a 1x2 array of a single time point of path trajectory

class RotorH(Hamiltonian): # subclass of Hamiltonian (generation 3)
    """A specific type of Hamiltonian that described the rotor molecule.

    Rotor Hamiltonian is described by the basis and the field. Class RotorH 
    is a subclass of Hamiltonian, which is a subclass of Observable.
    """
    def __init__(self, m, field): # pass in field array at a single time point
        #Hamiltonian.__init__([f.cosphi(const.m),f.sinphi(const.m)]). 
        # I used f.cosphi and f.sinphi here because I could not find a good way to inherit operator from Hamiltonian parent class
        # maybe there is a better way for inheritance bu the only methods I found were to initialize Hamiltonian inside RotorH, which seems backwards
        # so I resolved to just call constphi from functions.py, and did not use self.operator from Hamiltonian
        self.operator = (const.B*np.diag((np.arange(-m,m+1))**2,k=0)
                        -const.mu*f.cosphi(m)*field[0]
                        -const.mu*f.sinphi(m)*field[1])

    def evolve(self, state_i, dt): # maybe not be necessary for this function to exist, could possibly combine with act_on_state
        U = linalg.expm((-1j/const.hbar)*self.operator*dt)
        state_f = U @ state_i.as_ket()
        return State(const.m, state_f)

class DipoleX(Observable):
    def __init__(self, m):
        self.operator = f.cosphi(m)

class DipoleY(Observable):
    def __init__(self, m):
        self.operator = f.sinphi(m)



