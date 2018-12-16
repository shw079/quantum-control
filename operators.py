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
class ObservableAbstract(abc.ABC):
    """ An Operator is a variable that can be represented as an expectation value < variable|operator|variable >
    Class Operator provides types of operators and methods to utilize them.  This class is an abstract base class.
    """

    @abc.abstractmethod
    def __init__(self):
        """Initialize with matrix repr of operator"""
        pass

    @abc.abstractmethod
    def act_on_a_state(self, state):
        pass

    @abc.abstractmethod
    def get_expt(self, State):
        pass

class ObservableBase(ObservableAbstract):
    def act_on_a_state(self, State):
        return self.operator @ State.as_ket()

    def get_expt(self, State):
        expt = State.as_bra() @ self.operator @ State.as_ket()
        return np.asscalar(expt)

class HamiltonianBase(ObservableBase): # subclass of operator (generation 2)
    """ A Hamiltonian is one type of operator that is used to evolve the system
    Class Operator provides methods to operate on the state and evolve the system
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

class RotorH(HamiltonianBase): # subclass of Hamiltonian (generation 3)
    """ RotorH is a specific type of Hamiltonian that is used to evolve the system by operating on the state  
    Class RotorH is a subclass of Hamiltonian, which is a subclass of Operator.
    """
    def __init__(self, field): # pass in field array at a single time point
        #Hamiltonian.__init__([f.cosphi(const.m),f.sinphi(const.m)]). 
        # I used f.cosphi and f.sinphi here because I could not find a good way to inherit operator from Hamiltonian parent class
        # maybe there is a better way for inheritance bu the only methods I found were to initialize Hamiltonian inside RotorH, which seems backwards
        # so I resolved to just call constphi from functions.py, and did not use self.operator from Hamiltonian
        self.operator = const.B*np.diag((np.arange(-const.m,const.m+1))**2,k=0)-const.mu*f.cosphi(const.m)*field[0]-const.mu*f.sinphi(const.m)*field[1]

    def evolve(self, state_i, dt): # maybe not be necessary for this function to exist, could possibly combine with act_on_state
        U = linalg.expm((-1j/const.hbar)*self.operator*dt)
        state_f = U @ state_i.get_value()
        return State(const.m, state_f)

class DipoleX(ObservableBase):
    def __init__(self):
        self.operator = f.cosphi(const.m)

class DipoleY(ObservableBase):
    def __init__(self):
        self.operator = f.sinphi(const.m)



