'''!@namespace molecule

@brief Implementation of class Rotor as a basic unit to describe a 
system of interest.
'''

import abc
import numpy as np
from state import State
import functions as f
from scipy import linalg
import constants as const

class Molecule(abc.ABC):
    """!@brief Abstract base class for molecules (system of interest)
    """
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def evolve(self):
        """Evolve the state to next time point by a unitary operator"""
        pass

class Rotor(Molecule):
    """!@brief A basic unit to describe a rotor molecule.

    Class Rotor provides a basic unit to describe and manipulate a 
    rotor molecule. It can be used within a solver like 
    solver.solvers.PathToField or solver.solvers.FieldToPath.

    In contains information including the maximum energy quantum 
    number `m`, the current time, and the current state of the 
    molecule. It also has the external field applied to the rotor and 
    the hamiltonian of the rotor. A rotor molecule can evolve itself 
    according to its hamiltonian, update the time, state, field and 
    write them into its history, as well as exporting the history as 
    np.ndarray.

    """

    def __init__(self, m):
        """!@brief Initializes a Rotor instance with quantum number m 
        and sets it at its ground state.

        @param m: Maximum energy quantum number

        """
        ## Maximun energy quantum number
        self.m = m
        ground_state = np.zeros(2*m+1)
        ground_state[m] = 1.0
        ## State object (solver.state.State) containing weights for 
        ## basis to describe the molecule 
        self.state = State(m, ground_state)
        ## Field object (solver.field.Field) containing the external 
        ## control field that will change the hamiltonian of the
        ## molecule
        self.field = np.zeros(2)
        ## Molecule-specific Hamiltonian object, in this case, a 
        ## RotorH (solver.observable.RotorH).
        self.hamiltonian = self._get_hamiltonian()
        self.dipole_x = f.cosphi(self.m)
        self.dipole_y = f.sinphi(self.m)
        ## Current time
        self.time = 0.0
        ## A dictionary for history of `time`, `state`, and `field` 
        ## of the molecule. Each property's history is recorded as a 
        ## list, with each element a scalar for `time`, a 
        ## solver.state.State object for `state`, and a 
        ## solver.field.Field object for `field`.
        self.history = {'time':[self.time],
                        'state':[self.state],
                        'field':[self.field]}

    def evolve(self, dt):
        """!@brief Evolve and update the state of molecule using its 
        hamiltonian.

        Method `evolve` invokes the method in hamiltonian to evolve 
        the molecule state by `dt` forward in time. It then updates 
        the time and state, and records them into history. 

        @param dt: Step size of time.

        """

        #Use haniltonian to evolve the current state 
        U = linalg.expm((-1j/const.hbar)*self.hamiltonian*dt)
        weights = U @ self.state.as_ket()
        state_new = State(self.m, weights)
        #Update (including writing history) of time and state
        self.update_state(state_new)
        self.update_time(self.time+dt)

    def _get_hamiltonian(self):
        """!@brief Calculate rotor hamiltonian with the current 
        control field

        @return Hamiltonian matrix representation. A np.ndarray of 
        shape (2m+1,2m+1) where m is the maxinum energy quantum 
        number.

        """

        m = self.m
        field = self.field
        H = (const.B*np.diag((np.arange(-m,m+1))**2,k=0)
            -const.mu*f.cosphi(m)*field[0]
            -const.mu*f.sinphi(m)*field[1])

        return H


    def set_field(self, field):
        """!@brief Set the external field and calculate hamiltonian accordingly.

        `set_field` will set rotor.field to input field object and 
        change self.hamiltonian to the new hamiltonian based on the 
        new field. This DOES NOT add a new value to history but 
        instead change the last value of history, and hence should 
        only be used to set the initial field when a rotor object 
        instantiated within a solver. (solver.solvers.FieldToPath or 
        solver.solvers.PathToField)

        @param field: A solver.field.Field object containing the new 
        external field to set to the molecule.

        """

        self.field = field
        self.hamiltonian = self._get_hamiltonian()
        # rewrite history manually
        self.history['field'][-1] = field

    def update_time(self, time):
        """!@brief Set and update time of molecule with history appended."""
        self.time = time
        self.history['time'].append(time)

    def update_state(self, state):
        """!@brief Set and update state of molecule with history appended."""
        self.state = state
        self.history['state'].append(state)

    def update_field(self, field):
        """!@brief Set and update field of molecule with history 
        appended and hamiltonian recalculated.
        """

        self.field = field
        self.history['field'].append(field)
        #calculate and set the new hamiltonian
        self.hamiltonian = self._get_hamiltonian()

    def get_time_asarray(self):
        """!@brief Return history of time as an array.

        @return Numpy ndarray of shape (n,) where n is the number of 
        time points.
        """
        times = [time for time in self.history['time']]
        return np.stack(times)

    def get_states_asarray(self):
        """!@brief Return history of state as an array.

        @return Numpy ndarray of shape (2m+1,n) where m is the 
        maximum energy quantum number of the rotor and n is the 
        number of time points. Each column of this returned 2D-array
        describes the weights of basis wavefunctions at each time 
        point.

        """
        states = [state.value.flatten() for state in self.history['state']]
        return np.stack(states, axis=1)

    def get_fields_asarray(self):
        """!@brief Return history of field as an array.

        @return Numpy ndarray of shape (n,2) where n is the number of 
        time points. Each row is the control field at each time point.

        """
        fields = [field.flatten() for field in self.history['field']]
        return np.stack(fields, axis=0)











