'''Implementation of class Rotor as a basic unit to describe a system 
of interest.

'''

import abc
import numpy as np
from state import State
import functions as f
from scipy import linalg
import constants as const

class Molecule(abc.ABC):
    """Abstract base class for molecules (i.e., system of interest)
    """
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def evolve(self):
        """Evolve the state to next time point by a unitary operator"""
        pass

class Rotor(Molecule):
    """A basic unit to describe a rotor molecule.

    Class Rotor provides a basic unit to describe and manipulate a 
    rotor molecule. It can be used within a solver like 
    solvers.PathToField or solvers.FieldToPath.

    In contains information including the maximum energy quantum 
    number `m`, the current time, and the current state of the 
    molecule. It also has the external field applied to the rotor and 
    the hamiltonian of the rotor. A rotor molecule can evolve itself 
    according to its hamiltonian, update the time, state, field and 
    write them into its history, as well as exporting the history as 
    np.ndarray.

    Parameters
    ----------
    m: int
        Maximum energy quantum number

    Attributes
    ----------
    state: State object
        Contining amplitudes of basic wave functions for the molecule

    field: numpy.array, shape=(2,)
        External control field expresses as (e_x, e_y)

    hamiltonian: numpy.array, shape=(2m+1,2m+1)
        Matrix representation of molecule-specific Hamiltonian 
        operator.

    dipole_x: numpy.array, shape=(2m+1,2m+1)
        Matrix representation of operator for x-projection of dipole 
        moment.

    dipole_y: numpy.array, shape=(2m+1,2m+1)
        Matrix representation of operator for y-projection of dipole 
        moment.

    """

    def __init__(self, m):
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
        """Evolve and update the state of molecule using its 
        hamiltonian.

        Method `evolve` invokes the method in hamiltonian to evolve 
        the molecule state by `dt` forward in time. It then updates 
        the time and state, and records them into history. 

        Parameters
        ----------
        dt: float
            Step size of time.

        """

        #Use haniltonian to evolve the current state 
        U = linalg.expm((-1j/const.hbar)*self.hamiltonian*dt)
        weights = U @ self.state.as_ket()
        state_new = State(self.m, weights)
        #Update (including writing history) of time and state
        self.update_state(state_new)
        self.update_time(self.time+dt)

    def _get_hamiltonian(self):
        """Calculate rotor hamiltonian with the current control field.

        Returns
        -------
        H: numpy.array, shape=(2m+1,2m+1)
            Matrix representation for Hamiltonian operator.

        """

        m = self.m
        field = self.field
        H = (const.B*np.diag((np.arange(-m,m+1))**2,k=0)
            -const.mu*f.cosphi(m)*field[0]
            -const.mu*f.sinphi(m)*field[1])

        return H


    def set_field(self, field):
        """Set the external field and calculate hamiltonian 
        accordingly.

        `set_field` will set rotor.field to input field object and 
        change self.hamiltonian to the new hamiltonian based on the 
        new field. This DOES NOT add a new value to history but 
        instead change the last value of history, and hence should 
        only be used to set the initial field when a rotor object 
        instantiated within a solver. (solvers.FieldToPath or 
        solvers.PathToField)
    
        Parameters
        ----------
        field: numpy.array, shape=(2,)
            New external field expressed as (e_x, e_y) to set to the 
            molecule.

        """

        self.field = field
        self.hamiltonian = self._get_hamiltonian()
        # rewrite history manually
        self.history['field'][-1] = field

    def update_time(self, time):
        """Set and update time of molecule with history appended.

        Parameters
        ----------
        time: float
            Current time.

        """
        self.time = time
        self.history['time'].append(time)

    def update_state(self, state):
        """Set and update state of molecule with history appended.

        Parameters
        ----------
        state: State object
            Current state of the molecule.

        """
        self.state = state
        self.history['state'].append(state)

    def update_field(self, field):
        """Set and update field of molecule with history appended and 
        hamiltonian recalculated.

        Parameters
        ----------
        field: numpy.array, shape=(2,)
            New external field expressed as (e_x, e_y) to set to the 
            molecule.

        """

        self.field = field
        self.history['field'].append(field)
        #calculate and set the new hamiltonian
        self.hamiltonian = self._get_hamiltonian()

    def get_time_asarray(self):
        """Return history of time as an array.

        Returns
        -------
        times: numpy.array, shape=(n,)
            Array containing n time points.

        """
        times = [time for time in self.history['time']]
        return np.stack(times)

    def get_states_asarray(self):
        """Return history of state as an array.

        Returns
        -------
        states: numpy.array, shape=(2m+1,n)
            State amplitudes of the molecule at each time points. 
            Each column of this returned 2D-array is a state 
            amplitudes vector.

        """
        states = [state.value.flatten() for state in self.history['state']]
        return np.stack(states, axis=1)

    def get_fields_asarray(self):
        """Return history of field as an array.

        Returns
        -------
        fields: numpy.array, shape=(n,2)
            Control fields to apply to the molecule. Each row is a 
            field described as (e_x,e_y) at a time point.

        """
        fields = [field.flatten() for field in self.history['field']]
        return np.stack(fields, axis=0)











