import abc
import numpy as np
from state import State
from field import Field
import observable as obs

class Molecule(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def evolve(self):
        """Evolve the state to next time point by a unitary operator"""
        pass

class Rotor(Molecule):
    def __init__(self, m):
        self.m = m
        ground_state = np.zeros(2*m+1)
        ground_state[m] = 1.0
        self.state = State(m, ground_state)
        self.field = Field(np.zeros(2))
        self.hamiltonian = obs.RotorH(m, self.field)
        self.time = 0.0
        self.history = {'time':[self.time],
                        'state':[self.state],
                        'field':[self.field]}

    def evolve(self, dt):
        #Use haniltonian to evolve the current state 
        state_new = self.hamiltonian.evolve(self.state, dt)
        self.update_state(state_new)
        self.update_time(self.time+dt)

    def update_time(self, value):
        self.time = value
        self.history['time'].append(value)

    def update_state(self, state):
        self.state = state
        self.history['state'].append(state)

    def update_field(self, field):
        self.field = field
        self.history['field'].append(field)
        #calculate and set the new hamiltonian
        self.hamiltonian = obs.RotorH(self.m, field)

    def get_states_asarray(self):
        states = [state.value.flatten() for state in self.history['state']]
        return np.stack(states, axis=1)

    def get_fields_asarray(self):
        fields = [field.value.flatten() for field in self.history['field']]
        return np.stack(fields, axis=0)











