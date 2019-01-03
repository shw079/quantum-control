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

    # @abc.abstractmethod
    # def set_field(self):
    #     """Let user set value of field and then update Hamiltonian"""
    #     pass

    # @abc.abstractmethod
    # def _recalc_Hamiltonian(self):
    #     pass

    # @abc.abstractmethod
    # def _update_history(self, attr):
    #     """Aggregate newly assigned value for a specific attribute to its history"""
    #     pass

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
        self.update_attr('state', state_new)
        self.update_attr('time', self.time+dt)

    def update_attr(self, attr, value):
        setattr(self, attr, value)
        self.history[attr].append(value)










