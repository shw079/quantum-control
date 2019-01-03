'''testMolecule.py
'''

import unittest
import numpy as np
from molecule import Rotor
from state import State
from field import Field
import observable as obs
import constants as const

class test_molecule(unittest.TestCase):
    def setUp(self):
        self.rotor = Rotor(const.m)

    def test_rotor_init(self):
        state_prob = np.zeros(2*const.m+1)
        state_prob[const.m] = 1
        np.testing.assert_array_equal(self.rotor.state.value, state_prob)

    def test_update_attr(self):
        self.rotor.update_time(1.0)
        self.rotor.update_state(State(const.m))
        self.rotor.update_field(Field(np.array([1, 1])))
        for attr in self.rotor.history:
            self.assertTrue(len(self.rotor.history[attr]) == 2)

    def test_evolve(self):
        for i in range(5):
            self.rotor.evolve(0.1)
        self.assertTrue(len(self.rotor.history['state']) == 6)

    def test_get_history_asarray(self):
        for i in range(5):
            self.rotor.evolve(0.1)
            self.rotor.update_field(Field(np.array([0,0])))
        self.assertEqual(self.rotor.get_states_asarray().shape, (2*const.m+1,6))
        self.assertEqual(self.rotor.get_fields_asarray().shape, (6,2))


if __name__ == '__main__':
    unittest.main()