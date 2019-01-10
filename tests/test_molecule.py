'''!@namespace testing.testMolecule

@brief Unittests for molecule.py

'''

import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), "modules"))
import unittest
import numpy as np
from molecule import Rotor
from state import State
import constants as const

class test_molecule(unittest.TestCase):
    """!@brief Testing class for abstract base class Molecules.

    """

    def setUp(self):
        """!@brief Instantiate rotor object with quantum number m."""

        self.rotor = Rotor(const.m)

    def test_rotor_init(self):
        """!@brief Test rotor init function with state array generated."""

        state_prob = np.zeros(2*const.m+1)
        state_prob[const.m] = 1
        np.testing.assert_array_equal(self.rotor.state.value, state_prob)

    def test_update_attr(self):
        """!@brief Test update functions to update attributes of Rotor"""

        self.rotor.update_time(1.0)
        self.rotor.update_state(State(const.m))
        self.rotor.update_field(np.array([1, 1]))
        for attr in self.rotor.history:
            self.assertTrue(len(self.rotor.history[attr]) == 2)

    def test_evolve(self):
        """!@brief Test evolve function over 5 timesteps and the corresponding
        history of state array generated.
        
        """

        for i in range(5):
            self.rotor.evolve(0.1)
        self.assertTrue(len(self.rotor.history['state']) == 6)

    def test_get_history_asarray(self):
        """!@brief Test function to return history of states as array."""

        for i in range(5):
            self.rotor.evolve(0.1)
            self.rotor.update_field(np.array([0,0]))
        self.assertEqual(self.rotor.get_states_asarray().shape, (2*const.m+1,6))
        self.assertEqual(self.rotor.get_fields_asarray().shape, (6,2))


if __name__ == '__main__':
    unittest.main()
