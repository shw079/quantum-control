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
    def test_rotor_init(self):
        rotor = Rotor(const.m)
        state_prob = np.zeros(2*const.m+1)
        state_prob[const.m] = 1
        np.testing.assert_array_equal(rotor.state.value, state_prob)

    def test_update_attr(self):
        rotor = Rotor(const.m)
        rotor.update_attr('time', 1.0)
        rotor.update_attr('state', State(const.m))
        rotor.update_attr('field', Field(np.array([1, 1])))
        for attr in rotor.history:
            self.assertTrue(len(rotor.history[attr]) == 2)

    def test_evolve(self):
        pass




if __name__ == '__main__':
    unittest.main()