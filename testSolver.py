'''testSolver.py
'''

import numpy as np
import unittest
from state import State
import functions as f
import operators as op
import constants as const
import solver as s

class test_PathToField(unittest.TestCase):
    def test_init(self):
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)

    def test_get_det(self):
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        det = fsolver._get_det(fsolver.states[0])
        self.assertTrue(np.isscalar(det))

    def test_calc_state_and_field(self):
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        fsolver.calc_state_and_field()
