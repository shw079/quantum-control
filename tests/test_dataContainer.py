'''test_dataContainer.py
'''

import sys
sys.path.append('../modules')
import unittest
import numpy as np
from dataContainer import DataContainer

class test_DataContainer(unittest.TestCase):

    def test_init(self):
        """Instantiate DataContainer object with input using fake path."""
        #create a fake yet correct input path which is a 5x2 ndarray
        path = np.array([np.arange(5), np.arange(1,6)]).T
        data = DataContainer(path)
        self.assertEqual(data.n, path.shape[0])
        np.testing.assert_array_equal(data.path_desired, path)
        self.assertEqual(data.path_actual.shape, path.shape)

    def test_hasNanInf(self):
        """Raise ValueError if input path contains NaN or Inf"""
        path = np.array([[0.], [np.nan]])
        self.assertRaises(ValueError, DataContainer, path)
        path2 = np.array([[1.], [np.inf]])
        self.assertRaises(ValueError, DataContainer, path2)

    def test_inputTypeAndShape(self):
        """Raise errors if input path is not n-by-3 numpy ndarray."""
        #an input with wrong type
        path = [0., 1., 2.]
        self.assertRaises(TypeError, DataContainer, path)
        #an input with wrong dimensions
        path = np.array([0., 1., 2.])
        self.assertRaises(ValueError, DataContainer, path)
        #an input with wrong shape
        path = np.arange(4).reshape((1,4))
        self.assertRaises(ValueError, DataContainer, path)


if __name__ == '__main__':
    unittest.main()
