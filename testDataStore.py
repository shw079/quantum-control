'''testDataStore.py
'''

import unittest
import numpy as np
import constants as const
from dataStore import DataStore

class test_DataStore(unittest.TestCase):

    '''def test_noInput(self):
        """Instantiate DataStore object without input."""
        data = DataStore()
        #check if attr Const exists
        self.assertIn('Const',vars(data))
        '''

    def test_withInput(self):
        """Instantiate DataStore object with input using fake txy_desired."""
        #create a fake yet correct input txy_desired which is a 5x2 ndarray
        txy_desired = np.array([np.arange(5), np.arange(1,6)]).T
        data = DataStore(txy_desired)
        self.assertEqual(data.n, txy_desired.shape[0])
        np.testing.assert_array_equal(data.t, txy_desired[:,0])
        np.testing.assert_array_equal(data.path_desired, txy_desired)
        self.assertEqual(data.path_obs.shape, txy_desired.shape)
        self.assertEqual(data.state.shape, (2*const.m+1, data.n))
        self.assertEqual(len(data.noise_stat), 2)
        for key in data.noise_stat:
            self.assertEqual(data.noise_stat[key].shape, (data.n, 2))

    def test_timeOrder(self):
        """Raise ValueError if time provided not strictly increasing."""
        #an input with acceptable time points
        txy_desired = np.array([np.arange(5), np.arange(1,6)]).T
        data = DataStore(txy_desired)
        #an input with non-acceptable time points
        txy_desired2 = np.array([ [0,1,2,4,3], np.arange(1,6) ] ).T
        self.assertRaises(ValueError, DataStore, txy_desired2)

    def test_hasNanInf(self):
        """Raise ValueError if input txy_desired contains NaN or Inf"""
        txy_desired = np.array([[0.], [np.nan], ])
        self.assertRaises(ValueError, DataStore, txy_desired)
        txy_desired2 = np.array([[1.], [np.inf]])
        self.assertRaises(ValueError, DataStore, txy_desired2)

    def test_inputTypeAndShape(self):
        """Raise errors if input txy_desired is not n-by-3 numpy ndarray."""
        #an input with wrong type
        txy_desired = [0., 1., 2.]
        self.assertRaises(TypeError, DataStore, txy_desired)
        #an input with wrong dimensions
        txy_desired = np.array([0., 1., 2.])
        self.assertRaises(ValueError, DataStore, txy_desired)
        #an input with wrong shape
        txy_desired = np.arange(4).reshape((1,4))
        self.assertRaises(ValueError, DataStore, txy_desired)


if __name__ == '__main__':
    unittest.main()
