'''testDataStore.py
'''
import unittest
import numpy as np
import functions as f
from dataStore import DataStore
from fieldSolver import FieldSolver
import constants as const


class test_Functions(unittest.TestCase):

    def test_calcOperators(self):
        """check all operator calculator"""
        m=8
        cosphi,sinphi,ddphi,d2dphi2 = f.calcOperators(m)
        self.assertEqual(cosphi.shape,(2*m+1,2*m+1))
        self.assertEqual(sinphi.shape,(2*m+1,2*m+1))
        self.assertEqual(ddphi.shape,(2*m+1,2*m+1))
        self.assertEqual(d2dphi2.shape,(2*m+1,2*m+1))
        # more tests to verify operators arrays?
    
    def test_d2dt2(self):
        """test second derivative of path calculator using parameters of sigmoid"""
        input_path=np.array([[0,1.29935724140331e-46,8.76682606408199e-46,3.32719861353700e-45,9.97723055524127e-45],
        [0,5.37368575715481e-43,1.81282576109439e-42,4.58670806443427e-42,1.03155784161377e-41]]).T
        dt = 21621621.6216216
        d2x,d2y = f.d2dt2(input_path,dt)
        expected_d2x = 1.31939761793218e-60
        expected_d2y = 1.57881766660818e-57
        self.assertAlmostEqual(d2x[0],expected_d2x)
        self.assertAlmostEqual(d2y[0],expected_d2y)



if __name__ == '__main__':
    unittest.main()



