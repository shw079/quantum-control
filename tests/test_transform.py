'''testFunctions.py
'''
import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), "modules"))
import unittest
import numpy as np
import functions as f
import constants as const
from transform import transform_path


class test_Transform(unittest.TestCase):

    def test_transform(self):
        file = 'transform_data.dat'
        rawtrack = np.genfromtxt(file,delimiter=",")
        rawtrack[:,0] = rawtrack[:,0] - rawtrack[0,0]
        rawtrack[:,1] = rawtrack[:,1] - rawtrack[0,1]
        new_path,dt = transform_path(rawtrack)
        #self.assertEqual(np.isnan(new_path),False)
        self.assertTrue(len(new_path)>len(rawtrack))

        file_verify = 'transform_verify.dat'
        verify_path = np.genfromtxt(file_verify)
        diff = new_path[:,0]-verify_path[:,0]
        test = diff < 1e-6
        self.assertEqual(True, np.all(test))
         
if __name__ == '__main__':
    unittest.main()



