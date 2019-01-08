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
        print(len(rawtrack))
        rawtrack[:,0] = rawtrack[:,0] - rawtrack[0,0]
        rawtrack[:,1] = rawtrack[:,1] - rawtrack[0,1]
        new_path,dt = transform_path(rawtrack)
        self.assertTrue(len(new_path)>len(rawtrack))
        print(new_path[0,0])

        file_verify = 'transform_verify.dat'
        verify_path = np.genfromtxt(file_verify)
        self.assertAlmostEqual(new_path[-1,0],verify_path[-1,0])
        #self.assertAlmostEqual(new_path[:,0].any,verify_path[:,0].any)


if __name__ == '__main__':
    unittest.main()



