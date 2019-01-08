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
        self.assertAlmostEqual(new_path[0,-1],0.4548999818)


if __name__ == '__main__':
    unittest.main()



