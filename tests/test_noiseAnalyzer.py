import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), "modules"))
import unittest
from noiseAnalyzer import NoiseAnalyser
import numpy as np

#import other stuff you need

class test_noiseAnalyzer(unittest.TestCase):
    def test_init(self):
        """Test to create a instance of NoioseAnalyser"""
        # do your test
        input_field = np.arange(10).reshape((5,2))
        dt = 1000
        myNA = NoiseAnalyser(input_field, dt, 1, 10)
        self.assertTrue(myNA.dt == dt)

    def test_2(self):
        input_field = np.arange(10).reshape((5,2))
        dt = 1000
        myNA = NoiseAnalyser(input_field, dt, 0.0000001, 1)
        myNA.calc_noisy_field()
        np.testing.assert_array_almost_equal(myNA.noisy_field,input_field) 
    
    def test_3(self):
        input_path=np.arange(10).reshape((5,2))
        input_field = np.arange(10).reshape((5,2))
        dt = 1000
        myNA = NoiseAnalyser(input_field, dt, 1, 1)
        myNA.path=input_path
        myNA.calc_statistic()
        np.testing.assert_array_equal( myNA.pathmean,input_path)
        np.testing.assert_array_equal( myNA.pathvar,np.zeros((5,2)))
        
if __name__ == '__main__':
    unittest.main()
 
#self.assertTrue()
#self.assertEqual(arg1 ,arg2 )
#self.assertAlmostEqual(number1,number2)
#np.testing.assert_array_almost_equal(array1,array2)
#np.testing.assert_array_equal(arr1,arr2)        
