'''!@namespace testing.testFunctions

@brief Unittests for functions.py

'''

import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), "modules"))
import unittest
import numpy as np
import functions as f
import constants as const


class test_Functions(unittest.TestCase):
    """!@brief Testing class for functions.py."""

    def test_calc_cosphi(self):
        """!@brief check cosphi function"""
        m=8
        cosphi = f.cosphi(m)
        self.assertEqual(cosphi.shape,(2*m+1,2*m+1))
        for i in range(0,2*m):
            self.assertEqual(cosphi[i,i+1],0.5)
            self.assertEqual(cosphi[i+1,i],0.5)

    def test_calc_sinphi(self):
        """!@brief check sinphi function"""
        m=8
        sinphi = f.sinphi(m)
        self.assertEqual(sinphi.shape,(2*m+1,2*m+1))
        for i in range(0,2*m):
            self.assertEqual(sinphi[i,i+1],0.+0.5j)
            self.assertEqual(sinphi[i+1,i],0.-0.5j)

    def test_calc_ddphi(self):
        """!@brief check cosphi function"""
        m=8
        ddphi = f.ddphi(m)
        self.assertEqual(ddphi.shape,(2*m+1,2*m+1))
        for i in range(0,2*m+1):
            self.assertEqual(ddphi[i,i],(i-8)*1j)

    def test_calc_d2dpi2(self):
        """!@brief check d2dphi2 function"""
        m=8
        d2dphi2 = f.d2dphi2(m)
        self.assertEqual(d2dphi2.shape,(2*m+1,2*m+1))
        for i in range(0,2*m+1):
            self.assertEqual(d2dphi2[i,i],-1*abs(i-8)**2)

    def test_d2dt2(self):
        """!@brief check d2di2 function"""
        dt = 1
        xi,xf = 0,10
        n = int((xf-xi)/dt) + 1
        # f(x) = x^2
        x = np.power(np.linspace(xi,xf, num=n, dtype=float),2)
        ddx_truth = 2 * np.ones(n, dtype=float)
        ddx = f.d2dt2(x,dt)
        np.testing.assert_array_almost_equal(ddx, ddx_truth)

    def test_d2dt2_sigmoid(self):
        """!@brief check d2di2 function for sigmoid path"""
        #import path and expected result
        fname_path = 'testdata_solver/sigmoid_path.txt'
        fname_d2dt2 = 'testdata_solver/sigmoid_path_d2dt2.txt'
        path = np.genfromtxt(fname_path, dtype=float, delimiter=',')
        d2dt2_truth = np.genfromtxt(fname_d2dt2, dtype=float, delimiter=',')

        #determine dt
        n = 100000
        rotor_period = 2*np.pi*const.hbar/const.B
        t_final = 100*rotor_period/(2*np.pi)
        dt = t_final/n

        #calculate second derivative using f.d2dt2()
        d2dt2 = np.stack( (f.d2dt2(path[:,0],dt), f.d2dt2(path[:,1],dt)) , axis=1)

        #compare calculated result with expected result
        np.testing.assert_array_almost_equal(d2dt2, d2dt2_truth)



if __name__ == '__main__':
    unittest.main()



