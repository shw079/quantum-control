'''testDataStore.py
'''
import unittest
import numpy as np
import functions as f


class test_Functions(unittest.TestCase):

    def test_calc_cosphi(self):
        """check cosphi function"""
        m=8
        cosphi = f.cosphi(m)
        self.assertEqual(cosphi.shape,(2*m+1,2*m+1))
        for i in range(0,2*m):
            self.assertEqual(cosphi[i,i+1],0.5)
            self.assertEqual(cosphi[i+1,i],0.5)

    def test_calc_sinphi(self):
        """check sinphi function"""
        m=8
        sinphi = f.sinphi(m)
        self.assertEqual(sinphi.shape,(2*m+1,2*m+1))
        for i in range(0,2*m):
            self.assertEqual(sinphi[i,i+1],0.+0.5j)
            self.assertEqual(sinphi[i+1,i],0.-0.5j)

    def test_calc_ddphi(self):
        """check cosphi function"""
        m=8
        ddphi = f.ddphi(m)
        self.assertEqual(ddphi.shape,(2*m+1,2*m+1))
        for i in range(0,2*m+1):
            self.assertEqual(ddphi[i,i],(i-8)*1j)

    def test_calc_d2dpi2(self):
        """check d2dphi2 function"""
        m=8
        d2dphi2 = f.d2dphi2(m)
        self.assertEqual(d2dphi2.shape,(2*m+1,2*m+1))
        for i in range(0,2*m+1):
            self.assertEqual(d2dphi2[i,i],-1*abs(i-8)**2)


if __name__ == '__main__':
    unittest.main()



