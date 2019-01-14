'''Unittets for solvers.py

'''

import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), "modules"))
import numpy as np
import unittest
from state import State
import functions as f
import constants as const
import solvers as s
from molecule import Rotor

class test_PathToField(unittest.TestCase):
    """Testing class for class PathtoField in abstract base 
    class Solver.
    
    """

    def setUp(self):
        """instantiation of Rotor object with quantum
        number m.
        
        """

        self.rotor = Rotor(const.m)

    def test_init(self):
        """test init constructor with path input, dt, shape
        of predicted path, and operators used internally within 
        class instance
        
        """

        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        #maybe more?
        
        m = const.m
        #check _cosphi2
        self.assertEqual(fsolver._cosphi2.shape, (2*m+1,2*m+1))
        for i in range(2*m+1):
            if i is 0 or i is 2*m:
                self.assertEqual(fsolver._cosphi2[i,i],0.25)
            else:
                self.assertEqual(fsolver._cosphi2[i,i],0.5)
        for i in range(2*m-1):
            self.assertEqual(fsolver._cosphi2[i,i+2],0.25)
            self.assertEqual(fsolver._cosphi2[i+2,i],0.25)

        #check _sinphi2
        self.assertEqual(fsolver._sinphi2.shape, (2*m+1,2*m+1))
        for i in range(2*m+1):
            if i is 0 or i is 2*m:
                self.assertEqual(fsolver._sinphi2[i,i],0.25)
            else:
                self.assertEqual(fsolver._sinphi2[i,i],0.5)
        for i in range(2*m-1):
            self.assertEqual(fsolver._sinphi2[i,i+2],-0.25)
            self.assertEqual(fsolver._sinphi2[i+2,i],-0.25)


    def test_get_det(self):
        """Tests for private function _get_det to solve
        for determinant of matrix A
        
        """

        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        det = fsolver._get_det()
        self.assertTrue(np.isscalar(det))

    def test_solve(self):
        """Tests solve function to calculate the control
        field required for a time step
        
        """

        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        fsolver.solve()

class test_PathToField_sigmoid_path(unittest.TestCase):
    """Testing class for class PathtoField in abstract base 
    class Solver for a particular know given path: a sigmoid path.
    
    """

    def setUp(self):
        """initiate variables and obtain sigmoid path for 
        testing.

        ***This is no longer needed?***

        """

        #calculate time points
        n = 100000
        rotor_period = 2*np.pi*const.hbar/const.B
        self.t_final = 100*rotor_period/(2*np.pi)
        self.dt = self.t_final/n
        self.path_desired = self._get_sigmoid_path()
        self.time = np.arange(self.t_final, step=self.dt, dtype=float)

    def _get_sigmoid_path(self):
        """Generate sigmoid path as input for solver and 
        for solved path checking.
        
        """

        tf = self.t_final
        dt = self.dt
        t = np.arange(0, tf, step=dt, dtype=float)
        Q = 0.25*tf
        B2 = 0.0002
        v = 0.2
        sigmoid = 1/((1+Q*np.exp(-B2*t))**(1/v))
        path_x = 0.95*(t/tf)*np.sin(0.5*const.w1*t)*sigmoid
        path_y = 0.95*(t/tf)*np.cos(0.5*const.w1*t)*sigmoid
        path = np.stack( (path_x, path_y), axis=1 )

        m = const.m
        psi0 = np.zeros(2*m+1)
        psi0[m] = 1.0
        psi0_ket = psi0.reshape((2*m+1,1))
        psi0_bra = psi0_ket.conj().transpose()
        dx = np.asscalar(psi0_bra @ f.cosphi(m) @ psi0_ket)
        dy = np.asscalar(psi0_bra @ f.sinphi(m) @ psi0_ket)
        path[:,0] = path[:,0] - (path[0,0] - dx).real
        path[:,1] = path[:,1] - (path[0,1] - dy).real

        return path

    def test_solve_early(self):
        """Given a path, solve all fields, states, and paths. 
        Test the solved path with the initial input path for only the
        first 10 time points.
        
        """

        #only do the first 10 time points
        n = 10
        self.time = self.time[0:n]
        self.path_desired = self.path_desired[0:n,:]

        fsolver = s.PathToField(self.path_desired, dt=self.dt)
        fsolver.solve()
        time, fields, path, states = fsolver.export()

        # # Test weights of state add up to 1 at each time point
        # weights_sum = states.sum(axis=0)
        # np.testing.assert_array_almost_equal(weights_sum, np.ones(n))

        # Test path returned match path desired
        np.testing.assert_array_almost_equal(path.real, self.path_desired)

    @unittest.skip("Doesn't match MATLAB result around n=70")
    def test_solve_long(self):
        fsolver = s.PathToField(self.path_desired, t_final=self.t_final)
        fsolver.solve()

class test_FieldToPath(unittest.TestCase):
    """Testing class for class FieldToPath in abstract base 
    class Solver.
    
    """

    def setUp(self):
        """instantiation of Rotor object with quantum
        number m.
        
        """

        self.rotor = Rotor(const.m)

    def test_init(self):
        """test init constructor with fields input, calculated
        path shape.
        
        """

        n = 5
        fields = np.arange(2*n, dtype=float).reshape((n,2))
        psolver = s.FieldToPath(fields)
        self.assertEqual(psolver.n, n)
        field_const = field_const = 5.142 * 10**11 * 10**(-10)
        np.testing.assert_array_almost_equal(psolver.fields, 
                                             fields/field_const)
        self.assertEqual(len(psolver._fields_list),n)
        self.assertEqual(psolver.path.shape, (n,2))

    def test_init_with_dt(self):
        """tested __init__ function with dt given"""

        n = 5
        fields = np.arange(2*n, dtype=float).reshape((n,2))
        dt = 20
        psolver = s.FieldToPath(fields, dt=dt)
        self.assertEqual(psolver._t_final, dt*n)
        np.testing.assert_array_almost_equal(psolver.time, np.array([0., 20., 40., 60., 80.]))

class test_FieldToPath_sigmoid_path(unittest.TestCase):
    """Testing class for class FieldToPath in abstract base 
    class Solver for a particular know given path: a sigmoid path.
    
    """

    def setUp(self):
        """init system with test data for sigmoid"""

        self.rotor = Rotor(const.m)

        fname_fields = 'testdata_solver/fields_real_for_sigmoid_path.txt'
        fname_states = 'testdata_solver/states_for_sigmoid_path.txt'
        self.fields = np.genfromtxt(fname_fields, dtype=float, delimiter=',')
        self.states_expected = np.genfromtxt(fname_states, dtype=float, delimiter=',')

        n = 100000
        rotor_period = 2*np.pi*const.hbar/const.B
        self.t_final = 100*rotor_period/(2*np.pi)
        self.dt = self.t_final/n
        self.time = np.arange(self.t_final, step=self.dt, dtype=float)

    def test_solve_short(self):
        """Test solver in FieldToPath for time, states, and path
        output.
        
        """

        n = 50
        self.t_final = self.time[n]
        self.time = self.time[0:n]
        self.fields = self.fields[0:n,:]
        self.states_expected = self.states_expected[:,0:n]

        psolver = s.FieldToPath(self.fields, dt=self.dt)
        psolver.solve()
        time, path, states = psolver.export()
        self.assertEqual(time.shape, (n,))
        self.assertEqual(states.shape, (2*const.m+1,n))
        self.assertEqual(path.shape, (n,2))
        np.testing.assert_array_almost_equal(time, self.time)
        np.testing.assert_array_almost_equal(states, self.states_expected)

if __name__ == '__main__':
    unittest.main()
        












