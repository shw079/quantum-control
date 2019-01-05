'''test_solver.py
'''

import numpy as np
import unittest
from .state import State
from . import functions as f
from . import observable as obs
from . import constants as const
from . import solvers as s
from .molecule import Rotor

class test_PathToField(unittest.TestCase):
    def setUp(self):
        self.rotor = Rotor(const.m)

    def test_init(self):
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
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        det = fsolver._get_det()
        self.assertTrue(np.isscalar(det))

    def test_solve(self):
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        fsolver.solve()

class test_PathToField_sigmoid_path(unittest.TestCase):
    def setUp(self):
        #calculate time points
        n = 100000
        rotor_period = 2*np.pi*const.hbar/const.B
        self.t_final = 100*rotor_period/(2*np.pi)
        self.dt = self.t_final/n
        self.time = np.arange(self.t_final, step=self.dt, dtype=float)
        #import sigmoid path from text file as np.ndarray with shape (n,2)
        fname_path = 'testdata/sigmoid_path.txt'
        self.path_desired = np.genfromtxt(fname_path, dtype=float, delimiter=',')
        #import expected resulting path as np.ndarray with shape (n,2)
        fname_path_predicted_truth = 'testdata/sigmoid_path_result.txt'
        self.path_predicted_truth = np.genfromtxt(fname_path_predicted_truth, dtype=float, delimiter=',')
        #import expected fields as np.ndarray with shape (n,2)
        fname_field = 'testdata/fields_real_for_sigmoid_path.txt'
        self.fields_truth = np.genfromtxt(fname_field, dtype=float, delimiter=',')
        #import expected states as np.ndarray with shape (2m+1,n)
        fname_state = 'testdata/states_for_sigmoid_path.txt'
        self.states_truth = np.genfromtxt(fname_state, dtype=float, delimiter=',')

        #instantiate a rotor molecule
        self.rotor = Rotor(const.m)

    def test_solve_early(self):
        #only do the first 10 time points
        n = 50
        self.time = self.time[0:n]
        self.fields_truth = self.fields_truth[0:n,:]
        self.states_truth = self.states_truth[:,0:n]
        self.path_desired = self.path_desired[0:n,:]
        self.path_predicted_truth = self.path_predicted_truth[0:n,:]

        fsolver = s.PathToField(self.path_desired, dt=self.dt)
        fsolver.solve()
        time, fields, states = fsolver.export()

        np.testing.assert_array_almost_equal(fields, self.fields_truth)
        np.testing.assert_array_almost_equal(states, self.states_truth)
        prob_sum = states.sum(axis=0)
        np.testing.assert_array_almost_equal(prob_sum, np.ones(n))

    @unittest.skip("Doesn't match MATLAB result around n=70")
    def test_solve_long(self):
        fsolver = s.PathToField(self.path_desired, t_final=self.t_final)
        fsolver.solve()

class test_FieldToPath(unittest.TestCase):
    def setUp(self):
        self.rotor = Rotor(const.m)

    def test_init(self):
        n = 5
        fields = np.arange(2*n, dtype=float).reshape((n,2))
        psolver = s.FieldToPath(fields)
        self.assertEqual(psolver.n, n)
        np.testing.assert_array_almost_equal(psolver.fields, fields)
        self.assertEqual(len(psolver._fields_list),n)
        self.assertEqual(psolver.path.shape, (n,2))

    def test_init_with_dt(self):
        n = 5
        fields = np.arange(2*n, dtype=float).reshape((n,2))
        dt = 20
        psolver = s.FieldToPath(fields, dt=dt)
        self.assertEqual(psolver._t_final, dt*n)
        np.testing.assert_array_almost_equal(psolver.time, np.array([0., 20., 40., 60., 80.]))

class test_FieldToPath_sigmoid_path(unittest.TestCase):
    def setUp(self):
        self.rotor = Rotor(const.m)

        fname_fields = 'testdata/fields_real_for_sigmoid_path.txt'
        fname_states = 'testdata/states_for_sigmoid_path.txt'
        self.fields = np.genfromtxt(fname_fields, dtype=float, delimiter=',')
        self.states_expected = np.genfromtxt(fname_states, dtype=float, delimiter=',')

        n = 100000
        rotor_period = 2*np.pi*const.hbar/const.B
        self.t_final = 100*rotor_period/(2*np.pi)
        self.dt = self.t_final/n
        self.time = np.arange(self.t_final, step=self.dt, dtype=float)

    def test_solve_short(self):
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
        












