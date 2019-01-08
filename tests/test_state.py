'''test_State.py
'''
import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), "modules"))
import unittest
import numpy as np
import functions as f
import constants as const
from state import State


class test_State(unittest.TestCase):

    # def test_get_value(self):
    #     """check constructor and get_value"""
    #     state_input = np.arange(0,2*const.m+1)
    #     state_obj = State(const.m,state_input)
    #     np.testing.assert_array_equal(state_obj.get_value(), state_input)
    
    def test_init(self):
        """Init with or without specified state probability."""
        m = 5
        value = np.zeros(2*m+1)
        state = State(m)
        np.testing.assert_array_equal(state.value, value)

        value = np.ones(2*m+1)
        state = State(m, value)
        np.testing.assert_array_equal(state.value, value)

    def test_init_input(self):
        m = 5
        #correct input
        value = np.ones(2*m+1)
        state = State(m, value)
        np.testing.assert_array_equal(state.value, value)
        #incorrect input type
        value = [1]*(2*m+1)
        self.assertRaises(TypeError, State, m, value)
        #incorrect input size
        value = np.ones(m)
        self.assertRaises(ValueError, State, m, value)

    def test_calc_bra(self):
    	"""calculate bra or complex conjugate of input state"""
    	state_input = (np.arange(0,2*const.m+1))*1j
    	state_obj = State(const.m,state_input)
    	bra = state_obj.as_bra()
    	self.assertEqual(bra.shape,(1,2*const.m+1))
    	np.testing.assert_array_equal(bra,np.conj(state_input.reshape(1,2*const.m+1)))

    def test_calc_ket(self):
    	"""calculate ket (reshaped array of state) of input state"""
    	state_input = (np.arange(0,2*const.m+1))*1j
    	state_obj = State(const.m,state_input)
    	ket = state_obj.as_ket()
    	self.assertEqual(ket.shape,(2*const.m+1,1))
    	np.testing.assert_array_equal(ket,state_input.reshape(2*const.m+1,1))

    def test_expt(self):
    	"""calculate expectation value from bra, ket, and input operator"""
    	state_input = (np.arange(0,2*const.m+1))*1j
    	state_obj = State(const.m,state_input)
    	bra = state_obj.as_bra()
    	ket = state_obj.as_ket()
    	expectation = state_obj.get_expt(f.cosphi(const.m))
    	expected_result = bra@f.cosphi(const.m)@ket
    	np.testing.assert_array_equal(expectation,expected_result)



if __name__ == '__main__':
    unittest.main()



