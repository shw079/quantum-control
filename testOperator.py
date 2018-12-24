'''testOperator.py
'''

import unittest
import numpy as np
import constants as const
from operators import *
from state import State
import functions as f
from solver import *
from scipy import linalg

# class test_Path(unittest.TestCase):

#     def test_path_constructor(self):
#         """test path constructor"""
#         path_obj = Path()
#         np.testing.assert_array_equal(path_obj.operator,[f.cosphi(const.m),f.sinphi(const.m)])

#     def test_set_value(self):
#     	"""test setting path values"""
#     	path_obj = Path()
#     	xy = [1,2]
#     	xy_input = path_obj.set_value(xy)
#     	np.testing.assert_array_equal(path_obj.value,xy)

#     def test_act_on_state(self):
#     	"""test operating on state to calculate path"""
#     	state_input = (np.arange(0,2*const.m+1))*1j
#     	state_obj = State(state_input,const.m)
#     	path_obj = Path()
#     	expectation = path_obj.act_on_state(state_obj)
#     	expected_x = np.conj(state_input).reshape((1,2*const.m+1))@path_obj.operator[0]@state_input.reshape((2*const.m+1,1))
#     	expected_y = np.conj(state_input).reshape((1,2*const.m+1))@path_obj.operator[1]@state_input.reshape((2*const.m+1,1))
#     	np.testing.assert_array_equal(path_obj.get_value(),[expected_x.item(0),expected_y.item(0)])

class test_Dipole(unittest.TestCase):

    def test_dipoleX(self):
        dipoleX_obj = DipoleX()
        m=8
        expected_dipoleX = f.cosphi(m)
        np.testing.assert_array_equal(dipoleX_obj.operator,expected_dipoleX)

    def test_dipoleY(self):
        dipoleY_obj = DipoleY()
        m=8
        expected_dipoleY = f.sinphi(m)
        np.testing.assert_array_equal(dipoleY_obj.operator,expected_dipoleY)

class test_RotorH(unittest.TestCase):

    '''def test_H_operator(self):
        """test path constructor"""
        n=10
        t = 0
        field_input = np.arange(0,2)
        #field_obj = Field(n)
     	path = [1,2]
        field_obj = PathToField(n)
        field_obj.set_value(t,field_input)
        field = field_obj.get_value()
        rotor_obj = RotorH(field[0,:])
        expected_H = const.B*np.diag((np.arange(-const.m,const.m+1))**2,k=0)-const.mu*f.cosphi(const.m)*field[0,0]-const.mu*f.sinphi(const.m)*field[0,1]
        np.testing.assert_array_equal(rotor_obj.operator,expected_H)
        '''


        

    # def test_act_on_state(self):
    # 	"""test operating in state to obtain new state"""
    # 	# generate state object
    # 	state_input = (np.arange(0,2*const.m+1))*1j
    # 	state_obj = State(state_input,const.m)
    # 	# generate rotor object
    # 	n=10
    # 	t = 0
    # 	field_input = np.arange(0,2)
    # 	field_obj = Field(n)
    # 	field_obj.set_value(t,field_input)
    # 	field = field_obj.get_value()
    # 	rotor_obj = RotorH(field[0,:])
    # 	# act on state function called
    # 	rotor_obj.act_on_a_state(state_obj,t,t+1)
    # 	expected_operator = const.B*np.diag((np.arange(-const.m,const.m+1))**2,k=0)-const.mu*f.cosphi(const.m)*field[0,0]-const.mu*f.sinphi(const.m)*field[0,1]
    # 	expected_new_state = (linalg.expm((-1j/const.hbar)*expected_operator*(t+1)))@state_obj.get_value()
    # 	self.assertEqual(len(rotor_obj.get_value()),2*const.m+1)
    # 	np.testing.assert_array_equal(rotor_obj.get_value(),expected_new_state)

    def test_evolve(self):
    	"""test evolving system by passing along newly calculated state"""
    	# generate state object
    	state_input = (np.arange(0,2*const.m+1))*1j
    	state_obj = State(const.m,state_input)
    	# generate rotor object
    	n=10
    	t = 0
    	field = np.arange(0,2)
    	rotor_obj = RotorH(field)
    	# act on state function called
    	new_state = rotor_obj.evolve(state_obj,1)
    	expected_operator = const.B*np.diag((np.arange(-const.m,const.m+1))**2,k=0)-const.mu*f.cosphi(const.m)*field[0]-const.mu*f.sinphi(const.m)*field[1]
    	expected_new_state = (linalg.expm((-1j/const.hbar)*expected_operator*(t+1)))@state_obj.value
    	np.testing.assert_array_equal(new_state.value,expected_new_state)

if __name__ == '__main__':
    unittest.main()
