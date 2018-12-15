'''test_State.py
'''

import unittest
import numpy as np
import functions as f
import constants as const
from states import States
from state import State


class test_States(unittest.TestCase):

    def test_getValue(self):
        """check constructor and get_value"""
        m = 8
        n = 100
        states_obj = States(n,m)
        allstates=states_obj.get_value()
        self.assertEqual(allstates.shape,(2*const.m+1,n))

    def test_setValue(self):
        """check setting state values"""
        n = 10
        t = 0
        input_state = np.arange(0,2*const.m+1)
        state_obj = State(input_state,const.m)
        states_obj = States(n,const.m)
        states_obj.set_value(t,state_obj)
        allstates=states_obj.get_value()
        np.testing.assert_array_equal(allstates[:,0], input_state)



if __name__ == '__main__':
    unittest.main()





