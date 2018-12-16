'''testSolver.py
'''

import numpy as np
import unittest
from state import State
import functions as f
import operators as op
import constants as const
import solver as s

class test_PathToField(unittest.TestCase):
    def test_init(self):
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)

    def test_get_det(self):
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        det = fsolver._get_det(fsolver.states[0])
        self.assertTrue(np.isscalar(det))

    def test_calc_state_and_field(self):
        path_desired = np.arange(10).reshape((5,2))
        fsolver = s.PathToField(path_desired)
        fsolver.calc_state_and_field()

    def test_predefined_sigmoid_path(self):
        # should probably put the data inside another file and import instead
        """check input path"""
        input_path=np.array([[0,-0.103364011068980,0.346919195276499,-0.563178025732931,0.566286001964305],
        [0,-0.159423590524526,0.155071183489089,0.0879233264359249,-0.506872926855719]]).T
        dt = 21621621.6216216
        # data = DataStore(input_path)
        fs = s.PathToField(input_path)
        fs.calc_state_and_field()

        expected_field=np.array([[2.36058143825030e-07+0.0j,2.02064213344940e-07+0.0j],
            [3.75798580371715e-07 + 2.54225294236293e-27j,3.21681104868976e-07 + 2.17482244967637e-27j],
            [-1.03757051283114e-07 - 1.39319909533131e-27j,2.44958888320598e-07 - 5.50615452192899e-30j],
            [1.27954929407754e-06 + 4.18150395587884e-33j,3.23524205660377e-07 + 1.65378267567718e-32j],
            [1.42450464548976e-06 + 1.54412715012445e-25j,1.51462419430561e-07 - 1.67229842069920e-26j]])
        expected_state_step5=np.array([-2.56644749132289e-28 - 2.71978831146963e-28j,
                                        -4.36820067388291e-24 - 1.61761678513701e-24j,
                                        -4.20797135658652e-20 + 5.66271327681586e-22j,
                                        -2.28474567764552e-16 + 7.25628465188125e-17j,
                                        -6.33024732016318e-13 + 4.67773549984118e-13j,
                                        -3.65598118443247e-10 + 1.54386446833012e-09j,
                                        1.04332211496447e-06 + 2.30239172082715e-06j,
                                        0.00227753522253937 + 0.00294942683106131j,
                                        0.999992051804344 + 0.000716258687942210j,
                                        0.000973403450335052 + 0.000741269587615930j,
                                        -3.81329597515435e-06 + 7.06856801067158e-07j,
                                        -1.85338756280191e-09 + 2.25524150151543e-09j,
                                        -2.41960254674164e-13 + 1.21518698568277e-12j,
                                        3.71646922469583e-17 + 3.02924218282202e-16j,
                                        1.80625964418685e-20 + 4.46650715251373e-20j,
                                        3.04622865335820e-24 + 4.40462682369226e-24j,
                                        3.25592202462075e-28 + 3.17313644133731e-28j])
        expected_calc_path=([[0.0+0.0j,0.0+0.0j],[0.000819356425054785+2.16840434497101e-19j,0.000701363692838880-1.08423525971001e-19j],[0.00279254117586543+0.00000000000000j,0.00239039681830611-2.16840434497101e-19j],[0.00240372855088154+0.00000000000000j,0.00321605009954609-2.71315240917393e-20j],[0.00325356230619882-2.16840434497101e-19j,0.00220720424871606+2.16840434497101e-19j]])
        
        np.testing.assert_array_almost_equal(fs.field, expected_field, decimal=6)
        # np.testing.assert_array_almost_equal(fs.states[4].value, expected_state_step5)
        # np.testing.assert_array_almost_equal(fs.path_predicted, expected_calc_path)
