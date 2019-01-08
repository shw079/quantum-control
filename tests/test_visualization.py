import unittest
import numpy as np
from os.path import dirname, abspath, join
import sys
sys.path.append(join(dirname(dirname(abspath(__file__))), "modules"))

from visualization import Visualization


class test_data:
    """Class to contain temporary data for unit testing

    """
    def __init__(self):
        self.state = np.load("example_state.npz")['arr_0']
        Const = namedtuple("Const", ["m"])
        self.t = np.arange(10000)
        self.field = None
        self.path_actual = np.array([[0, 0]])
        self.path_desired = np.array([[0, 0]])


class TestVisualization(unittest.TestCase):
    """Unit testing for visualization module

    """
    def test_density(self):
        """Test whether the calculation of probability density 
           from state is correct

        """
        proba_calculated = Visualization(test_data())\
            .density(n_grid=101)\
            .astype(np.float16)
        proba_expected = np.load("example_proba.npz")['arr_0']
        np.testing.assert_array_almost_equal(proba_calculated,
                                             proba_expected,
                                             decimal=3)
    

if __name__ == "__main__":
    unittest.main()


