'''test_Field.py
'''
import sys
sys.path.append('../modules')
import unittest
import numpy as np
import functions as f
from field import Field


class test_Field(unittest.TestCase):

    # def test_input(self):
    #     """check constructor"""
    #     n=10
    #     #t = 0
    #     #field_input = np.arange(0,2)
    #     field_obj = Field(n)
    #     field = field_obj.get_value()
    #     self.assertEqual(field.shape,(n,2))

    # def test_set_value(self):
    #     """check set value"""
    #     n=10
    #     t = 0
    #     field_input = np.arange(0,2)
    #     field_obj = Field(n)
    #     field_obj.set_value(t,field_input)
    #     check_field = field_obj.get_value()
    #     np.testing.assert_array_equal(check_field[t,:],field_input)

    def test_init(self):
        """Test instantiation of a Field instance."""
        value = np.array([1,2])
        values = [value, value.reshape(1,2), value.reshape(2,1)]
        for v in values:
            field = Field(v)
            self.assertEqual( field.value.all(), v.all() )

    def test_init_input(self):
        """Test input type and shape."""
        values = [[1,2], np.array([1, 2, 3])]
        self.assertRaises(TypeError, Field, values[0])
        self.assertRaises(ValueError, Field, values[1])


if __name__ == '__main__':
    unittest.main()



