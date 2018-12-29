''' field.py
'''
import numpy as np

class Field(object):
    """ A Field is a 2xn array where n is the number of time points
    Class Field provides a container for calculated fields and functions to access them
    """

    def __init__(self, n):
        """ Initializes a Field instance with appropriate length (n number of time points) """
        self.value = np.zeros((n,2)) # value is one time point of field

    def set_value(self, i, field):
        """ Allow for user to set values of field """
        self.value[i,:] = field 

    def get_value(self):
        """ Allow for user to return values of field """
        return self.value