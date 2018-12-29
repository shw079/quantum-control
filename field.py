''' field.py
'''
import numpy as np

class Field(object):
    """Class Field provides a container for x and y field data.

    A Field instance contains the value of x- and y-component of field 
    data at a single time point.

    Parameter:

        value: an nd.array object of shape (2,), (1,2), or (2,1). The 
        first and second values correspond to the x- and y-component 
        of the field data, respectively.

    """

    def __init__(self, value):
        """Initializes a Field instance with value set to input value."""
        if not isinstance(value, np.ndarray):
            errmsg = "Input value must be an np.array."
            raise TypeError(errmsg)
        elif value.size is not 2:
            errmsg = "Input value must contain two scalars only."
            raise ValueError(errmsg)
        else:
            self.value = value.reshape(2) # value is one time point of field

    # def set_value(self, i, field):
    #     """ Allow for user to set values of field """
    #     self.value[i,:] = field 

    # def get_value(self):
    #     """ Allow for user to return values of field """
    #     return self.value