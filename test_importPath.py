import unittest
import numpy as np
import numpy.testing as npt
import importPath

'''Unit tests for importPath module'''

class test_importPath(unittest.TestCase):

	def test_objectCreation(self):
		root = importPath.import_path()


if __name__ == '__main__':
	unittest.main()


