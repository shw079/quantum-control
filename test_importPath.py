import unittest
import numpy as np
import numpy.testing as npt
import importPath

'''Unit tests for importPath module'''

class test_importPath(unittest.TestCase):

	def test_objectCreation(self):
		root = importPath.import_path()
		
	def test_loadFile(self):
		filename = "example_data.dat"
		root = importPath.import_path()
		root.load_from_file(filename)
		coords = root.get_coordinates()
		
		#starts at (0,0)
		self.assertEqual(0,coords[0,0])
		self.assertEqual(0,coords[0,1])
		


if __name__ == '__main__':
	unittest.main()


