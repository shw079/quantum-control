import unittest
import numpy as np
import numpy.testing as npt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
import importPath

'''Unit tests for importPath module'''

class test_importPath(unittest.TestCase):

	def test_objectCreation(self):
		'''Test that object creation does not fail'''	
		
		root = importPath.import_path()
		self.assertEqual(root.counter,0)
		
	def test_loadFile(self):
		'''Test ability to load data from file'''
		
		filename = "example_user_data.dat"
		root = importPath.import_path()
		root.load_from_file(filename)
		coords = root.get_coordinates()
		
		#make sure coords are right dimension
		npt.assert_equal((495,2),coords.shape)
		#check coords start at [0,0]
		npt.assert_array_almost_equal([0,0],coords[0])
		#check last coords - need to subtract coords at zero
		npt.assert_array_almost_equal(np.array([142,454])-np.array([148, 443]),coords[-1])
		#make sure no nan
		npt.assert_equal(np.isnan(coords),False)
		
	def test_loadFunction(self):
		'''Test ability to load data from user function'''
		
		filename = "example_user_function.py"
		root = importPath.import_path()
		root.load_from_file(filename)
		coords = root.get_coordinates()
		
		#check coords start at [0,0]
		npt.assert_array_almost_equal([0,0],coords[0])
		#check last coords - need to subtract coords at zero
		t0 = np.pi/2.0
		t1 = np.pi*10.0
		npt.assert_array_almost_equal(np.array([np.cos(t1)*t1,np.sin(t1)*t1])-
									  np.array([np.cos(t0)*t0,np.sin(t0)*t0]),coords[-1])

        @unittest.skip("Opens gui window, can't test on Travis CI")
	def test_clear(self):
		'''Test clear button actually removes data'''

		filename = "example_user_function.py"
		root = importPath.import_path()
		root.load_from_file(filename)

		root.after(500, lambda: root.clear())
		root.after(1000, lambda: root.destroy())
		root.mainloop()
    	
		self.assertRaises(ValueError, root.get_coordinates)
		self.assertEqual(root.previous_x,int(root.width/2.0))
		self.assertEqual(root.current_x,int(root.width/2.0))
		self.assertEqual(root.previous_y,int(root.height/2.0))
		self.assertEqual(root.current_y,int(root.height/2.0))
		

if __name__ == '__main__':
	unittest.main()


