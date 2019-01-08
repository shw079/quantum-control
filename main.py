''' main.py
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'modules'))
import importPath, solvers
from transform import transform_path
from dataContainer import DataContainer
from visualization import Visualization

# Get user-specified path
user_path = importPath.import_path()
user_path.mainloop()
path_in = user_path.get_coordinates()

#filename = "tests/example_user_data.dat"
#root = importPath.import_path()
#root.load_from_file(filename)
#path_in = root.get_coordinates()

# Determine delta t and modify path array if necessary
path_desired, dt = transform_path(path_in)

# Instantiate a DataContainer object with path specified by the user
data = DataContainer(path_desired, dt)

# Solve for the required control fields
s = solvers.PathToField(data.path_desired, data.dt_atomic)
s.solve()
data.t, data.field, data.path_actual, data.state = s.export()


#plt.plot(data.path_desired[:,0],data.path_desired[:,1],label='desired')
#plt.plot(data.path_actual[:,0],data.path_actual[:,1],label='actual')
#plt.legend()
#plt.show()

# Analyze resulting paths from noisy control fields

# Visualize results
vis = Visualization(data)
vis.density()
vis.trajectory()
#vis.field()
