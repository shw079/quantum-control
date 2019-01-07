''' main.py
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'modules'))
import importPath, solvers
from dataContainer import DataContainer
from visualization import Visualization

# Get user-specified path
user_path = importPath.import_path()
user_path.mainloop()
path_in = user_path.get_coordinates()

# Determine delta t and modify path array if necessary
# Below two lines just pretend as we got the desired `dt` and 
# `path_desired`.
dt = 1000 #This is just for demo
path_desired = path_in #This is just for demo

# Instantiate a DataContainer object with path specified by the user
data = DataContainer(path_desired, dt)

# Solve for the required control fields
s = solvers.PathToField(data.path_desired, data.dt)
s.solve()
data.t, data.field, data.path_actual, data.state = s.export()

# Analyze resulting paths from noisy control fields


# Visualize results
vis = Visualization(data)
vis.density()
vis.trajectory()
vis.field()
