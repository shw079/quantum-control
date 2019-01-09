''' main.py
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'modules'))
import importPath, solvers
from dataContainer import DataContainer
from visualization import Visualization

# Get user-specified path
root = importPath.import_path()
root.mainloop()
path_in = root.get_coordinates()

# Instantiate a DataContainer object with path specified by the user
data = DataContainer(path_in)

# Solve for the required control fields
s = solvers.PathToField(data.path_desired, data.dt_atomic)
s.solve()
data.t, data.field, data.path_actual, data.state = s.export()


# Analyze resulting paths from noisy control fields

# Visualize results
vis = Visualization(data)
vis.density()
vis.trajectory()
vis.fields()
