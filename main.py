''' main.py
'''

from module import importPath, solvers
from module.dataContainer import DataContainer
from module.visualization import Visualization

# Get user-specified path
user_path = importPath.import_path()
user_path.mainloop()

# Instantiate a DataContainer object with path specified by the user
data = DataContainer(user_path.get_coordinates)

# Solve for the required control fields
field_solver = solvers.PathToField(data.path_desired)
field_solver.solve()
data.path_actual, data.state = field_solver.export()[1:]

# Analyze resulting paths from noisy control fields

# Visualize results
