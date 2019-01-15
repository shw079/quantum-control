''' main.py
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.', 'modules'))
from modules import importPath, solvers
from transform import transform_path
from dataContainer import DataContainer
from visualization import Visualization
from noiseAnalyzer import NoiseAnalyser
import matplotlib.pyplot as plt

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
myNA = NoiseAnalyser(data.field, data.dt_atomic,0.01, 4)
data.noise_stat_mean,data.noise_stat_var = myNA.analyze()

plt.figure(1)
plt.plot(data.t,data.path_actual[:,0],label='path_actual')
plt.plot(data.t,data.noise_stat_mean[:,0],label='mean')
plt.figure(2)
plt.plot(data.t,data.noise_stat_var[:,0])
plt.legend()
plt.show()

# Visualize results
#vis = Visualization(data)
#vis.density()
#vis.trajectory()
#vis.fields()
