'''noiseanalyser.py
'''

import numpy as np
from solvers import FieldToPath
import math
import functions as f
import constants as const

class NoiseAnalyser(object):

  """this part does some noise analysis for a given field

  """
  def __init__(self,smoothfield,dt,variance,numfield):
      self.field=smoothfield
      self.dt=dt
      self.numfield=numfield
      self.variance=variance
 
  def calc_noisy_field(self):
      for x in range(numfield):
          sx=np.random.normal(0, self.variance, len(self.field))
          sy=np.random.normal(0, self.variance, len(self.field))
          noisy_field[:,0]=self.field[:,0]+sx.transpose
          noisy_field[:,1]=self.field[:,1]+sx.transpose
    
     
  def calc_path(self):
      path_solver = FieldToPath(noisy_field, dt)
      
      # Then invoke the solve() method of the path_solver object
      path_solver.solve()
      
      # Finally, use the export() method to export time, path, states (in this
      # order.) But one can also just export 'path' by the following line.
      path = path_solver.export()

  def calc_statistic(self):
      pass


