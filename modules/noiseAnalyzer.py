'''noiseanalyser.py
'''

import numpy as np
from solvers import FieldToPath
from joblib import Parallel, delayed

class NoiseAnalyser(object):
    """Class for doing some noise analysis for a given field to calculate the mean and variance for the output path. 
    The NoiseAnalyzer module uses some data in the DataContainer object and some data are specified by the user.Since calulating path from each noisy field is independet of the calculating the path for the other noisy fields, this part can be parallel. In this module:
    
    - The field that is calculated through the simulation is the input. 
    - The user specifies the variance of the noise and the number of samples that are going to be made for analysis. 
    - Every set of noise is added to the field and the path is calculated for each set of noisy field through FieldToPath.
    - In the last part mean and variance of calculated paths are calculated as the output.

    Parameters
    ----------
    smoothfield : numpy.array, shape=(n,2)
        Control fields e_x and e_y at each time point calculated by 
        PathToField. Now, stored in the DataContainer.
 
    dt : float, optional (default=1000)
        Difference of time between two adjacent time points. This is 
        path-specific and is currently calculated when a 
        dataContainer.DataContainer object is instantiated with a
        desired path.
    
    variance : folat
        Variance of the distribution of numbers that are added to the field as noise.

    numfield : integer
        Number of samples that are used for noise analysis. 

    processors : int, optional(default=4)
        Number of proccessors for the parallelizing this part of the code.  

    Attributes
    ----------
    n : integer
        number of points.

    field : numpy.array, shape=(n, 2)
        Control fields.
    
    dt : float, optional (default=1000)
        Difference of time between two adjacent time points.

    numfield : integer
        Number of samples that are used for noise analysis.

    variance : folat
        Variance of the distribution of numbers that are added to the field as noise.

    processors : int, optional(default=4)
        Number of proccessors for the parallelizing this part of the code.  

    path : numpy.arrray shape(n,2*numfield)
        Matrix that contains all paths that are calculate from noisy fields.

    noisy_field : numpy.arrray shape(n,2*numfield)
        Matrix that contains all noisy field controls.

    pathmean : numpy.array, shape(n,2)
        Mean path which is calculated from all paths that are output of PathToField solver.

    pathvar : numpy.array, shape(n,2)
        Variance of the noisy field.

    
    """

    def __init__(self,smoothfield,dt,variance,numfield,processors=4):
        self.field=smoothfield
        self.dt=dt
        self.numfield=numfield
        self.variance=variance
        self.processors = processors
        self.path=np.empty((len(self.field), 2 * self.numfield),dtype=complex)
        
 
    def calc_noisy_field(self):
        """This method produces some random number with normal 
        distribution to be added to the control field.

        """
        noisy_field=np.empty((len(self.field), 2 * self.numfield), dtype=complex)
        for i in range(self.numfield):
            sx=np.random.normal(0, self.variance, len(self.field))
            sy=np.random.normal(0, self.variance, len(self.field))
            noisy_field[:,0+i*2]=self.field[:,0]+np.transpose(sx)*self.field[:,0]
            noisy_field[:,1+i*2]=self.field[:,1]+np.transpose(sy)*self.field[:,1]
            
        self.noisy_field = noisy_field.astype(float)
       
     
    def calc_a_path(self, i):
        """Calculates the path from PathToField for one noisy field. 
          
        Parameters
        ----------
        i : integer
            counter

        Returns
        ----------
        path : numpy.array, shape=(n,2) 
            matrix containing on path.


        """
        #def calc_a_path(i):
        path_solver = FieldToPath(self.noisy_field[:,[i*2,i*2+1]], self.dt)
        # Then invoke the solve() method of the path_solver object
        path_solver.solve()
        return path_solver.export()[1]

    def calc_path(self):
        """Parallel version of calc_a_path to calculate the path for all noisy fields. 
        

        """
        noisy_paths = Parallel(n_jobs=self.processors)(delayed(self.calc_a_path)(i) for i in range(0,self.numfield))
        for i in range(0, len(noisy_paths)):
            self.path[:,[i*2,i*2+1]] = noisy_paths[i]
 
    def calc_statistic(self):
        """Calculate the mean path from the calculated path from noisy fields. This method also calcules a matrix with the same dimension as the path that shows the variance of each point cooridante variance from the mean path.
       

        """
        xcollec= np.zeros((len(self.path),self.numfield))
        ycollec= np.zeros((len(self.path),self.numfield))
        for j in range(len(self.path)):
            for i in range(2*self.numfield):
                mod=i%2
                if mod==0:
                    xcollec[j,int(i/2)]=self.path[j,i]
                else:
                    ycollec[j,int((i-1)/2)]=self.path[j,i]
        pathxmean = np.zeros(len(self.path))
        pathymean = np.zeros_like(pathxmean)
        pathxvar = np.zeros_like(pathxmean)
        pathyvar = np.zeros_like(pathxmean)
        for i in range(len(self.path)):
            pathxmean[i]=sum(xcollec[i,:])/self.numfield
            pathymean[i]=sum(ycollec[i,:])/self.numfield
            pathxvar[i] =np.var(xcollec[i,:])
            pathyvar[i] =np.var(ycollec[i,:])
        self.pathmean = np.stack((pathxmean,pathymean),axis=1)
        self.pathvar = np.stack((pathxvar,pathyvar),axis=1)

    def analyze(self):
        """ This is a wraper of other member method to do the statistics.    

        Returns
        ----------
        pathmean : numpy.array, shape(n,2)
            Mean of the path from noisy fields. 

        pathvar : numpy.array, shape(n,2)
            variance of the path from noisy fields.

        """
        self.calc_noisy_field()
        self.calc_path()
        self.calc_statistic()

        return self.pathmean.astype(float), self.pathvar.astype(float)

