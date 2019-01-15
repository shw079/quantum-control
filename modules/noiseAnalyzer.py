'''noiseanalyser.py
'''

import numpy as np
from solvers import FieldToPath
#import math
#import functions as f
#import constants as const
from multiprocessing import Pool

class NoiseAnalyser(object):
    """Analyze the impact of noisiness of control field on resulting 
    path.

    A NoiseAnalyser object simulates noisy fields and uses 
    solvers.FieldToPath to calculate the resulting path. It then for given control 
    fields with scaled noise drew from a Gaussian distribution with mean=0 and specified variance.
    """

    def __init__(self,smoothfield,dt,variance,numfield,processors=8):
        self.field=smoothfield
        self.dt=dt
        self.numfield=numfield
        self.variance=variance
        self.processors = processors
        self.path=np.empty((len(self.field), 2 * self.numfield),dtype=complex)

    def analyze(self):
        self.calc_noisy_field()
        self.calc_path()
        self.calc_statistic()

        return self.pathmean.astype(float), self.pathvar.astype(float)        
 
    def _calc_noisy_field(self):
        noisy_field=np.empty((len(self.field), 2 * self.numfield), dtype=complex)
        for i in range(self.numfield):
            sx=np.random.normal(0, self.variance, len(self.field))
            sy=np.random.normal(0, self.variance, len(self.field))
            noisy_field[:,0+i*2]=self.field[:,0]+np.transpose(sx)*self.field[:,0]
            noisy_field[:,1+i*2]=self.field[:,1]+np.transpose(sy)*self.field[:,1]
            
        self.noisy_field = noisy_field.astype(float)
     
    def _calc_path(self):
        noisy_field_list = np.hsplit(self.noisy_field, self.numfield)
        with Pool(self.processors) as p:
            path_list = p.map(self._calc_a_path,
                              [noisy_field_list[i] for i in range(self.numfield)])
        self.path = np.hstack(path_list)

    def _calc_a_path(self, a_field):
        path_solver = FieldToPath(a_field, self.dt)
        path_solver.solve()
        return path_solver.export()[1]
 
    def _calc_statistic(self):
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

