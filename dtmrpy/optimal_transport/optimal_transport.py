import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import ot

from dtmrpy import DT_GMM

from scipy.spatial.distance import cdist

import geomstats
from geomstats.geometry.spd_matrices import SPDMatrices, SPDBuresWassersteinMetric

#for bures log map
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.geometry.matrices import Matrices

space = SPDMatrices(3)
space.equip_with_metric(SPDBuresWassersteinMetric)




######### For Wasserstein-type distance
def bures_distance_matrix2(Sigma_x,Sigma_y):
    Lx, Qx = torch.linalg.eigh(Sigma_x)
    Sigma_x_sqrt = Qx @ torch.diag_embed(torch.sqrt(Lx*(Lx>0))) @ Qx.mH
    
    cross_term = torch.matmul(torch.matmul(Sigma_x_sqrt, Sigma_y.unsqueeze(1)),Sigma_x_sqrt.unsqueeze(0))
    
    N=cross_term.shape[0]
    M=int(1000000/cross_term.shape[1])
    
    for i in np.arange(0,N,M):
        Lc,Qc = torch.linalg.eigh(cross_term[i:(i+M)])
        cross_term[i:i+M] = Qc @ torch.diag_embed(torch.sqrt((Lc*(Lc>0)))) @ Qc.mH

    return torch.einsum('ijkk -> ij', Sigma_x.unsqueeze(0) + Sigma_y.unsqueeze(1) - 2*cross_term).T #the transpose here should be fixed


def wasserstein_type_distance(mu0,mu1):
    (weights0,means0,sigma0)=(mu0.weights_,mu0.means_,torch.tensor(mu0.covariances_).to('cuda'))
    (weights1,means1,sigma1)=(mu1.weights_,mu1.means_,torch.tensor(mu1.covariances_).to('cuda'))

    M = cdist(means0,means1)**2 + bures_distance_matrix2(sigma0,sigma1).cpu().numpy()
    
    a = weights0.reshape(-1)/np.sum(weights0)
    b = weights1.reshape(-1)/np.sum(weights1)

    return M, ot.emd(a,b,M)




############ For Bures Log Map
def my_symmetric_sqrt(M, p):
    U,S,V = np.linalg.svd(M)
    return U@np.diag(S**p)@V

def my_bures_log(point, base_point):
        """Compute the Bures-Wasserstein logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the Bures-Wasserstein metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., n, n]
            Riemannian logarithm.
        """        
        # compute B^1/2(B^-1/2 A B^-1/2)B^-1/2 instead of sqrtm(AB^-1)
        sqrt_bp = my_symmetric_sqrt(base_point, 0.5)
        inv_sqrt_bp = my_symmetric_sqrt(base_point, -0.5)
        pdt = my_symmetric_sqrt(Matrices.mul(sqrt_bp, point, sqrt_bp),.5)
        sqrt_product = Matrices.mul(sqrt_bp, pdt, inv_sqrt_bp)
        transp_sqrt_product = Matrices.transpose(sqrt_product)
        return sqrt_product + transp_sqrt_product - 2 * base_point





##TODO: SPACE NEEDS TO BE ADDED AS AN INPUT PARAMETER!!! 
class wasserstein_type_barycenter():
    
    def __init__(self, n_support = 200, barycenter=None, max_iter=10):
        self.n_support = n_support
        self.max_iter = max_iter
        self.barycenter_ = barycenter
        self.features_ = None
        self.Pi_list=[]
        #other parameters for how to calculate barycenter

    def means_update(self, barycenter, measure_list, Pi_list):

        return np.mean(np.array([measure_list[i].means_.T@Pi_list[i].T@np.diag(1/barycenter.weights_.reshape(-1)) for i in range(len(measure_list))]),0).T

    def covariance_update(self, space, barycenter, measure_list, Pi_list):

        Sigma0=space.projection(barycenter.covariances_)
        Sigma0_update = np.zeros_like(Sigma0)
        num_measures=len(measure_list)

        for p, measure in enumerate(measure_list):

            Pi=Pi_list[p]

            Sigma1=space.projection(measure.covariances_)

            for i in range(Sigma0.shape[0]):
                sigma0=Sigma0[i]
                ind = np.where(Pi[i]>0)[0]
                Sigma1_log = np.array([my_bures_log(sigma1,sigma0) for sigma1 in Sigma1[ind]])

                #Update Covariance
                temp=np.zeros((3,3))
                w_i=sum(Pi[i])
                for j in range(len(ind)):
                    temp += Pi[i,ind[j]]*Sigma1_log[j]

                Sigma0_update[i] += space.metric.exp(temp/w_i,sigma0)/num_measures

        return Sigma0_update

    def free_support_barycenter_update(self, measure_list, barycenter=None, N=200):

        if self.barycenter_==None:
            # initialize weights
            init_weights = np.ones((N,1))/N
            init_means = np.zeros(3)+np.random.normal(size=(N,3))
            init_covariances = SPDMatrices(3).random_point(N)
            barycenter = DT_GMM(weights=init_weights, means=init_means, covariances=init_covariances)

        else:

            self.Pi_list = [wasserstein_type_distance(barycenter, measure)[1] for measure in measure_list]
            barycenter.means_ = self.means_update(barycenter, measure_list, self.Pi_list)
            barycenter.covariances_ = self.covariance_update(space, barycenter, measure_list, self.Pi_list) 

        return barycenter 
    
    def fit(self,measure_list, plot_steps=False):
        
        for i in range(self.max_iter):
    
            self.barycenter_=self.free_support_barycenter_update(measure_list, self.barycenter_, N = self.n_support)
        
            if plot_steps==True:
                plt.figure().add_subplot(projection='3d')
                self.barycenter_.plot()
                plt.show()
                
                
    def fit_transform(self, measure_list, plot_steps=False):
        
        self.fit(measure_list, plot_steps=plot_steps)
        
        self.features_ = np.array([self.get_features(mu, pi) for (mu, pi) in zip(measure_list, self.Pi_list)])
    
        
    def get_features(self, mu, pi):
        
        X = torch.tensor(mu.means_)
        Y = torch.tensor(self.barycenter_.means_)
        a = torch.tensor(self.barycenter_.weights_)
        T = torch.tensor(pi)
        
        return torch.sum(((torch.diag(1/a)*T)).unsqueeze(2)*(X.unsqueeze(0)-Y.unsqueeze(1)),1).numpy()       
