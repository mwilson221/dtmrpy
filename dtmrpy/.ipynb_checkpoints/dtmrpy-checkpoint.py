import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

            
def get_diffusion_data(src, fib):

    src_dim = src['dimension'][0]
    fib_dim = fib['dimension'][0]

    #Note here we permute dimensions 0 and 1 - this is an artifact of DSI studio
    mag0=fib['fa0'].reshape(fib_dim[1],fib_dim[0],fib_dim[2])

    #Note here we permute dimensions 0 and 1 - this is an artifact of DSI studio
    index0 = fib['index0'].reshape(fib_dim[1],fib_dim[0],fib_dim[2])
    dir0 = fib['odf_vertices'][:,index0]

    sigma=np.zeros((fib_dim[1],fib_dim[0],fib_dim[2],3))

    for i in range(fib_dim[1]):
        for j in range(fib_dim[0]):
            for k in range(fib_dim[2]):
                v = dir0[:,i,j,k]
                v=v*mag0[i,j,k]
                sigma[i,j,k,:] = v
                
    return sigma


def get_diffusion_data_hcp(fib):
    fib_dim = fib['dimension'][0]

    #Note here we permute dimensions 0 and 1 - this is an artifact of DSI studio
    mag0=fib['fa0'].reshape(fib_dim[0],fib_dim[2],fib_dim[1])

    #Note here we permute dimensions 0 and 1 - this is an artifact of DSI studio
    index0 = fib['index0'].reshape(fib_dim[0],fib_dim[2],fib_dim[1])
    dir0 = fib['odf_vertices'][:,index0]

    sigma=np.zeros((fib_dim[0],fib_dim[2],fib_dim[1],3))

    for i in range(fib_dim[0]):
        for j in range(fib_dim[2]):
            for k in range(fib_dim[1]):
                v = dir0[:,i,j,k]
                v=v*mag0[i,j,k]
                sigma[i,j,k,:] = v
                
    return sigma


def get_covariances(fib,num_dir=5):
    #Get 'covariance field' from batch outer product for vector field directions
    fib_dim = fib['dimension'][0]
    vector_field_list=[]
    #Note here we permute dimensions 0 and 1 - this is an artifact of DSI studio
    covariances = np.zeros((3,3,fib_dim[1],fib_dim[0],fib_dim[2]))
    
    for i in range(num_dir):
        direction = fib['odf_vertices'][:,fib['index'+str(i)].reshape(fib_dim[1],fib_dim[0],fib_dim[2])]
        magnitude = fib['fa'+str(i)].reshape(fib_dim[1],fib_dim[0],fib_dim[2])

        vector_field_list.append(direction*magnitude)

    for vector_field in vector_field_list:
        covariances = covariances + np.einsum('ixyz,jxyz->ijxyz', vector_field, vector_field)
        
    return np.transpose(covariances,[2,3,4,0,1])


class DT_GMM:
    
    def __init__(self, weights, means, covariances):

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        
        
    def plot(self, weights = True):
        
        U,S,_ = np.linalg.svd(self.covariances_)
        
        if weights:
            w=self.weights_
            w=w/max(w)
        else:
            w=np.ones(self.weights_.shape)

        for i, (x,y,z) in enumerate(self.means_):
            rgb = sum(abs(U[i]@np.diag(S[i])).T)

            plt.plot(x,y,z,'.',c=rgb/sum(rgb), alpha = w[i][0])






            
    
