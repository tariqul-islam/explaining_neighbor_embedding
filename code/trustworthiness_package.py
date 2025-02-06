import numpy as np
import numba
from numba import prange

@numba.jit(nopython=True, parallel=True)
def get_first_order_graph(X,n_neighbors):
    
    N = X.shape[0]
    
    dist = np.zeros((N, N), dtype=np.float32)
    #euclidean_distances(X_train, squared = False)

    sort_idx = np.zeros((N,n_neighbors), dtype=np.int32)
    
    for i in range(N):
        if (i+1)%10000 == 0:
            print('Completed ', i+1, ' of ', N)
        
        for j in prange(i+1,N):
            dist[i,j] = np.sum( (X[i]-X[j])**2 )
            dist[j,i] = dist[i,j]
        
        sort_z = np.argsort(dist[i,:])
        #sort_z = sort_z[sort_z!=i]
        sort_idx[i,:] = sort_z[sort_z!=i]
    
    return sort_idx, dist

@numba.jit(nopython=True, parallel=True)
def my_trustworthiness(Y,sort_idx,K):
    #print('in my trustworthiness')
    N = Y.shape[0]

    #idxs = np.arange(N).astype(np.int64)

    val = 0.0
    for i in prange(N):
        dist = np.sum((Y - Y[i,:])**2,axis=1)
        #dist = dist[idxs!=i]
        #print(dist.shape)
        
        sort_idy = np.argsort(dist)#[1:]
        sort_idy = sort_idy[sort_idy!=i] #np.delete(sort_idy,np.argwhere(sort_idy==i))
        
        for j in range(K):
            r_0 = np.argwhere(sort_idy[j]==sort_idx[i,:])
            r = r_0[0,0] 
            
            r_v = r - K + 1
            if r_v>0:
                val += r_v
        #print(val)
    #print(val)
    val = val * 2.0 / ( N*K * (2*N - 3*K - 1))
    
    #print(val)
    
    return 1 - val
