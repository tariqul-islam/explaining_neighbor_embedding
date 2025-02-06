import numpy as np
from scipy.linalg import orthogonal_procrustes

def my_procrustes(data1, data2):
    #adapted from https://github.com/scipy/scipy/blob/v1.11.2/scipy/spatial/_procrustes.py#L15-L131
    mtx1 = np.array(data1, dtype=np.double, copy=True)
    mtx2 = np.array(data2, dtype=np.double, copy=True)
    
    
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)
    
    mtx1 /= norm1
    mtx2 /= norm2
    
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    
    disparity = np.sum(np.square(mtx1 - mtx2))
    
    mtx2 = np.dot(data2, R.T) #just get the rotation, reflection - and no scaling
    
    
    return disparity, mtx2

def procrustes_distances(standard_array, array, verbose=True): 
    pds = []
    X_pdx = []

    for i in range(len(array)):
        d,x_pd = my_procrustes(standard_array, array[i])
        pds.append(d)
        X_pdx.append(x_pd)

    pds = np.array(pds)
    X_pdx = np.array(X_pdx)
    if verbose:
        print('Procrusted Distance: Mean: ', np.mean(pds), ' STD: ', np.std(pds))
    
    return pds, X_pdx

def procrustes_matrix(standard_array, array):
    diagonal, X_pdx = procrustes_distances(standard_array, array)
    
    N = len(array)
    pd_mat = np.zeros((N,N))
    
    for i in range(N):
        row, _ = procrustes_distances(array[i], array, verbose=False)
        pd_mat[i,:] = row
    
    #print(np.diag(pd_mat))
    
    for i in range(N):
        pd_mat[i,i] = diagonal[i]
        
    return pd_mat, X_pdx
