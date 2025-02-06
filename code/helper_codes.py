import numpy as np

def plot_idxs(array,idxs,figsize=(10,10), title=None, values = None, tick_off=True):
    
    plt.figure(figsize=figsize)
    
    if title is not None:
        plt.title(title)
        
    n_plot = int(np.sqrt(len(idxs)))
    
    for i in range(n_plot**2):
        plt.subplot(n_plot, n_plot, i+1)
        plt.scatter(array[idxs[i],:,0], array[idxs[i],:,1], c=y_train, s=0.01, cmap='Spectral')
        if values is not None:
            plt.title(str(values[i]))
        if tick_off:
            plt.xticks([])
            plt.yticks([])
        
    return

def plot_low_k_idxs(array, metric, k, title=None):

    idxs_arg = np.argpartition(metric, k)[:k]
    values = metric[idxs_arg]

    idxs_arg_pointer = np.argsort(values)

    values = values[idxs_arg_pointer]
    idxs_arg = idxs_arg[idxs_arg_pointer]

    plot_idxs(array, idxs_arg, title=title, values=values)
    
    return

def plot_high_k_idxs(array, metric, k, title=None):

    idxs_arg = np.argpartition(metric, -k)[-k:]
    values = metric[idxs_arg]

    idxs_arg_pointer = np.argsort(values)

    values = values[idxs_arg_pointer]
    idxs_arg = idxs_arg[idxs_arg_pointer]

    plot_idxs(array, idxs_arg, title=title, values=values)
    
    return

def procrustes_distances(standard_array, array): 
    pds = []
    X_pdx = []

    for i in range(len(array)):
        _,x_pd,d = procrustes(standard_array, array[i])
        pds.append(d)
        X_pdx.append(x_pd)

    pds = np.array(pds)
    X_pdx = np.array(X_pdx)
    print('Procrusted Distance: Mean: ', np.mean(pds), ' STD: ', np.std(pds))
    
    return pds, X_pdx
