"""
Due to file size limits on Github we subsampled the results with factor 100. 
We see that this only slightly adjusts the results presented in the paper and therefore provide this subsampled data for the reader to use.
If access to the full (non-subsampled) data is needed, please contact the authors.
"""

import os, sys
import pickle
import numpy as np
import jax.numpy as jnp

#%% imports
import pathmagic as pathmagic
PlotFolder, ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()

def subsample_file(path):
    if ".pkl" not in path:
        if ".png" in path:
            return
        raise Exception(f"File {path} is not a pickle file or png file")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # print(sys.getsizeof(pickle.dumps(data)))
    ## subsample data
    if isinstance(data, dict):
        for key, val in data.items():
            if key == "var_true":
                pass
            else:
                assert(float(key) < 1e8), "should be able to convert key to float"
                if val.shape[2] == 12000:
                    data[key] = val[:, :, ::100]
                else:
                    assert(val.shape[2] == 120), "shape not original 12000 but also not subsampled 120"
    elif isinstance(data, np.ndarray):
        assert(data.flatten().shape[0] < 200), "data shape too large"
    elif isinstance(data, list):
        new_data = []
        for list_ele in data:
            if list_ele.shape[2] == 12000:
                list_ele = list_ele[:, :, ::100]
            else:
                assert(list_ele.shape[2] == 120), "shape not original 12000 but also not subsampled 120"
            new_data.append(list_ele)
        data = new_data
    else:
        raise ValueError(f"data type {type(data)} not recognized")
    # print(sys.getsizeof(pickle.dumps(data)))
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return 

for folder in os.listdir(PlotFolder):
    folder_abspath = os.path.join(PlotFolder, folder)
    for file in os.listdir(folder_abspath):
        file_abspath = os.path.join(folder_abspath, file)
        if os.path.isdir(file_abspath):
            subfolder_abspath = file_abspath
            for subfile in os.listdir(subfolder_abspath):
                subfile_abspath = os.path.join(subfolder_abspath, subfile)
                subsample_file(subfile_abspath)
        else:
            subsample_file(file_abspath)
    print(f"Finished subsampling {folder}")
    
PlotFolder_wandb = PlotFolder.replace("plots", "plots_wandb1")
for folder in os.listdir(PlotFolder_wandb):
    folder_abspath = os.path.join(PlotFolder_wandb, folder)
    for file in os.listdir(folder_abspath):
        file_abspath = os.path.join(folder_abspath, file)
        if "n-PDEKF_plot_H_H_0.01_Q_NN" in file_abspath and "1e-05" in file_abspath:
            a = 0
        if os.path.isdir(file_abspath):
            subfolder_abspath = file_abspath
            for subfile in os.listdir(subfolder_abspath):
                subfile_abspath = os.path.join(subfolder_abspath, subfile)
                subsample_file(subfile_abspath)
        else:
            subsample_file(file_abspath)
    print(f"Finished subsampling {folder}")