"""
All functionalities related to plotting and handling of the pickle result files
"""

import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize
from matplotlib.pyplot import hist2d
from copy import deepcopy
from copy import copy
import matplotlib.cm as cm
from collections import defaultdict 
import scipy.stats as st
import jax
from functools import partial
import jax.numpy as jnp

#%% set plotting settings (latex style plots)
VisualizationFolder = os.path.dirname(os.path.abspath(__file__))
plotstylepath = os.path.join(VisualizationFolder, "plot_style.txt")
plt.style.use(plotstylepath)

mpl.rcParams['mathtext.fontset'] = 'cm'
# mpl.rcParams.update({'font.size':50})
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
color_cycle = iter(colors)
styles = ['solid', 'dotted']
fontsize = 30
labelsize = 38
def ts_heatmap(Y, t=None, ax=None, normalize_y=False, colorscale=None, cmap='viridis', dt=0.005, **kwargs):
    """
    Produces a heatmap of a collection of time series.
 
    Arguments:
        Y : array
            Array of shape (m,n) containing time series values, where m is
            number of time series, and n is number of points per time
            series. Expects the length of each time series to be the same.
        t : array
            Array containing the values of the time steps. Should be length
            m, corresponding to the length of the 1st axis of Y.
 
    Returns:
        matplotlib figure
    """
    default_n_bins_y = 100
    colorscale='log'
    if (colorscale == 'linear') or (colorscale is None):
        norm = Normalize()
    elif colorscale == 'log':
        norm = LogNorm(vmin=0.6)
        cmap = copy(cm.get_cmap(cmap))
        cmap.set_bad(cmap(0))
        
    try:
        Y = np.array(Y)
        assert len(Y.shape) == 2, "Y should have 2 dimensions"
        t_grid = np.ones(Y.shape) * t[np.newaxis,:]
        t_concat = t_grid.flatten()
        Y_concat = Y.flatten()
        weights = np.ones(len(Y_concat))
    except:
        lengths = [len(metric_i) for metric_i in Y]
        maxlength = np.max(lengths)
        inverted_Y = []
        for time_i in range(maxlength):
            preds = []
            for pred_i in range(len(Y)):
                try:
                    preds.append(Y[pred_i][time_i])
                except:
                    pass
            inverted_Y.append(preds)
        ti = [np.linspace(0, lengths[i] * dt, lengths[i]) for i in range(len(Y))]
        ts = []
        ys = []
        ws = []
        for time_idx in range(len(t)):
            n_active = len(inverted_Y[time_idx])
            ys.append(inverted_Y[time_idx])
            ts.append([t[time_idx]]*n_active)
            ws.append(n_active * [1/n_active * maxlength])
        Y_concat = np.concatenate(ys)
        t_concat = np.concatenate(ts)
        w_concat = np.concatenate(ws)
        if normalize_y:
            weights = w_concat
        else:
            weights = np.ones(len(Y_concat))
    bins_x = (len(t)-1)
    bins_y = default_n_bins_y
    bins = [bins_x, bins_y]
    ax.hist2d(t_concat, Y_concat, bins=bins, weights=weights, cmap=cmap, norm=norm, **kwargs)

#%%
def get_selected_paths(var=None, varname=None, modeltype=None, EnsembleSave=None, modeltype_path=None, MC=None):
    if modeltype_path == None: modeltype_path = os.path.join(EnsembleSave, modeltype)
    modeltype_results_paths = []
    for result_file in sorted(os.listdir(modeltype_path)):
        if var is not None and not f'{varname}{str(round(var,2)).replace(".","_")}' in result_file: continue
        H_val = float((result_file[result_file.find("H")+1:result_file.find("H")+5]).replace("_","."))
        if H_val not in MC.H: 
            print(f"Value {H_val} not in MC.beta")
            continue
            raise Exception(f"Value {H_val} not in MC.H")
        beta_val = float((result_file[result_file.find("beta")+4:result_file.find("beta")+8]).replace("_","."))
        if beta_val not in [round(MC.beta[i], 2) for i in range(len(MC.beta))]: 
            print(f"Value {beta_val} not in MC.beta")
            continue
            raise Exception(f"Value {beta_val} not in MC.beta")
        result_path = os.path.join(modeltype_path, result_file)
        modeltype_results_paths.append(result_path)       
    return modeltype_results_paths               
            
def zip_w(var=None, varname=None, modeltype_path=None):
    if not os.path.exists(modeltype_path):
        print(f"Modeltype path {modeltype_path} does not exist")
        return

    savepathdir = modeltype_path.replace("/results", "/results_zipped")
    savepath = os.path.join(savepathdir, f'w_{varname}{str(round(var,2)).replace(".","_")}.npz')
    if os.path.exists(savepath): return

    selected_paths = get_selected_paths(var=var, varname=varname, modeltype_path=modeltype_path)
    if len(selected_paths) == 0: 
        print(f"no results yet for {varname} = {var}")
        return
    first_result_path = os.path.join(modeltype_path, selected_paths[-1])
    return_dict_0, _ = load_data(first_result_path)

    wshape = return_dict_0['filtering']['w'].shape
    if wshape[-2] < 6: 
        print("Already file without len(w) > 6")    # has already been made smaller, no need to still zip and/or backup
        return
    
    # iterate over all result files and concatenate into 1 big array --> 192 simulations --> 192 x w_shape_i
    w_array = np.empty(((len(selected_paths),) + wshape), dtype=np.float32)
    for i, result_file in enumerate(selected_paths):
        result_path = os.path.join(modeltype_path, result_file)
        return_dict, plotting_info = load_data(result_path)
        wi = return_dict['filtering']['w']
        w_array[i] = wi
        print(i)
        
    if not os.path.exists(savepathdir): os.makedirs(savepathdir)
    np.savez_compressed(savepath, w=w_array)
    print(f"Weights Saved to {savepath}")
    return None

def remove_w_NN(var=None, varname=None, modeltype_path=None):
    selected_paths = get_selected_paths(var=var, varname=varname, modeltype_path=modeltype_path)
    for i, result_file in enumerate(selected_paths):
        result_path = os.path.join(modeltype_path, result_file)
        return_dict, plotting_info = load_data(result_path)
        n_p = len(plotting_info['trainable_names']) # number of trainable physical parameters
        ## overwrite to only save last n_p elements!
        if return_dict['filtering']['w'].shape[-2] > 6:
            return_dict['filtering']['w'] = return_dict['filtering']['w'][:,:,-n_p:,:]
            write_data(return_dict, plotting_info, result_path) # overwrite the file with smaller w array
            print(i)
    return

def get_selected_pred(var=None, varname=None, modeltype_path=None, load=False, MC=False, verbose=True):
    
    selected_paths = get_selected_paths(var=var, varname=varname, modeltype_path=modeltype_path, MC=MC)
    modeltype_results_list = []
    for i, result_file in enumerate(selected_paths):
        result_path = os.path.join(modeltype_path, result_file)
        with open(result_path, 'rb') as f:
            return_dict = pickle.load(f)
        modeltype_results_list.append(return_dict)
        if verbose:
            if i in np.arange(0,len(selected_paths),int(len(selected_paths)/5)): print(f"Loaded {i}/{len(selected_paths)}")
    result_arr = np.array(modeltype_results_list)
    return result_arr

def get_selected_results(var=None, varname=None, modeltype_path=None, MC=False):
    selected_paths = get_selected_paths(var=var, varname=varname, modeltype_path=modeltype_path, MC=MC)
    modeltype_results_list = []
    plotting_info_list = []
    for i, result_file in enumerate(selected_paths):
        result_path = os.path.join(modeltype_path, result_file)
        return_dict, plotting_info = load_data(result_path)
        if np.any(np.isnan(return_dict['filtering']['w'][:,:,-2,0])): print(f"NaN results for simulation {i}")
        modeltype_results_list.append(return_dict)
        plotting_info_list.append(plotting_info)
        if i in np.arange(0,len(selected_paths),int(len(selected_paths)/5)): print(f"Loaded {i}/{len(selected_paths)}")
    result_dict = {'results': modeltype_results_list, 'plotting_info': plotting_info_list}
    return result_dict

def write_data(return_dict, plotting_info, result_path):
    return_dict_path = os.path.join(result_path, 'return_dict.pkl')
    plotting_info_path = os.path.join(result_path, 'additional_plotting_info.pkl')
    with open(f'{return_dict_path}', 'wb') as f:
        pickle.dump(return_dict, f)
    with open(f'{plotting_info_path}', 'wb') as f:
        pickle.dump(plotting_info, f)
    return None
                      
def load_data(result_path):
    return_dict_path = os.path.join(result_path, 'return_dict.pkl')
    plotting_info_path = os.path.join(result_path, 'additional_plotting_info.pkl')
    with open(f'{return_dict_path}', 'rb') as f:
        return_dict = pickle.load(f)
    with open(f'{plotting_info_path}', 'rb') as f:
        plotting_info = pickle.load(f)
    return return_dict, plotting_info                                

@partial(jax.jit, static_argnums=(1,))
def jaxnanmean_jitted(moving_result, axis=-1):
    return jnp.nanmean(moving_result, axis=axis)

@partial(jax.jit, static_argnums=(1,))
def rolling_window(a: jnp.ndarray, window: int):
  idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
  return a[idx]

def moving_average(camsimu_innov_error, mw=3):
    padded_array = jnp.pad(camsimu_innov_error, ((0, 0), (mw - 1, 0)), mode='constant', constant_values=np.nan)
    moving_result = jax.vmap(rolling_window, in_axes=(0, None))(padded_array, mw)
    mean_window = jaxnanmean_jitted(moving_result, axis=-1)
    return mean_window

@partial(jax.jit, static_argnums=(2,3))
def process_selection_data(val, mw_innov, subselect_n_ensembles, return_best_idxs=False):
    sorted_idxs = jnp.argsort(mw_innov)
    best_idxs = sorted_idxs[:subselect_n_ensembles]
    worst_idxs = sorted_idxs[subselect_n_ensembles:]
    
    best_array = val[best_idxs]
    worst_array = val[worst_idxs]
    
    if return_best_idxs:
        return best_array, worst_array, best_idxs
    return best_array, worst_array

def get_best_x_ensembles(val, selection_data, subselect_n_ensembles, mw=1000, transpose=False, return_best_idxs=False):
    """
    val.shape = n_ensemble x time
    selection_data.shape = n_ensemble x time
    
    returns 
    best_array.shape: n_ensemble_selected x time
    worst_array.shape: n_ensemble - n_ensemble_selected x time
    
    
    if transpose then all the shapes are transposed: i.e (time x ...)
    """
    if transpose:
        if len(val.shape) > 2:
            if len(val.shape) > 3:
                raise Exception("Not implemented for >3D input")
            val = np.transpose(val, axes=(1,0,2))
        else:
            val = val.T
        selection_data = selection_data.T
    
    best_array = []
    worst_array = []
    if mw > 1:
        selection_data = moving_average(selection_data, mw)
    if return_best_idxs:
        best_array, worst_array, best_idxs = jax.vmap(process_selection_data, in_axes=(1, 1, None, None))(val, selection_data, subselect_n_ensembles, return_best_idxs)
    else:
        best_array, worst_array = jax.vmap(process_selection_data, in_axes=(1, 1, None))(val, selection_data, subselect_n_ensembles)

    
    if not transpose:   # with transposed inputs, output should not be transposed and vice versa
        best_array = best_array.T
        worst_array = worst_array.T
        
    if return_best_idxs:
        return best_array, worst_array, best_idxs
    return best_array, worst_array

def calc_mean_std(metric_results, lengths, maxlength):
    all_equal = all([lengths[i] == lengths[0] for i in range(len(lengths))])
    
    if all_equal:   ## vectorize if all lists are of equal length -- faster 
        mean = np.mean(metric_results, axis=0)
        std = np.std(metric_results, axis=0)
    else:           ## if I want to calculate mean of all lasting lists
        means = []
        stds = []
        for i in range(maxlength):
            vals = []
            for results in metric_results:
                try:
                    vals.append(results[i])
                except:
                    pass
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        mean = np.array(means)
        std = np.array(stds) 
    return mean, std    


def scan_get_best_x_ensembles(carry, x, subselect_n_ensembles, mw=1, return_best_idxs=False):
    val, selection_data = x

    # comply with previous shape
    val = val.T
    selection_data = selection_data.T
    
    y = get_best_x_ensembles(val, selection_data, subselect_n_ensembles, mw=mw, return_best_idxs=return_best_idxs)
    return carry, y

def dict_to_results_JAX(results_dict, model, ax, var_i, plot_axes, plot_what, padded, subselect_n_ensembles, dt, heatmap, normalize_y, var_idx, line_per_line, plot_mean, plot_std, load=True, save=False, overwrite=False, savefolder=None, selection_criteria="Qx", plot_error=True, plot_on_single_axis=False, plot=True, savepath=None, metrics_plot=False, print_end_mean=False, sharey=False):
    names = []
    first_one_model = True

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'])

    if 'hybrid' in model: model = model.replace("hybrid_", 'H-')
    if 'physics' in model: model = model.replace("physics_", 'p-')
    if 'blackbox' in model: model = model.replace("blackbox_", 'n-')

    style = styles[0] if 'PDEKF' in model else styles[1]
    marker = 'v' if 'PDEKF' in model else 'None'
    marker = 'None'
    if metrics_plot:
        # color = next(color_cycle)
        pass
    else:
        color = colors[0] if 'H-' in model else colors[1] if 'p-' in model else colors[2]
        # pass
    if load:
        if os.path.exists(savepath):
            with open(savepath, 'rb') as f:
                metric_results_all = pickle.load(f)
        else:
            import sys
            import warnings
                        
            RED = "\033[91m"
            RESET = "\033[0m"

            warnings.warn(
                f"{RED}\n⚠️ WARNING: Missing File\n"
                f"File not found: {savepath}\n"
                "\nThis file has not been created yet. To resolve this issue, you have two options:\n"
                "1️⃣ **Generate the file:**\n"
                "   - Run the script with: `load=False, save=True`\n"
                "   - Requires all results in the folder results/ to be made (or unzipped). \n"
                "\n"
                "2️⃣ **Manually extract from the provided zip files:**\n"
                "   - Unzip the following: `/visualization/plots & /visualization/plots_wandb1`\n"
                "   - in `/visualization/plots`, unzip `/visualization/plots/time_evol_global_Qx_v1` \n \t and `/visualization/plots/time_evol_global_Qx_v2`. Combine the content into a single folder \n \t `/visualization/plots/time_evol_global_Qx\n"
                "       (- this v1 & v2 were provided in seperate files due to size limit on github push.) \n"
                "   - More details in the README: 🔗 https://github.com/cevheck/Physic-Informed-DEKF/blob/main/README.md\n"
                f"{RESET}",
                stacklevel=2
            )
            sys.exit()
            raise Exception(f"File {savepath} has not been created yet. Run with load=False & save=True to generate the visualization pickle files. Alternatively, unzip the provided zipfiles at /visualization/plots as described in the README (https://github.com/cevheck/Physic-Informed-DEKF/blob/main/README.md). To create time_evol_global_Qx, one will also need to concatenate the v1 & v2, also described in the README")
    if not load:
        metric_good_results = defaultdict(list)
        results = results_dict['results']
        plotting_inputs = results_dict['plotting_info']
        
        trainable_names = plotting_inputs[0]['trainable_names']
        n_p = len(trainable_names)

        if plot_what not in trainable_names: 
            if plot_what != "eps" and plot_what != "Qx": raise Exception("Not implemented for variables other than the trainable param")
            if plot_what == "eps": val = jnp.array([jnp.abs(res_i['filtering']['x_error'][:,:,1,0]) for res_i in results])
            if plot_what == "Qx": val = jnp.array([jnp.abs(res_i['filtering']['Qx'][:,:,1,1]) for res_i in results])            
            var_true = jnp.zeros_like(val[:,:,0])
        else:
            var_true = jnp.array([pl_inp_i['p_true'][plot_what] for pl_inp_i in plotting_inputs])
            which_p = np.argwhere(np.array(trainable_names) == plot_what)[0,0]
            p_idx = -(n_p-which_p)      
            val = jnp.array([res_i['filtering']['w'][:,:,p_idx,0] for res_i in results])

        if "Qx" in selection_criteria or selection_criteria == 'innov':
            if "Qx" in selection_criteria:
                mw = 1  # optional moving window to average over during selection
                selection_data = jnp.array([res_i['filtering']['Qx'][:,:,1,1] for res_i in results])
            elif selection_criteria == "innov":
                mw = 1  # optional moving window to average over during selection
                selection_data = jnp.array([np.abs(res_i['filtering']['x_error'])[:,:,1,0] for res_i in results])
            if 'average3' in selection_criteria or 'weighted_average3' in selection_criteria:
                subselect_n_ensembles = 3
            if 'weighted_average3' in selection_criteria:
                return_best_idxs = True
                scanfunc = partial(scan_get_best_x_ensembles, subselect_n_ensembles=subselect_n_ensembles, mw=mw, return_best_idxs=True)
            else:
                scanfunc = partial(scan_get_best_x_ensembles, subselect_n_ensembles=subselect_n_ensembles, mw=mw)
            init = ()
            xs = (val, selection_data)
            carry, y = jax.lax.scan(scanfunc, init, xs)
            if 'weighted_average3' in selection_criteria:
                metric_good_results, metric_bad_results, best_idxs = y
            else:
                metric_good_results, metric_bad_results = y
            if 'average3' in selection_criteria:
                if 'weighted' in selection_criteria:
                    metric_good_results = jnp.transpose(metric_good_results, (0,2,1))
                    selection_data = jnp.array([np.abs(res_i['filtering']['x_error'])[:,:,1,0] for res_i in results])
                    weighting_data = selection_data[np.arange(selection_data.shape[0])[:, None, None], np.arange(selection_data.shape[1])[None, :, None], best_idxs]
                    weights = 1/weighting_data
                    sum_of_weights = jnp.sum(1/weighting_data, axis=-1)
                    weights_normalized = weights / sum_of_weights[:, :, None]
                    metric_good_results = jnp.sum(weights_normalized * metric_good_results, axis=-1)
                    metric_good_results = np.expand_dims(metric_good_results, axis=1)   
                else:
                    metric_good_results = jnp.mean(metric_good_results, axis=1)
                    metric_good_results = np.expand_dims(metric_good_results, axis=1)   
        elif selection_criteria == 'all':
            metric_good_results = val
        else:
            if selection_criteria == "average":
                val = jnp.mean(val, axis=-1)
            elif selection_criteria == 'weighted_average':
                selection_data = jnp.array([np.abs(res_i['filtering']['x_error'])[:,:,1,0] for res_i in results])
                weights = 1/selection_data
                sum_of_weights = jnp.sum(1/selection_data, axis=-1)
                weights_normalized = weights / sum_of_weights[:, :, None]
                val = jnp.sum(weights_normalized * val, axis=-1)
            else:
                raise Exception(f"Not implemented for selection criteria = {selection_criteria}. Only for ['Qx', 'innov', 'average', 'weighted_average']")
            metric_good_results = np.expand_dims(val, axis=1)   
        if np.unique(var_true, axis=1).shape[1] != 1: raise Exception("Not implemented for multiple unique values with JAX") # focus on case with single unique value each iter
        var_true = np.unique(var_true, axis=1).flatten()
        metric_results_all = {}
        for key in sorted(jnp.unique(var_true)):    # if multiple values of the (ground truth) variable are present, save per value (e.g. different V levels)
            idxs = jnp.argwhere(var_true == key)[:,0]
            key = str(round(key,2))
            vals = metric_good_results[idxs]
            metric_results_all[key] = np.array(vals)
            
        metric_results_all['var_true'] = sorted(jnp.unique(var_true))
        if save:
            savepath_dir = os.path.dirname(savepath)
            if not os.path.exists(savepath_dir): os.makedirs(savepath_dir) 
            if os.path.exists(savepath):
                with open(savepath, 'rb') as f:
                    metric_results_all_prev = pickle.load(f)      
                if metric_results_all.keys() != metric_results_all_prev.keys(): 
                    if overwrite:
                        with open(savepath, 'wb') as f:
                            pickle.dump(metric_results_all, f)
                        print(f"Results saved to {savepath}")
                    else:
                        raise Exception("Keys of the dictionaries are not the same")
                for key in metric_results_all:
                    if len(metric_results_all[key]) != len(metric_results_all_prev[key]) or not (np.all(metric_results_all[key] == metric_results_all_prev[key])):
                        if overwrite:
                            with open(savepath, 'wb') as f:
                                pickle.dump(metric_results_all, f)
                            print(f"Results saved to {savepath}")
                        else:
                            raise Exception(f"Values of the dictionaries are not the same for key {key}")
            else:   # path does not exist yet
                with open(savepath, 'wb') as f:
                    pickle.dump(metric_results_all, f)
                print(f"Results saved to {savepath}")
    if plot:
        var_true = metric_results_all['var_true']
        del metric_results_all['var_true']
        markers_on = int(3/dt)
        startlength = 0
        hline_x = []
        hline_y = []
        
        if plot_on_single_axis: # convert each variable to the deviation to their groundtruth variable to be able to plot it on single axis
            single_axis_metric_results = []
            for key_idx, (key) in enumerate(sorted(metric_results_all.keys())):
                var_true_key = np.array(var_true)[np.array([round(float(key), 2) == round(float(var_i), 2) for var_i in var_true])]
                metric_results = metric_results_all[key]
                if metric_results.shape[1] != 1: raise Exception("Only checked for single selected DEKF")
                metric_results = metric_results[:,0,:]
                if plot_error: metric_results = metric_results - var_true_key
                single_axis_metric_results.append(metric_results)
            all_single_axis_results = np.concatenate(single_axis_metric_results, axis=0)
            correct_shape_single_axis_results = np.expand_dims(all_single_axis_results, axis=1)
            metric_results_all = {'0': correct_shape_single_axis_results}
            var_true = [0.0]
        for key_idx, (key) in enumerate(sorted(metric_results_all.keys())):
            var_true_key = np.array(var_true)[np.array([round(float(key), 2) == round(float(var_i), 2) for var_i in var_true])]
            metric_results = metric_results_all[key]
            if metric_results.shape[1] != 1: raise Exception("Only checked for single selected DEKF")
            metric_results = metric_results[:,0,:]
            lengths = [len(metric_i) for metric_i in metric_results]
            maxlength = np.max(lengths)      

            if plot_error: metric_results = metric_results - var_true_key
            if plot_what == 'eps':
                if dt == 0.5:
                    mw = 1  # already working on subsampled version
                else:
                    # smooth out spiky results
                    mw = 100
                metric_results = moving_average(metric_results, mw)
            if padded:
                padded_metric_results = np.array([np.concatenate((result_i, [result_i[-1]] * (maxlength - len(result_i)))) for result_i in metric_results])
                mean = np.mean(padded_metric_results, axis=0)
                std = np.std(padded_metric_results, axis=0)
            elif plot_mean or plot_std:
                mean, std = calc_mean_std(metric_results, lengths, maxlength)       
            t = np.linspace(startlength*dt, (startlength+maxlength) * dt, maxlength)
            if first_one_model: names.append(f'${model}$')
            
            if heatmap:
                if padded:
                    ts_heatmap(padded_metric_results, t=t, ax=ax, normalize_y=normalize_y)
                else:
                    ts_heatmap(metric_results, t=t, ax=ax, normalize_y=normalize_y, dt=dt)
            if line_per_line:
                ti = [np.linspace(startlength*dt, (startlength+lengths[i]) * dt, lengths[i]) for i in range(len(metric_results))]
                if line_per_line == 'lines':
                    if False:   
                        if True:
                            print("Plotting only worst line atm")
                            worst_idx = np.argmax([np.abs(metric_results[i][-1] - float(key)) for i in range(len(metric_results))])
                            idx_plot = worst_idx
                        else:
                            print("Plotting only best line atm")
                            best_idx = np.argmin([np.abs(metric_results[i][-1] - float(key)) for i in range(len(metric_results))])
                            idx_plot = best_idx
                        ax.plot(ti[idx_plot], metric_results[idx_plot], color=color)
                    else:   # real line per line plot
                        [ax.plot(ti[i], metric_results[i], color=color, alpha=0.05) for i in range(len(metric_results))]
                elif line_per_line == 'scatter':
                    [ax.scatter(ti[i], metric_results[i], color=color, alpha=0.05) for i in range(len(metric_results))]
            if plot_std:
                # if 'n-' not in model:
                if True:
                    try:
                        ax.fill_between(t, mean-2*std, mean+2*std, color=color, alpha=0.3)
                    except:
                        ax.fill_between(t, mean-2*std, mean+2*std, alpha=0.3)
                    # ax.errorbar(t[::1000], mean[::1000], std[::1000], marker='s', mfc='red', mec='green', ms=20, mew=4)
                    # ax.errorbar(t[::1000], mean[::1000], 2*std[::1000], marker='s', color=color)
            if plot_mean:
                if key_idx == 0:
                    try:
                        ax.plot(t, mean, markevery=markers_on, marker=marker, markersize=10, label=f'${model}$', linestyle=style, linewidth=3, color=color)
                    except:
                        ax.plot(t, mean, markevery=markers_on, marker=marker, markersize=10, label=f'${model}$', linestyle=style, linewidth=3)
                else:
                    try:
                        ax.plot(t, mean, markevery=markers_on, marker=marker, markersize=10, linestyle=style, linewidth=3, color=color)
                    except:
                        ax.plot(t, mean, markevery=markers_on, marker=marker, markersize=10, linestyle=style, linewidth=3)
            if plot_what != "eps":  # get line to connect groundtruth
                hline_x.append((startlength)*dt)
                hline_x.append((startlength+maxlength)*dt)
                real_y = var_true_key if not plot_error else 0.0
                hline_y.append(real_y)
                hline_y.append(real_y)
            startlength += maxlength
        if plot_what != "eps": 
            if False:#plot_what == plot_axes: 
                ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=f'${plot_what}={var_i}$')
            elif plot_what == 'beta':
                ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=rf'${{\beta}}_{{\mathrm{{true}}}}$')
                # ax.plot(hline_x, hline_y, color='black', linestyle='-.')
            elif plot_what == 'H':
                # ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=rf'${plot_what}_{'true'}$')
                ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=f'${plot_what}_{{\mathrm{{true}}}}$')
                # ax.plot(hline_x, hline_y, color='black', linestyle='-.')
            elif plot_what == 'V':
                ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=r'$\tilde{{{0}}}_{{\mathrm{{true}}}}$'.format(plot_what))
            if print_end_mean:
                print(f"real y = {real_y}")
                try:
                    print(f"Mean error at end for model {model} = {np.abs(mean[-1] - real_y[0])}")
                except:
                    print(f"Mean error at end for model {model} = {np.abs(mean[-1] - real_y)}")
        set_layout(ax,plot_what, plot_axes, var_i, ax_idx=var_idx, metrics_plot=metrics_plot, plot_error=plot_error, sharey=sharey)

def pred_to_errorbars(results, model, ax, plot_axes, var_i, i=0, load=True, save=False, overwrite=False, plot=True, savepath=None, print_mean=False, return_results=False):
    # if 'blackbox' in model: plot_std = False        # avoid cluttering of plot
    if 'hybrid' in model: model = model.replace("hybrid_", 'H-')
    if 'physics' in model: model = model.replace("physics_", 'p-')
    if 'blackbox' in model: model = model.replace("blackbox_", 'n-')
    # if 'blackbox' in model and 'eps' not in plot_what: return # no H / Beta predictions on blackbox
    color = colors[0] if 'H-' in model else colors[1] if 'p-' in model else colors[2]
    
    if load:
        try:
            with open(savepath, 'rb') as f:
                results = pickle.load(f)
        except:
            import sys, warnings
            RED = "\033[91m"
            RESET = "\033[0m"

            warnings.warn(
                f"{RED}\n⚠️ WARNING: Missing File\n"
                f"File not found: {savepath}\n"
                "\nThis file has not been created yet. To resolve this issue, you have two options:\n"
                "1️⃣ **Generate the file:**\n"
                "   - Run the script with: `load=False, save=True`\n"
                "   - Requires all results in the folder results/ to be made (or unzipped). \n"
                "\n"
                "2️⃣ **Manually extract from the provided zip files:**\n"
                "   - Unzip the following: `/visualization/plots & /visualization/plots_wandb1`\n"
                "   - in `/visualization/plots`, unzip `/visualization/plots/time_evol_global_Qx_v1` \n \t and `/visualization/plots/time_evol_global_Qx_v2`. Combine the content into a single folder \n \t `/visualization/plots/time_evol_global_Qx\n"
                "       (- this v1 & v2 were provided in seperate files due to size limit on github push.) \n"
                "   - More details in the README: 🔗 https://github.com/cevheck/Physic-Informed-DEKF/blob/main/README.md\n"
                f"{RESET}",
                stacklevel=2
            )
            sys.exit()
    if not load:
        if save:
            if os.path.exists(savepath):
                if not overwrite: raise Exception(f"File {savepath} already exists")
            else:
                if not os.path.exists(os.path.dirname(savepath)): 
                    os.makedirs(os.path.dirname(savepath))
            with open(savepath, 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved to {savepath}")
    if plot:
        bplot = ax.boxplot([results],positions=[i], labels=[rf'${model}$'], vert=True, sym='',patch_artist=True)
        for patch in bplot["boxes"]:
            patch.set_facecolor(color)       
        for line in bplot["medians"]:
            line.set_color('k')      
        ax.set_yscale('log')
    if print_mean:
        print(f"Mean error for model {model}, {plot_axes} = {var_i}: {np.mean(results)}")
    if return_results: return results, colors

        
def dict_to_results(results_dict, ax, var_i, plot_axes, plot_what, result_metric, padded, k, subselect_n_ensembles, dt, heatmap, normalize_y, var_idx, line_per_line, plot_mean, plot_std, selection_criteria="Qx"):
    raise Exception("Outdated version. No guarantees on correct results have been checked in a long time. Use JAX=True for faster and more up to date version")
    names = []
    first_one_model = True
    for c1, (model, modelresults) in enumerate(results_dict.items()):
        if len(modelresults) == 0: return
        if 'hybrid' in model: model = model.replace("hybrid_", 'H-')
        if 'physics' in model: model.replace("physics", 'p-')
        if 'blackbox' in model and 'hat' in result_metric: continue # no H / Beta predictions on blackbox
        
        style = styles[0] if 'PDEKF' in model else styles[1]
        marker = 'v' if 'PDEKF' in model else 'None'
        color = colors[1] if 'p-' in model else colors[0] if 'H-' in model else colors[2]

        metric_bad_results = defaultdict(list)
        metric_good_results = defaultdict(list)
        results = modelresults['results']
        plotting_inputs = modelresults['plotting_info']
        for cam_nr, (res_i, pl_inp_i) in enumerate(zip(results, plotting_inputs)):
            n_ensemble = res_i['filtering']['x'].shape[1]
            trainable_names = pl_inp_i['trainable_names']
            n_p = len(trainable_names)
            if plot_what not in trainable_names: 
                if plot_what != "eps": raise Exception("Not implemented for variables other than the trainable param")
                val = np.abs(res_i['filtering']['x_error'])[:,:,1,0]
            else:
                var_true = pl_inp_i['p_true'][plot_what]
                which_p = np.argwhere(np.array(trainable_names) == plot_what)[0,0]
                p_idx = -(n_p-which_p)      
                val = res_i['filtering']['w'][:,:,p_idx,0]
            if selection_criteria == "Qx":
                mw = 1
                selection_data = res_i['filtering']['Qx'][:,:,1,1]
            elif selection_criteria == "innov":
                mw = 1000
                selection_data = np.abs(res_i['filtering']['x_error'])[:,:,1,0]
            else:
                raise Exception(f"Not implemented for selection criteria = {selection_criteria}. Only for 'Qx' and 'innov'")
            # comply with previous shape
            val = val.T
            selection_data = selection_data.T
            
            best_res, worst_res = get_best_x_ensembles(val, selection_data, subselect_n_ensembles, mw=mw)
            
            if plot_what == 'eps':
                metric_good_results[plot_what].extend(best_res)
                metric_bad_results[plot_what].extend(worst_res)
            elif plot_what == plot_axes: 
                assert(np.allclose(var_true, var_i))
                key = str(round(var_i,2))
                metric_good_results[key].extend(best_res)
                metric_bad_results[key].extend(worst_res)
            else:
                unique_vars = sorted(np.unique(var_true))
                for key in unique_vars:
                    idxs = np.argwhere(var_true == key).flatten()
                    key = str(round(key,2))
                    metric_good_results[key].extend(best_res[:,idxs])
                    metric_bad_results[key].extend(worst_res[:,idxs])
        metric_results_all = metric_good_results
        markers_on = int(3/dt)  # every 1s
        startlength = 0
        hline_x = []
        hline_y = []
        for key_idx, (key) in enumerate(sorted(metric_results_all.keys())):
            metric_results = metric_results_all[key]
            lengths = [len(metric_i) for metric_i in metric_results]
            maxlength = np.max(lengths)      

            if padded:
                padded_metric_results = np.array([np.concatenate((result_i, [result_i[-1]] * (maxlength - len(result_i)))) for result_i in metric_results])
                mean = np.mean(padded_metric_results, axis=0)
                std = np.std(padded_metric_results, axis=0)
            elif plot_mean or plot_std:
                ## if I want to calculate mean of all lasting lists
                means = []
                stds = []
                for i in range(maxlength):
                    vals = []
                    for results in metric_results:
                        try:
                            vals.append(results[i])
                        except:
                            pass
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                mean = np.array(means)
                std = np.array(stds)         
            t = np.linspace(startlength*dt, (startlength+maxlength) * dt, maxlength)
            if first_one_model: names.append(f'${model}$')
            
            if heatmap:
                if padded:
                    ts_heatmap(padded_metric_results, t=t, ax=ax, normalize_y=normalize_y)
                else:
                    ts_heatmap(metric_results, t=t, ax=ax, normalize_y=normalize_y, dt=dt)
            if line_per_line:
                ti = [np.linspace(startlength*dt, (startlength+lengths[i]) * dt, lengths[i]) for i in range(len(metric_results))]
                if line_per_line == 'lines':
                    if False:   
                        if True:
                            print("Plotting only worst line atm")
                            worst_idx = np.argmax([np.abs(metric_results[i][-1] - float(key)) for i in range(len(metric_results))])
                            idx_plot = worst_idx
                        else:
                            print("Plotting only best line atm")
                            best_idx = np.argmin([np.abs(metric_results[i][-1] - float(key)) for i in range(len(metric_results))])
                            idx_plot = best_idx
                        ax.plot(ti[idx_plot], metric_results[idx_plot], color=color)
                    else:
                        [ax.plot(ti[i], metric_results[i], color=color, alpha=0.05) for i in range(len(metric_results))]
                elif line_per_line == 'scatter' or True:
                    [ax.scatter(ti[i], metric_results[i], color=color, alpha=0.05) for i in range(len(metric_results))]
            if plot_std:
                ax.fill_between(t, mean-2*std, mean+2*std, color=color, alpha=0.1)
            if plot_mean:
                if key_idx == 0:
                    ax.plot(t, mean, color=color, markevery=markers_on, marker=marker, markersize=10, label=f'${model}$', linestyle=style, linewidth=3)
                else:
                    ax.plot(t, mean, color=color, markevery=markers_on, marker=marker, markersize=10, linestyle=style, linewidth=3)
            if plot_what != "eps":
                hline_x.append((startlength)*dt)
                hline_x.append((startlength+maxlength)*dt)
                hline_y.append(float(key))
                hline_y.append(float(key))
            startlength += maxlength
        if plot_what != "eps": 
            if plot_what == plot_axes: 
                ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=f'${plot_what}={var_i}$')
            else:
                # ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=rf'${plot_what}_{'true'}$')
                ax.plot(hline_x, hline_y, color='black', linestyle='-.', label=f'${plot_what}_{{\mathrm{{true}}}}$')
    if 'H' in plot_axes:
        units = r'\mathrm{m}'
        plot_what_string = rf'{plot_axes}'
    elif 'beta' in plot_axes:
        plot_what_string = rf'\{plot_axes}'
        units = r''
    else:
        raise Exception(f"Not implemented for {plot_axes} other variables than H and beta")
        
    ax.set_title(rf'${plot_what_string} = {str(round(var_i,2))}{units}$')
    # set_ticks(axes[var_idx],result_metric, ax_idx=var_idx)
    ax.legend()

def dict_to_pareto(results_dict, model, Q_NN_added_rel, Q_NN_idx, ax, var_i, plot_axes, plot_what, selection_criteria, subselect_n_ensembles, whichplot="type2", load=True, save=False, savefolder=None):
    if 'hybrid' in model: model = model.replace("hybrid_", 'H-')
    if 'physics' in model: model = model.replace("physics_", 'p-')
    if 'blackbox' in model: model = model.replace("blackbox_", 'n-')
    
    # style = styles[0] if 'PDEKF' in model else styles[1]
    # marker = 'v' if 'PDEKF' in model else 'None'
    # if whichplot == "type2":
    #     color = colors[0] if 'H-' in model else colors[1] if 'p-' in model else colors[2]
    # elif whichplot == "type1":
    #     color = colors[Q_NN_idx]

    if load:
        # savepath = os.path.join(savefolder, f'{model}_plot_{plot_what[0]}_{plot_what[1]}_{plot_axes}_{var_i}_Q_NN_added_rel_{Q_NN_added_rel}.pkl')
        metric_good_results = []
        for plot_what_i in plot_what:
            savepath_i = os.path.join(savefolder, f'{model}_plot_{plot_what_i}_{plot_axes}_{var_i}_Q_NN_added_rel_{Q_NN_added_rel}.pkl')
            if Q_NN_added_rel == 0.0:
                savepath_i = savepath_i.replace("_Q_NN_added_rel_0", "")
            try:
                with open(savepath_i, 'rb') as f:
                    metric_good_results_i = pickle.load(f)
            except:
                import sys
                import warnings
                            
                RED = "\033[91m"
                RESET = "\033[0m"

                warnings.warn(
                    f"{RED}\n⚠️ WARNING: Missing File\n"
                    f"File not found: {savepath_i}\n"
                    "\nThis file has not been created yet. To resolve this issue, you have two options:\n"
                    "1️⃣ **Generate the file:**\n"
                    "   - Run the script with: `load=False, save=True`\n"
                    "   - Requires all results in the folder results/ to be made (or unzipped). \n"
                    "\n"
                    "2️⃣ **Manually extract from the provided zip files:**\n"
                    "   - Unzip the following: `/visualization/plots & /visualization/plots_wandb1`\n"
                    "   - in `/visualization/plots`, unzip `/visualization/plots/time_evol_global_Qx_v1` \n \t and `/visualization/plots/time_evol_global_Qx_v2`. Combine the content into a single folder \n \t `/visualization/plots/time_evol_global_Qx\n"
                    "       (- this v1 & v2 were provided in seperate files due to size limit on github push.) \n"
                    "   - More details in the README: 🔗 https://github.com/cevheck/Physic-Informed-DEKF/blob/main/README.md\n"
                    f"{RESET}",
                    stacklevel=2
                )
                sys.exit()
            metric_good_results_i_true = metric_good_results_i['var_true']
            metric_good_results_ii = deepcopy(metric_good_results_i)
            del metric_good_results_ii['var_true']
            all_rel_metric_vals = []
            for key, val in metric_good_results_ii.items():
                dist = [np.abs(metric_good_results_i_true_i - float(key)) for metric_good_results_i_true_i in metric_good_results_i_true]
                # Finding the closest among var true
                min_closest_idx = np.argmin(dist)
                metric_good_results_i_true_i = metric_good_results_i_true[min_closest_idx]
                if plot_what_i != 'V': 
                    if plot_what_i == 'beta':
                        if np.abs(float(key) - metric_good_results_i_true_i) > 0.01: raise Exception("No close key (<0.000001) found")
                    else:
                        if np.abs(float(key) - metric_good_results_i_true_i) > 0.000001: raise Exception("No close key (<0.000001) found")
                elif plot_what_i == 'V': 
                    if np.abs(float(key) - round(metric_good_results_i_true_i,2)) > 0.000001: raise Exception("No close key found")
                metric_good_results_i_rel = jnp.abs(val - metric_good_results_i_true_i)
                all_rel_metric_vals.append(metric_good_results_i_rel)
            
            metric_good_results_i_rel = jnp.concatenate(all_rel_metric_vals)
            metric_good_results.append(metric_good_results_i_rel)
    else:
        raise Exception("Create savefiles first by running time_evolution.py with load=False, save=True for the required settings")

    metric_results_all = metric_good_results
    x_ax, y_ax = metric_results_all
    
    if not (len(x_ax) > 0): 
        print(f"No results found for {model}")
        return
    
    if True:   
        # # take mean over time
        # x_ax = np.mean(x_ax, axis=-1)
        # y_ax = np.mean(y_ax, axis=-1)
        
        # take mean over last values in time
        if x_ax.shape[2] == 120:
            # subsampled version (with factor 100, so only take last two values instead of last 200 for averaging)
            x_ax = np.mean(x_ax[:,:,-2:], axis=-1)
            y_ax = np.mean(y_ax[:,:,-2:], axis=-1)
        else:
            x_ax = np.mean(x_ax[:,:,-100:], axis=-1)
            y_ax = np.mean(y_ax[:,:,-100:], axis=-1)
    
    if True:    # average over best_n_ensemble (mostly = 1 anyways)
        x_ax = np.mean(x_ax, axis=-1)
        y_ax = np.mean(y_ax, axis=-1)
        
    if True:   # average over all simulations --> Qnn reduced to single pointvalue --> scatter
        x_ax = np.mean(x_ax)
        y_ax = np.mean(y_ax)
        print(f"Q_NN_added_rel = {Q_NN_added_rel} --> x = {x_ax}, y = {y_ax}")
        # ax.scatter(x_ax, y_ax, label=f'$Q_{{NN}} = {Q_NN_added_rel}$', color=color, marker=marker, linestyle=style)
        # ax.scatter(x_ax, y_ax, color=color, marker=marker, linestyle=style, s=10)
        # ax.scatter(x_ax, y_ax, color=color, marker=marker, linestyle=style)
    else:
        xmin = np.min(x_ax) - 0.008
        xmax = np.max(x_ax) + 0.008
        ymin = np.min(y_ax) - 0.01
        ymax = np.max(y_ax) + 0.01
        
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        
        import scipy.stats as st
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x_ax, y_ax])
        kernel = st.gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)
        cset = ax.contour(xx, yy, f, colors=color)
        # cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        # ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        # ax.clabel(cset, inline=1, fontsize=10)

        # for j in range(len(cset.allsegs)):
        #     for ii, seg in enumerate(cset.allsegs[j]):
        #         # plt.plot(seg[:,0], seg[:,1], '.-', color=color, label=f'Cluster{j}, level{ii}')
        #         ax.plot(seg[:,0], seg[:,1], '.-', color=color)
            

        
    if 'H' in plot_axes:
        units = r'\mathrm{m}'
        plot_what_string = rf'{plot_axes}'
    elif 'beta' in plot_axes:
        plot_what_string = rf'\{plot_axes}'
        units = r''
    else:
        raise Exception(f"Not implemented for {plot_axes} other variables than H and beta")
    
    title1 = plot_what[0] if plot_what[0] != 'Qx' else 'Q_x'
    title2 = plot_what[1] if plot_what[1] != 'Qx' else 'Q_x'
    ax.set_xlabel(f"$Mean \ {plot_what[0]} \ error$")
    ax.set_ylabel(f"$Mean \ {plot_what[1]} \ error$")
    
    ax.set_xlabel(f"Mean ${plot_what[0]}$ error")
    ax.set_ylabel(f"Mean ${plot_what[1]}$ error")
    
    ax.set_title(rf'${plot_what_string} = {str(round(var_i,2))}{units}$')
    # ax.legend(prop={'size': 15}, ncol=1, loc='upper right') 
    return x_ax, y_ax

def dict_to_QNN_on_x_axis(model, Q_NN_added_rel, Q_NN_idx, ax, ax2, var_i, plot_axes, plot_what, selection_criteria, subselect_n_ensembles, load=True, save=False, savefolder=None, Q_weight_added=1e-9):
    if 'hybrid' in model: model = model.replace("hybrid_", 'H-')
    if 'physics' in model: model = model.replace("physics_", 'p-')
    if 'blackbox' in model: model = model.replace("blackbox_", 'n-')

    if load:
        # savepath = os.path.join(savefolder, f'{model}_plot_{plot_what[0]}_{plot_what[1]}_{plot_axes}_{var_i}_Q_NN_added_rel_{Q_NN_added_rel}.pkl')
        metric_good_results = []
        for plot_what_i in plot_what:
            if "p-" in model:
                savepath_i = os.path.join(savefolder, f'{model}_plot_{plot_what_i}_{plot_axes}_{var_i}.pkl')
            else:
                savepath_i = os.path.join(savefolder, f'{model}_plot_{plot_what_i}_{plot_axes}_{var_i}_Q_NN_added_rel_{Q_NN_added_rel}.pkl')
                if Q_NN_added_rel == 0.0:
                    savepath_i = savepath_i.replace("_Q_NN_added_rel_0", "")
            with open(savepath_i, 'rb') as f:
                metric_good_results_i = pickle.load(f)
            metric_good_results_i_true = metric_good_results_i['var_true']
            metric_good_results_ii = deepcopy(metric_good_results_i)
            del metric_good_results_ii['var_true']
            all_rel_metric_vals = []
            for key, val in metric_good_results_ii.items():
                dist = [np.abs(metric_good_results_i_true_i - float(key)) for metric_good_results_i_true_i in metric_good_results_i_true]
                # Finding the closest among var true
                min_closest_idx = np.argmin(dist)
                metric_good_results_i_true_i = metric_good_results_i_true[min_closest_idx]
                if plot_what_i != 'V': 
                    if plot_what_i == 'beta':
                        if np.abs(float(key) - metric_good_results_i_true_i) > 0.01: raise Exception("No close key (<0.000001) found")
                    else:
                        if np.abs(float(key) - metric_good_results_i_true_i) > 0.000001: raise Exception("No close key (<0.000001) found")
                elif plot_what_i == 'V': 
                    if np.abs(float(key) - round(metric_good_results_i_true_i,2)) > 0.000001: raise Exception("No close key found")
                metric_good_results_i_rel = jnp.abs(val - metric_good_results_i_true_i)
                all_rel_metric_vals.append(metric_good_results_i_rel)
            
            metric_good_results_i_rel = jnp.concatenate(all_rel_metric_vals)
            metric_good_results.append(metric_good_results_i_rel)
    else:
        raise Exception("Create savefiles first by running time_evolution.py with load=False, save=True for the required settings")
    metric_results_all = metric_good_results
    x_ax, y_ax = metric_results_all
    
    Q_NN_added = Q_weight_added * Q_NN_added_rel
    if not (len(x_ax) > 0): 
        print(f"No results found for {model}")
        return
    
    if True:   
        # # take mean over time
        # x_ax = np.mean(x_ax, axis=-1)
        # y_ax = np.mean(y_ax, axis=-1)
        
        # take mean over last values in time
        x_ax = np.mean(x_ax[:,:,-100:], axis=-1)
        y_ax = np.mean(y_ax[:,:,-100:], axis=-1)
    
    if True:    # average over best_n_ensemble (mostly = 1 anyways)
        x_ax = np.mean(x_ax, axis=-1)
        y_ax = np.mean(y_ax, axis=-1)
        
    if True:   # average over all simulations --> Qnn reduced to single pointvalue --> scatter
        x_ax_std = np.std(x_ax, axis=-1)
        y_ax_std = np.std(y_ax, axis=-1)
        x_ax = np.mean(x_ax)
        y_ax = np.mean(y_ax)
    else:
        raise NotImplementedError("Not implemented for other than scatter plot")

    if 'H' in plot_axes:
        units = r'\mathrm{m}'
        plot_what_string = rf'{plot_axes}'
    elif 'beta' in plot_axes:
        plot_what_string = rf'\{plot_axes}'
        units = r''
    else:
        raise Exception(f"Not implemented for {plot_axes} other variables than H and beta")
    
    title1 = plot_what[0] 
    title2 = plot_what[1]
    error1 = error2 = False
    if title1 == 'Qx': 
        title1 = 'Q_{\omega}'
    elif title1 == 'eps': 
        title1 = '\epsilon'
    elif title1 == 'beta':
        title1 = r'\beta'
    else:
        error1 = True
    if title2 == 'Qx': 
        title2 = 'Q_{\omega}'
    elif title2 == 'eps': 
        title2 = '\epsilon'
    else:
        error2 = True
    alph = r"\alpha"
    ax.set_xlabel(rf"$Q_{alph}$", fontsize=fontsize)
    
    if error1:
        ax.set_ylabel(rf"${title1}$ error", color=colors[0], fontsize=fontsize)
    else:
        ax.set_ylabel(rf"${title1}$", color=colors[0], fontsize=fontsize)
    if error2:
        ax2.set_ylabel(rf"${title2}$ error", color=colors[1], fontsize=fontsize)
    else:
        ax2.set_ylabel(rf"${title2}$", color=colors[1], fontsize=fontsize)
    
    
    ax.set_title(rf'${plot_what_string} = {str(round(var_i,2))}{units}$', fontsize=fontsize)
    # ax.legend(prop={'size': 15}, ncol=1, loc='upper right') 
    
    xi = Q_NN_added
    yi = x_ax
    yi2 = y_ax
    yi_std = x_ax_std
    yi2_std = y_ax_std
    return xi, yi, yi2, yi_std, yi2_std, colors



def set_layout(ax, plot_what, plot_axes, var_i, ax_idx=None, metrics_plot=False, plot_error=False, sharey=False):
    set_all_axis_equal = False if metrics_plot else True
    ## x axis
    if sharey:
        xticks = [0, 20, 40, 60]
    else:
        xticks = [0, 10, 20, 30, 40, 50, 60]
    ax.set_xticks(xticks)
    xticks_new = [f"${round(tick, 2)}$" for tick in xticks]
    ax.set_xticklabels(xticks_new)
    
    ## title
    if 'H' in plot_axes:
        units = r'\mathrm{m}'
        plot_what_string = rf'{plot_axes}'
        title = rf'${plot_what_string} = {str(round(var_i,2))}{units}$'
    elif 'beta' in plot_axes:
        plot_what_string = rf'\{plot_axes}'
        units = r''
        if round(var_i, 2) == 1.57: title = rf'${plot_what_string} = \pi / 2$'
        if round(var_i, 2) == 2.36: title = rf'${plot_what_string} = 3\pi / 4$'
        if round(var_i, 2) == 3.14: title = rf'${plot_what_string} = \pi$'
        if round(var_i, 2) == 3.93: title = rf'${plot_what_string} = 5\pi / 4$'
    else:
        raise Exception(f"Not implemented for {plot_axes} other variables than H and beta")
    
    ## y-axis
    if sharey and ax_idx != 0:
        pass
    else:
        if 'eps' in plot_what:
            if 'eps_{f}' in plot_what:
                if ax_idx in [0, 1]:
                    yticks = [0, 1, 2, 3, 4, 5]
                    yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
                    ax.set_ylim(bottom=0.01, top=5)
                if ax_idx in [2, 3]:
                    yticks = [0, 2, 4, 6, 8, 10]
                    yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
                    ax.set_ylim(bottom=0.01, top=10)
                rad_title = r'\mathrm{[rad/s]}'
                ax.set_ylabel(rf'$\hat{{plot_what}} \ {rad_title}$', fontsize=fontsize)
            else:
                if ax_idx in [0]:
                    yticks = [0.01, 0.02, 0.03, 0.04]
                    ax.set_ylim(bottom=0.005, top=0.04)
                elif ax_idx in [1]:
                    yticks = [0.01, 0.03, 0.05, 0.07]
                    ax.set_ylim(bottom=0.01, top=0.07)
                else:
                    yticks = [0, 0.05, 0.10, 0.15]
                    ax.set_ylim(bottom=0.01, top=0.15)
                yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
                unit_title = r'\mathrm{[rad/s]}'
                plot_whatt = plot_what.replace('eps', 'epsilon')
                ax.set_ylabel(rf'$|{{\epsilon_\omega}}| \ {unit_title} $', fontsize=fontsize)
        elif 'H' in plot_what:
            if plot_error:
                # yticks_new = [r'$-0.01$', r'$0$', r'$0.01$']
                # yticks = [-0.01, 0, 0.01]
                yticks_new = [r'$-0.02$', r'$-0.01$', r'$0$', r'$0.01$', r'$0.02$']
                yticks = [-0.02, -0.01, 0, 0.01, 0.02]
            else:
                yticks_new = [r'$0.01$', r'$0.03$', r'$0.05$', r'$0.07$']
                yticks = [0.01, 0.03, 0.05, 0.07]
            # yticks_new = [f"${round(tick, 2)}$" for tick in yticks]

            if plot_error:
                pass
                # ax.set_ylim(bottom=-0.015, top=0.015)
                ax.set_ylim(bottom=-0.0015, top=0.0015) # resets to very low value each time, will be overwritten by last plot with quite low std
            elif set_all_axis_equal:
                ax.set_ylim(bottom=0.005, top=0.08)
            elif metrics_plot:  # bounds for the plot comparing multiple metrics
                if round(var_i,2) == 0.01: 
                    yticks_new = [r'$0.01$', r'$0.03$']
                    yticks = [0.01, 0.03]
                    ax.set_ylim(bottom=0.005, top=0.035)
                if round(var_i,2) == 0.03: 
                    yticks_new = [r'$0.02$', r'$0.03$', r'$0.04$']
                    yticks = [0.02, 0.03, 0.04]
                    ax.set_ylim(bottom=0.019, top=0.041)
                if round(var_i,2) == 0.05: 
                    yticks_new = [r'$0.03$', r'$0.05$']
                    yticks = [0.03, 0.05]
                    ax.set_ylim(bottom=0.028, top=0.055)
                if round(var_i,2) == 0.07:
                    yticks_new = [r'$0.03$', r'$0.05$', r'$0.07$']
                    yticks = [0.03, 0.05, 0.07] 
                    ax.set_ylim(bottom=0.028, top=0.075)
            else:               # bounds for the plot of variables through time
                if round(var_i,2) == 0.01: 
                    # yticks_new = [r'$0.01$', r'$0.03$']
                    # yticks = [0.01, 0.03]
                    ax.set_ylim(bottom=0.002, top=0.035)
                if round(var_i,2) == 0.03: 
                    # yticks_new = [r'$0.02$', r'$0.03$', r'$0.04$']
                    # yticks = [0.02, 0.03, 0.04]
                    ax.set_ylim(bottom=0.01, top=0.055)
                if round(var_i,2) == 0.05: 
                    # yticks_new = [r'$0.03$', r'$0.05$']
                    # yticks = [0.03, 0.05]
                    ax.set_ylim(bottom=0.028, top=0.08)
                if round(var_i,2) == 0.07:
                    # yticks_new = [r'$0.03$', r'$0.05$', r'$0.07$']
                    # yticks = [0.03, 0.05, 0.07] 
                    ax.set_ylim(bottom=0.0, top=0.1)
            unit_title = r'\mathrm{[m]}'
            if plot_error:
                ax.set_ylabel(rf'$\Delta{{{plot_what}}} \ {unit_title}$', fontsize=fontsize)
            else:
                ax.set_ylabel(rf'$\hat{{{plot_what}}} \ {unit_title}$', fontsize=fontsize)
        elif 'beta' in plot_what:
            if plot_error:
                yticks_new = [r'$- \frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$']
                yticks = [-np.pi / 2, 0, np.pi / 2]

                yticks_new = [r'$- \frac{\pi}{4}$', r'$0$', r'$\frac{\pi}{4}$']
                yticks = [-np.pi / 4, 0, np.pi / 4]
            else:
                yticks_new = [r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$']
                yticks = [np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4]
                
            # set_all_axis_equal = False
            if plot_error:
                ax.set_ylim(bottom=-1.8, top=1.8)
                ax.set_ylim(bottom=-0.0001, top=0.001)
                ax.set_ylim(bottom=-3*np.pi / 8, top=3*np.pi / 8)
            elif set_all_axis_equal:
                ax.set_ylim(bottom=1, top=4.5)
            elif metrics_plot:  # bounds for the plot comparing multiple metrics
                if round(var_i,2) == 1.57: 
                    yticks_new = [r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
                    yticks = [np.pi / 2, 3 * np.pi / 4, np.pi]
                    ax.set_ylim(bottom=1.5, top=3.55)
                if round(var_i,2) == 2.36: 
                    yticks_new = [r'$\frac{3\pi}{4}$', r'$\pi$']
                    yticks = [3 * np.pi / 4, np.pi]
                    ax.set_ylim(bottom=2.3, top=3.2)
                if round(var_i,2) == 3.14: 
                    yticks_new = [r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{9\pi}{8}$']
                    yticks = [3 * np.pi / 4, np.pi, 9 * np.pi / 8]
                    ax.set_ylim(bottom=2.3, top=3.6)
                if round(var_i,2) == 3.93: 
                    yticks_new = [r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$']
                    yticks = [3 * np.pi / 4, np.pi, 5 * np.pi / 4]
                    ax.set_ylim(bottom=2.3, top=4)
            else:               # bounds for the plot of variables through time
                if round(var_i,2) == 1.57: 
                    yticks_new = [r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
                    yticks = [np.pi / 2, 3 * np.pi / 4, np.pi]
                    ax.set_ylim(bottom=1, top=3.55)
                if round(var_i,2) == 2.36: 
                    yticks_new = [r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']
                    yticks = [np.pi / 2, 3 * np.pi / 4, np.pi]
                    yticks_new = [r'$\frac{3\pi}{4}$', r'$\pi$']
                    yticks = [3 * np.pi / 4, np.pi]
                    ax.set_ylim(bottom=2.3, top=3.2)
                if round(var_i,2) == 3.14: 
                    yticks_new = [r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{9\pi}{8}$']
                    yticks = [3 * np.pi / 4, np.pi, 9 * np.pi / 8]
                    ax.set_ylim(bottom=2.3, top=3.6)
                if round(var_i,2) == 3.93: 
                    yticks_new = [r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{5\pi}{4}$']
                    yticks = [3 * np.pi / 4, np.pi, 5 * np.pi / 4]
                    ax.set_ylim(bottom=2.3, top=4)
            unit_title = r''
            if plot_error:
                ax.set_ylabel(rf'$\Delta{{\{plot_what}}} \ {unit_title}$', fontsize=fontsize)
            else:
                ax.set_ylabel(rf'$\hat{{\{plot_what}}} \ {unit_title}$', fontsize=fontsize)
        elif 'V' in plot_what:
            yticks = [-2, -1, 0, 1, 2]
            yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
            ax.set_ylim(bottom=-3, top=3)
            ax.set_ylabel(r'$\Delta{V} \ [\mathrm{V}]$', fontsize=fontsize)
        elif 'Qx' in plot_what:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.set_xlabel(r'$\mathrm{Time \ [s]}$', fontsize=fontsize)
            return None  # skip y-axis, this is not for clean plotting in any of the results anyways
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks_new)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    if 'beta' in plot_what:
        ax.tick_params(axis='y', which='major', labelsize=50)

    ax.set_xlabel(r'$\mathrm{Time \ [s]}$', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    return None

def set_layout_predictionplot(ax, model, plot_axes, var_i):
    # plot_what = rf'\fontsize{30}$MSE$ \ \fontsize{50}${{\epsilon_{{\omega, f}}}}$'
    plot_what = rf'$MSE({{\epsilon_{{\omega, f}}}})$'
    plot_what = rf'$MSE \ {{\epsilon_{{\omega, f}}}}$'
    if 'H' in plot_axes:
        units = r'\mathrm{m}'
        plot_what_string = rf'{plot_axes}'
    elif 'beta' in plot_axes:
        plot_what_string = rf'\{plot_axes}'
        units = r''
    else:
        raise Exception(f"Not implemented for {plot_axes} other variables than H and beta")
    
    if 'H' in plot_axes:
        units = r'\mathrm{m}'
        plot_what_string = rf'{plot_axes}'
        title = rf'${plot_what_string} = {str(round(var_i,2))}{units}$'
    elif 'beta' in plot_axes:
        plot_what_string = rf'\{plot_axes}'
        units = r''
        if round(var_i, 2) == 1.57: title = rf'${plot_what_string} = \pi / 2$'
        if round(var_i, 2) == 2.36: title = rf'${plot_what_string} = 3\pi / 4$'
        if round(var_i, 2) == 3.14: title = rf'${plot_what_string} = \pi$'
        if round(var_i, 2) == 3.93: title = rf'${plot_what_string} = 5\pi / 4$'
    else:
        raise Exception(f"Not implemented for {plot_axes} other variables than H and beta")
    ax.set_title(title, fontsize=fontsize)
    
    xticks = ax.get_xticklabels()
    ax.set_xticklabels(xticks, fontsize=20)
    
    yticks = ax.get_yticks()
    
    def log_10_product(x):
        return f'$10^{{{int(np.log10(x))}}}$'
    # yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
    yticks_new = [log_10_product(tick) for tick in yticks]
    ax.set_yticklabels(yticks_new, fontsize=labelsize)
    def on_resize(event):
        # Update tick labels when resizing
        yticks = ax.get_yticks()
        # yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
        yticks_new = [log_10_product(tick) for tick in yticks]
        ax.set_yticklabels(yticks_new, fontsize=labelsize-10)
    fig = ax.figure
    # Connect to the resize event of the canvas
    fig.canvas.mpl_connect('resize_event', on_resize)   # when resizing, reformat the yticks

    ax.set_ylabel(plot_what, fontsize=fontsize)
    ax.set_ylim(bottom=0.0)
    # ax.tick_params(axis='both', which='major', labelsize=labelsize)
    return None

def set_plot_bounds(axes, result_metric, where='upper'):
    if True:
        if 'H' in result_metric:
            bottom = 0.005
            top = 0.08
        elif 'beta' in result_metric:
            bottom = 1.5
            top = 4.0
        for ax in axes:
            ax.set_ylim(bottom=bottom, top=top)

def fix_legend_pareto(ax, where='lower', Q_NN_added_rel_opts=None):
    if where == 'upper':
        ax.legend(prop={'size': 20}, ncol=1, loc='upper right') 
    elif where == 'lower':
        ax.legend(prop={'size': 15}, ncol=1, loc='lower right') 
    else: 
        raise NotImplementedError
    
def fix_legend_metrics(ax, selection_criteria_opts, plotting_order, fig=None):
    plotting_order = np.argsort(plotting_order)
    handles, labels = ax.get_legend_handles_labels()
    handles_groundtruth = [handle for (handle, label) in zip(handles, labels) if 'true' in label]
    labels_groundtruth = [label for label in labels if 'true' in label]
    if len(np.unique(labels_groundtruth)) > 1: raise Exception("Multiple groundtruth labels found")
    label_groundtruth = labels_groundtruth[0]
    handle_groundtruth = handles_groundtruth[0]
    other_handles = [handle for handle, label in zip(handles, labels) if 'true' not in label]
    other_handles = list(np.array(other_handles)[plotting_order])
    ordered_labels = ['${Q_x}$', '$|\epsilon_w|$', '$\mathcal{A}$', '$\mathcal{WA}$', '$\mathcal{A}(Q_{x,3}$)', '$\mathcal{WA}(Q_{x,3}$)']
    if len(ordered_labels) != len(other_handles): raise Exception("Not all labels found")

    fig.legend(title='Selection function $\mathcal{S}$', handles=other_handles, labels=ordered_labels, prop={'size': 20}, ncol=1, loc=7) 
    # fig.legend(handles=[handle_groundtruth], labels=[label_groundtruth], prop={'size': 20}, ncol=1, loc=1) 
    # fig.legend(handles=[handle_groundtruth], labels=[label_groundtruth], prop={'size': 20}, ncol=1, loc=1, bbox_to_anchor=(0.88, 0.8))
    ax.legend(handles=[handle_groundtruth], labels=[label_groundtruth], prop={'size': 20}, ncol=1, loc=1)
    fig.subplots_adjust(right=0.8)
      
def fix_legend(ax, where='upper'):
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels = ['H-PDEKF', 'p-PDEKF', 'n-PDEKF', 'true']

    # Create a dictionary to store unique handles and their corresponding labels
    sorted_handles = []
    sorted_labels = []
    for label_i in ordered_labels:  # Look for labels containing "HDEKF"
        for handle, label in zip(handles, labels):
            if label_i in label:
                sorted_handles.append(handle)
                sorted_labels.append(label)
                break


    if where == 'upper':
        # ax.legend(handles=sorted_handles, labels=sorted_labels, prop={'size': 20}, ncol=1, loc='upper right') 
        ## place a little bit higher to be out of the plot
        ax.legend(handles=sorted_handles, labels=sorted_labels, prop={'size': 20}, ncol=1, loc='upper right', bbox_to_anchor=(1, 1.2)) 
    elif where == 'lower':
        ax.legend(handles=sorted_handles, labels=sorted_labels, prop={'size': 25}, ncol=3, loc='lower right') 
    else: 
        raise NotImplementedError
      
def increase_plot_bounds(axes, where='upper'):
    pos0, pos1, pos2, pos3 = axes[0].get_position(), axes[1].get_position(), axes[2].get_position(), axes[3].get_position()
    pos0.y0 += 0.06     # increase the lower bound of the first plot
    pos0.y1 += 0.06     # increase the upper bound of the first plot
    pos1.y0 += 0.06 
    pos1.y1 += 0.06
    pos2.y1 -= 0.04
    pos3.y1 -= 0.04
    pos2.y0 -= 0.04
    pos3.y0 -= 0.04

    pos0.x0 -= 0.02     # decrease the left bound of the first plot
    pos0.x1 -= 0.02     # decrease the right bound of the first plot
    pos2.x0 -= 0.02
    pos2.x1 -= 0.02
    pos1.x0 += 0.02
    pos1.x1 += 0.02
    pos3.x0 += 0.02
    pos3.x1 += 0.02

    axes[0].set_position(pos0)
    axes[1].set_position(pos1)
    axes[2].set_position(pos2)
    axes[3].set_position(pos3)

def increase_plot_bounds_multiple_crit(axes, where='upper'):
    pos0, pos1, pos2, pos3 = axes[0].get_position(), axes[1].get_position(), axes[2].get_position(), axes[3].get_position()
    pos0.y0 += 0.03     # increase the lower bound of the first plot
    pos0.y1 += 0.03     # increase the upper bound of the first plot
    pos1.y0 += 0.03 
    pos1.y1 += 0.03
    pos2.y1 -= 0.02
    pos3.y1 -= 0.02

    pos0.x0 -= 0.02     # decrease the left bound of the first plot
    pos0.x1 -= 0.02     # decrease the right bound of the first plot
    pos2.x0 -= 0.02
    pos2.x1 -= 0.02
    pos1.x0 += 0.02
    pos1.x1 += 0.02
    pos3.x0 += 0.02
    pos3.x1 += 0.02

    axes[0].set_position(pos0)
    axes[1].set_position(pos1)
    axes[2].set_position(pos2)
    axes[3].set_position(pos3)
    
def increase_plot_bounds_pareto(axes, where='upper'):
    try:
        pos0, pos1, pos2, pos3 = axes[0].get_position(), axes[1].get_position(), axes[2].get_position(), axes[3].get_position()
    except:
        pos0, pos1 = axes[0].get_position(), axes[1].get_position()

    pos0.y0 += 0.09
    pos0.y1 += 0.07
    pos1.y0 += 0.09
    pos1.y1 += 0.07
    pos2.y1 -= 0.02
    pos3.y1 -= 0.02

    pos0.x0 -= 0.02
    pos0.x1 -= 0.02
    pos2.x0 -= 0.02
    pos2.x1 -= 0.02
    # pos1.x0 += 0.02
    # pos1.x1 += 0.02
    # pos3.x0 += 0.02
    # pos3.x1 += 0.02

    axes[0].set_position(pos0)
    axes[1].set_position(pos1)
    axes[2].set_position(pos2)
    axes[3].set_position(pos3)