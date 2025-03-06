"""
(Figure 4a and Figure 5)

To recreate images, run with load=True, save=False
    - with plot_whats = ["eps"] (Figure 4a)
    - with plot_whats = ["H","beta","V"] (Figure 5)
Takes best n ensemble members (based on 'selection_criteria') and plot their parameter variation over time.
save: save these results for later usage (also on other files e.g. for pareto plots) 
    - can iterate over all possible parameters by adding each to the list of 'plot_whats' 
    - already did this in order to allow you to use "load=True"
load: load pre-saved results (if available; else will return error)
plot: plot the resulting time evolution 
    - only makes sense when looking at a single 'plot_whats' and a single Q_NN_added_rel_opts for each modeltype: 'Q_NN_added_rel_opts_each'
    - set to False when creating all of the results for the first time (save=True; load=False situation)
possible parameters
    - H
    - beta
    - V
    - eps
    - Qx
possible ways of plotting
    - heatmap       ! heatmap for multiple models is not possible, plot would get overwritten with latest model !
    - line per line (each simulation is a line)
    - mean (and std)
    
Additional option for saving the conversion and loading for later re-use.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import gc
from copy import deepcopy
#%%
subselect_n_ensembles = 1   
selection_criteria = 'Qx'   # 'Qx' or 'innov', 'average', 'weighted_average'

#%% load/save settings
JAX = True      # use JAX version to speed up conversion
load = True    # load pre-saved results (if available; else will return error)
save = False     # save results for later usage with load = True
plot = True     # plot results
overwrite = True
wandb_for_bb = True
print_current = False
print_end_mean = True
#%% which models
modeltypes = ['hybrid_PDEKF', 'physics_PDEKF', 'blackbox_PDEKF']

if plot:    # plot with some chosen values of Q_NN_added_rel_opts
    Q_NN_added_rel_opts_each = {}        # if provided, overwrite Q_NN_added_rel_opts with this (optimal) value for each of the models
    Q_NN_added_rel_opts_each['hybrid_PDEKF'] = 0.001
    # Q_NN_added_rel_opts_each['hybrid_PDEKF'] = 0.0
    Q_NN_added_rel_opts_each['physics_PDEKF'] = 0
    if wandb_for_bb:
        Q_NN_added_rel_opts_each['blackbox_PDEKF'] = 0.00001
    else:
        Q_NN_added_rel_opts_each['blackbox_PDEKF'] = 0.001
else:    # just create all the results
    Q_NN_added_rel_opts_each = None        # if not provided, iterates over Q_NN_added_rel_opts as given below
Q_NN_added_rel_opts = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]  # Loop through all options (used when plot = False to just pre-create and save all results to be loaded later in this and other files)  
Q_NN_added_rel_opts = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]  # Loop through all options (used when plot = False to just pre-create and save all results to be loaded later in this and other files)  
if Q_NN_added_rel_opts_each: Q_NN_added_rel_opts = [None]
global_model_opts = [True]

switch_models = False    # switch order to have more clear view on hybrid uncertainty (smaller than physics)
if switch_models:
    modeltypes = modeltypes[::-1]
    Q_NN_added_rel_opts_each = Q_NN_added_rel_opts_each[::-1]

#%% what to plot
beta_opts = [np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]
H_opts = [0.01, 0.03, 0.05, 0.07]
plot_axes_base = 'H'  # 'H', 'beta' # the selection of subsets per axis based on this variable (in paper = H and Beta)
plot_whats = ['H', 'beta', 'V']  # which evolution(s) to plot in time
# plot_whats = ['eps']  # which evolution(s) to plot in time
plot_whats = ['H', 'V', 'beta', 'eps']  # which evolution(s) to plot in time

#%% how to plot
sharey = False          # share y axis between all plots    (used for plot_whats = ['H', 'beta', 'V'] // not used for plot_whats = ['eps'])
plot_relative = True   # plot variable - variable_setpoint
heatmap = False         # plot heatmap instead of lines
if heatmap: assert(len(modeltypes) == 1), "! heatmap for multiple models is not possible, plot would get overwritten with latest model !"
padded = False          # elongate finished time series with last value (padding)
normalize_y = True      # normalize y axis of heatmap 
# line_per_line = "scatter" # "lines" or "scatter" or True (=scatter)
line_per_line = False
plot_mean = True
plot_std = {}        # if provided, overwrite Q_NN_added_rel_opts with this (optimal) value for each of the models
plot_std['hybrid_PDEKF'] = True
plot_std['physics_PDEKF'] = True
plot_std['blackbox_PDEKF'] = True
#%% imports
import pathmagic as pathmagic
PlotFolder, ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()
savefolder_start = os.path.join(PlotFolder, f'time_evol_{selection_criteria}')
import setup_src
from plot_util import get_selected_results, dict_to_results, dict_to_results_JAX, increase_plot_bounds, fix_legend
#%% Config Object
from configuration import MyConfig as MC 
from configuration import update_MC, configurate_MC

func = dict_to_results_JAX if JAX else dict_to_results

for plot_what in plot_whats:
    if plot_what != "beta": # plot_axes_base == 'H'; always plot against H selection unless we are plotting beta
        plot_axes = plot_axes_base
    else:
        plot_axes = 'beta'    # plot beta against beta selection
    #%% 
    if plot_axes == 'beta':
        var_opts = beta_opts
    elif plot_axes == 'H':
        var_opts = H_opts
    
    if plot_what == 'V' or plot_relative:
        plot_error = True
    plot_on_single_axis = True if plot_what == 'V' else False   # combine multiple V-values on single plot (i.f.o. H for example)

    #%%
    if plot:
        if sharey:
            fig, axes = plt.subplots(1,4, sharey=True, figsize=(30, 5))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2,2)
            axes = axes.flatten()

    for global_model in global_model_opts:
        savefolder = deepcopy(savefolder_start)
        if not global_model: raise Exception("To be validated")
        if global_model:
            savefolder = savefolder.replace('time_evol', 'time_evol_global') 

        for var_idx, var_i in enumerate(var_opts):
            if print_current: print(f"Creating plot for {plot_axes} = {var_i} ")
            for Q_NN_idx, Q_NN_added_rel in enumerate(Q_NN_added_rel_opts):
                if print_current and Q_NN_added_rel != None: print(f"Creating results for Q_NN_added_rel = {Q_NN_added_rel} ")
                for model_idx, modeltype in enumerate(modeltypes):
                    if Q_NN_added_rel_opts_each != None:
                        Q_NN_added_rel = Q_NN_added_rel_opts_each[modeltype]
                        if print_current and Q_NN_added_rel != None: print(f"Creating results for model={modeltype} and Q_NN_added_rel = {Q_NN_added_rel} ")
                    plot_std_i = plot_std[modeltype]
                    # print(f"Creating results for modeltype = {modeltype} ")
                    model, KFtype = modeltype.split('_')
                    #%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
                    MC_i = MC()
                    configurate_MC(MC_i)

                    #%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
                    update_MC(MC_i, which_model=model, which_kalman_opt=KFtype, wandb_results=0)
                    EnsembleSave_i = setup_src.get_ensemble_save_path(ResultFolder, MC_i)
                    if Q_NN_added_rel != 0 and model != 'physics':
                        MC.Q_NN_added_rel = Q_NN_added_rel
                        EnsembleSave_Qsetting = EnsembleSave_i + f'_Q_NN_added_rel_{Q_NN_added_rel}/'  
                    else:
                        EnsembleSave_Qsetting = EnsembleSave_i 

                    modeltype_path_i = os.path.join(EnsembleSave_Qsetting, modeltype)
                    savepath_i = os.path.join(savefolder, f'{modeltype}_plot_{plot_what}_{plot_axes}_{var_i}.pkl')
                    if 'hybrid' in model: savepath_i = savepath_i.replace("hybrid_", 'H-')
                    if 'physics' in model: savepath_i = savepath_i.replace("physics_", 'p-')
                    if 'blackbox' in model: savepath_i = savepath_i.replace("blackbox_", 'n-')
                    if Q_NN_added_rel != 0 and model != 'physics': savepath_i = savepath_i.replace(".pkl", f"_Q_NN_added_rel_{Q_NN_added_rel}.pkl")

                    if model == 'blackbox' and wandb_for_bb == True:
                        savepath_i = savepath_i.replace("plots", "plots_wandb1")
                        savefolder_i = savefolder.replace('/plots/', '/plots_wandb1/') 
                        modeltype_path_i = modeltype_path_i.replace("results/", "results_wandb1/")
                    else:
                        savefolder_i = savefolder
                        
                    if not load:    # if we are not loading, we still need to create the results
                        selected_results = get_selected_results(var=var_i, varname=plot_axes, modeltype_path=modeltype_path_i, MC=MC)
                        if len(selected_results['results']) == 0: continue
                    else:
                        selected_results = None
                    
                    if plot:
                        ax = axes[var_idx]
                    else: 
                        ax = None
                    func(selected_results,  ax=ax, model=modeltype, var_i=var_i, plot_axes=plot_axes, plot_what=plot_what, padded=padded, 
                                                        subselect_n_ensembles=subselect_n_ensembles, 
                                                        dt=MC.dt, heatmap=heatmap, normalize_y=normalize_y, var_idx=var_idx,
                                                        line_per_line=line_per_line, plot_mean=plot_mean, plot_std=plot_std_i,
                                                        selection_criteria=selection_criteria, plot=plot, sharey=sharey,
                                                        savefolder=savefolder_i, load=load, save=save, overwrite=overwrite,
                                                        plot_error=plot_error, plot_on_single_axis=plot_on_single_axis, savepath=savepath_i, print_end_mean=print_end_mean)
                    
                    gc.collect()
                    del selected_results
                    gc.collect()
    if plot: 
        def on_resize(event):
            # Update tick labels when resizing
            # yticks = ax.get_yticks()
            # yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
            # ax.set_yticklabels(yticks_new, fontsize=38)
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.7)
        fig.canvas.mpl_connect('resize_event', on_resize)   # when resizing, reformat the yticks

        plt.tight_layout()
        # fix_legend(axes[0])
        # increase_plot_bounds(axes)
        # fig.set_size_inches((30, 15), forward=False)
        plt.savefig(f'/home/cedric/Downloads/{plot_what}.png')
        plt.show()
        plt.close()

