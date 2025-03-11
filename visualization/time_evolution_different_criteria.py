"""
(Figure 6)
Same functioning as time_evolution.py
    - see comments there for more information on each variable's purpose
    
Run once with 
    plot_whats = ['H']
    plot_whats = ['beta']
    
Adaptation to this script to plot the time evolution of the hybrid model with different selection criteria in a single plot
    - Criteria:
        - Qx: selection based on the (adaptive) process uncertainty, Qx
        - innov: selection based on the innovation value (moving window can be added in plot_util.py)
        - average: selection based on the average of the ensemble members
        - weighted_average: selection based on the weighted average of the ensemble. Weighted based on innovation values
        - average3_Qx: selection of 3 ensemble members based on the Qx value and additionally averaged
        - weighted_average3_Qx: selection of 3 ensemble members based on the Qx value and additionally weighted averaged
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import gc
from copy import deepcopy
#%%
subselect_n_ensembles = 1

#%% load/save settings
JAX = True
load = True
save = False
plot = True
overwrite = True
wandb_for_bb = True

#%% how to plot
sharey = True          # share y axis between all plots
plot_relative = True   # plot variable - variable_setpoint
heatmap = False         # plot heatmap instead of lines
padded = False          # elongate finished time series with last value (padding)
normalize_y = True      # normalize y axis of heatmap 
# line_per_line = "scatter" # "lines" or "scatter" or True (=scatter)
line_per_line = False
plot_mean = True
plot_std = False

#%% imports
import pathmagic as pathmagic
PlotFolder, ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()

import setup_src
from plot_util import get_selected_results, dict_to_results, dict_to_results_JAX, increase_plot_bounds_multiple_crit, fix_legend_metrics
#%% Config Object
from configuration import MyConfig as MC 
from configuration import update_MC, configurate_MC

## saved data works with subsample ratio of 100 --> for correct time axis need to adapt here aswell (see README.md for more details)
MC.dt = MC.dt * 100
func = dict_to_results_JAX if JAX else dict_to_results

selection_criteria_opts = ['Qx', 'innov', 'average', 'weighted_average']
selection_criteria_opts = ['Qx', 'innov', 'average', 'weighted_average', 'average3_Qx', 'weighted_average3_Qx']
plt.rcParams['axes.prop_cycle'] = plt.rcParams['axes.prop_cycle'][:len(selection_criteria_opts)]


if True:    # change plotting order to get other colors in the background
    selection_criteria_opts_new = []
    color_opts = []
    plotting_order = [1,3,2,5,4,0]
    for selection_criteria in plotting_order:   # plot in different order to get other colors in the background
        selection_criteria_opts_new.append(selection_criteria_opts[selection_criteria])
        color_opts.append(plt.rcParams['axes.prop_cycle'].by_key()['color'][selection_criteria])
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_opts)
    selection_criteria_opts = selection_criteria_opts_new

if plot:
    if sharey:
        fig, axes = plt.subplots(1,4, sharey=True, figsize=(30, 7))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2,2)
        axes = axes.flatten()
    
# selection_criteria_opts = ['average', 'weighted_average', 'average3_Qx', 'weighted_average3_Qx']
for selection_criteria in selection_criteria_opts:
    savefolder_start = os.path.join(PlotFolder, f'time_evol_{selection_criteria}')
    #%% which models
    # modeltypes = ['hybrid_PDEKF', 'physics_PDEKF', 'blackbox_PDEKF']
    modeltypes = ['hybrid_PDEKF']   # show comparison only for hybrid model
        
    #%% Q_NN for each modeltype
    Q_NN_added_rel_opts_each = {}        # if provided, overwrite Q_NN_added_rel_opts with this (optimal) value for each of the models
    Q_NN_added_rel_opts_each['hybrid_PDEKF'] = 0.001
    Q_NN_added_rel_opts_each['physics_PDEKF'] = 0
    if wandb_for_bb:
        Q_NN_added_rel_opts_each['blackbox_PDEKF'] = 0.00001
    else:
        Q_NN_added_rel_opts_each['blackbox_PDEKF'] = 0.001

    global_model_opts = [True]

    #%% what to plot
    beta_opts = [np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]
    H_opts = [0.01, 0.03, 0.05, 0.07]
    plot_axes_base = 'H'  # 'H', 'beta'
    plot_whats = ['H']  # 'beta' # 'H' # 'eps
    plot_whats = ['beta']  # 'beta' # 'H' # 'eps
    Q_NN_added_rel_opts = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    Q_NN_added_rel_opts = [0]

    if Q_NN_added_rel_opts_each: Q_NN_added_rel_opts = [None]

    for plot_what in plot_whats:
        if plot_what != "beta":
            plot_axes = plot_axes_base
        else:
            plot_axes = 'beta'
        #%% 
        if plot_axes == 'beta':
            var_opts = beta_opts
        elif plot_axes == 'H':
            var_opts = H_opts
        
        if plot_what == 'V' or plot_relative:
            plot_error = True
        plot_on_single_axis = True if plot_what == 'V' else False

        # for global_model in [False, True]:
        for global_model in global_model_opts:
            savefolder = deepcopy(savefolder_start)
            if not global_model: raise Exception("To be validated")
            if global_model:
                savefolder = savefolder.replace('time_evol', 'time_evol_global') 

            for var_idx, var_i in enumerate(var_opts):
                print(f"Creating plot for {plot_axes} = {var_i} ")
                for Q_NN_idx, Q_NN_added_rel in enumerate(Q_NN_added_rel_opts):
                    print(f"Creating results for Q_NN_added_rel = {Q_NN_added_rel} ")
                    for model_idx, modeltype in enumerate(modeltypes):
                        if Q_NN_added_rel_opts_each != None:
                            # Q_NN_added_rel = Q_NN_added_rel_opts_each[model_idx][0]
                            Q_NN_added_rel = Q_NN_added_rel_opts_each[modeltype]
                        print(f"Creating results for modeltype = {modeltype} ")
                        model, KFtype = modeltype.split('_')
                        #%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
                        MC_i = MC()
                        configurate_MC(MC_i)

                        #%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
                        update_MC(MC_i, which_model=model, which_kalman_opt=KFtype, wandb_results=0)
                        EnsembleSave_i = setup_src.get_ensemble_save_path(ResultFolder, MC_i)
                        # if Q_NN_added_rel != 0 and model == 'hybrid':
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
                        # if Q_NN_added_rel != 0 and model == 'hybrid': savepath_i = savepath_i.replace(".pkl", f"_Q_NN_added_rel_{Q_NN_added_rel}.pkl")
                        
                        if model == 'blackbox' and wandb_for_bb == True:
                            savepath_i = savepath_i.replace("plots", "plots_wandb1")
                            savefolder_i = savefolder.replace('/plots/', '/plots_wandb1/') 
                            modeltype_path_i = modeltype_path_i.replace("results/", "results_wandb1/")
                        else:
                            savefolder_i = savefolder
                        
                        if not load:
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
                                                            line_per_line=line_per_line, plot_mean=plot_mean, plot_std=plot_std,
                                                            selection_criteria=selection_criteria, plot=plot, sharey=sharey,
                                                            savefolder=savefolder, load=load, save=save, overwrite=overwrite,
                                                            plot_error=plot_error, plot_on_single_axis=plot_on_single_axis, savepath=savepath_i, metrics_plot=True)
                        
                        gc.collect()
                        del selected_results
                        gc.collect()
if plot: 
    # axes[0].legend
    # increase_plot_bounds_multiple_crit(axes)
    # fig.set_size_inches((30, 15), forward=False)
    def on_resize(event):
        # Update tick labels when resizing
        # yticks = ax.get_yticks()
        # yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
        # ax.set_yticklabels(yticks_new, fontsize=38)
        
        plt.tight_layout()
        if False: #save_legend:
            fix_legend_metrics(axes[0], selection_criteria_opts, plotting_order=plotting_order, fig=fig)
            plt.savefig(f'/home/cedric/Downloads/{plot_what}_allmetrics.png')

    fig.canvas.mpl_connect('resize_event', on_resize)   # when resizing, reformat the yticks
    # fix_legend_metrics(axes[0], selection_criteria_opts, plotting_order=plotting_order, fig=fig)
    
    plt.tight_layout()
    plt.savefig(f'/home/cedric/Downloads/{plot_what}.png')

    plt.show()
    plt.close()

