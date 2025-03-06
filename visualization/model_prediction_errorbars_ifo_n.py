"""
(Figure B.9)
Similar to model_prediction_errorbars.py, but now for multiple values of Q_alpha.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import gc
from copy import deepcopy
import matplotlib as mpl

#%% plot settings
share_y_axis = False

#%%
subselect_n_ensembles = 1
selection_criteria = 'Qx'   # 'Qx' or 'innov'
wandb_for_bb = True
print_mean = True
#%% which models
modeltypes = ['hybrid_PDEKF', 'physics_PDEKF', 'blackbox_PDEKF']
# modeltypes = ['hybrid_PDEKF', 'blackbox_PDEKF']

Q_NN_added_rel_opts = {}        # if provided, overwrite Q_NN_added_rel_opts with this (optimal) value for each of the models
Q_NN_added_rel_opts['hybrid_PDEKF'] = [0.001]
Q_NN_added_rel_opts['physics_PDEKF'] = [0]
if wandb_for_bb:
    Q_NN_added_rel_opts['blackbox_PDEKF'] = [0.00001]
else:
    Q_NN_added_rel_opts['blackbox_PDEKF'] = [0.001]
    
global_model_opts = [True]
Q_NN_added_rel_opts['blackbox_PDEKF'] = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
Q_NN_added_rel_opts['hybrid_PDEKF'] = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
n_range = [2,5,10,50,100,500,1000,5000]

#%% load/save settings
load = True
save = False
plot = True
overwrite = True
saveplot = False    # ugly plots, better to self put on widescreen and then save manually
#%% what to plot
beta_opts = [np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]
H_opts = [0.01, 0.03, 0.05, 0.07]
plot_axes_base = 'H'  # 'H', 'beta'
plot_whats = ['eps_pred']
predict_eval = True
assert(predict_eval)

#%% imports
import pathmagic as pathmagic
PlotFolder, ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()
savefolder_start = os.path.join(PlotFolder, f'time_evol_{selection_criteria}')
import setup_src
from plot_util import get_selected_pred, pred_to_errorbars, increase_plot_bounds, set_layout_predictionplot, fix_legend
#%% Config Object
from configuration import MyConfig as MC 
from configuration import update_MC, configurate_MC

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
    
    plot_error = True if plot_what == 'V' else False       # plot var-var_true instead of var
    plot_on_single_axis = True if plot_what == 'V' else False

    #%%
    if plot:
        if share_y_axis:
            fig, axes = plt.subplots(1,4, sharey=True)
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(2,2)
            axes = axes.flatten()

    # for global_model in [False, True]:
    for global_model in global_model_opts:
        savefolder = deepcopy(savefolder_start)
        if not global_model: raise Exception("To be validated")
        if global_model:
            savefolder = savefolder.replace('time_evol', 'time_evol_global') 
        for var_idx, var_i in enumerate(var_opts):
            # print(f"Creating plot for {plot_axes} = {var_i} ")
            for model_i, modeltype in enumerate(modeltypes):
                model, KFtype = modeltype.split('_')
                # print(f"Creating results for modeltype = {modeltype} ")
                Q_NN_added_rel_opts_model = Q_NN_added_rel_opts[modeltype]
                for Q_idx, Q_NN_added_rel in enumerate(Q_NN_added_rel_opts_model):
                    result_list = []
                    for n in n_range:
                        # print(f"Creating results for Q_NN_added_rel = {Q_NN_added_rel} ")
                        #%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
                        MC_i = MC()
                        configurate_MC(MC_i)

                        #%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
                        update_MC(MC_i, which_model=model ,wandb_results=0)
                        ResultSave_i = setup_src.get_ensemble_save_path(ResultFolder, MC_i, predict_eval=predict_eval)

                        # if Q_NN_added_rel != 0 and model == 'hybrid':
                        if Q_NN_added_rel != 0:
                            MC.Q_NN_added_rel = Q_NN_added_rel
                            EnsembleSave_Qsetting = ResultSave_i + f'_Q_NN_added_rel_{Q_NN_added_rel}/'  
                        else:
                            EnsembleSave_Qsetting = ResultSave_i 
                            
                        if plot_what == 'eps_pred':
                            EnsembleSave_Qsetting += f'/n_{n}/'
                            savefolder_i = savefolder + f'/n_{n}/'
                        else:
                            savefolder_i = savefolder

                        modeltype_path_i = os.path.join(EnsembleSave_Qsetting, modeltype)
                        savepath_i = os.path.join(savefolder_i, f'{modeltype}_plot_{plot_what}_{plot_axes}_{var_i}.pkl')
                        if 'hybrid' in model: savepath_i = savepath_i.replace("hybrid_", 'H-')
                        if 'physics' in model: savepath_i = savepath_i.replace("physics_", 'p-')
                        if 'blackbox' in model: savepath_i = savepath_i.replace("blackbox_", 'n-')
                        if Q_NN_added_rel != 0: savepath_i = savepath_i.replace(".pkl", f"_Q_NN_added_rel_{Q_NN_added_rel}.pkl")
                        # if Q_NN_added_rel != 0 and model == 'hybrid': savepath_i = savepath_i.replace(".pkl", f"_Q_NN_added_rel_{Q_NN_added_rel}.pkl")

                        if wandb_for_bb and modeltype == "blackbox_PDEKF":
                            modeltype_path_i = modeltype_path_i.replace("results/", "results_wandb1/")
                            modeltype_path_i = modeltype_path_i.replace("blackbox_PDEKFwandb", "blackbox_PDEKF")
                            
                        if not load:
                            selected_results = get_selected_pred(var=var_i, varname=plot_axes, modeltype_path=modeltype_path_i, MC=MC_i, verbose=False)
                            if len(selected_results) != 192:
                                if len(selected_results) != 177:
                                    if len(selected_results) != 120:
                                        if len(selected_results) != 59:
                                            raise Exception
                            # if len(selected_results) == 0: continue
                        else:
                            selected_results = None
                        
                        if plot:
                            ax = axes[var_idx]
                        else: 
                            ax = None
                        results, colors = pred_to_errorbars(selected_results,  plot_axes=plot_axes, var_i=var_i, i=model_i, ax=ax, model=modeltype, plot=False,
                                                            load=load, save=save, overwrite=overwrite, savepath=savepath_i, print_mean=print_mean, return_results=True)
                                        
                        result_list.append(results)
                        gc.collect()
                        del selected_results
                        gc.collect()
                    if plot:
                        color = colors[0] if 'hybrid' in model else colors[1] if 'physics' in model else colors[2]
                        for i, res in enumerate(result_list):
                            # bplot = ax.boxplot([res],positions=[i], labels=[rf'n = ${n_range[i]}$'], vert=True, sym='',patch_artist=True)
                            bplot = ax.boxplot([res],positions=[i], labels=[rf'${n_range[i]}$'], vert=True, sym='',patch_artist=True)
                            for patch in bplot["boxes"]:
                                patch.set_facecolor(color)       
                            for line in bplot["medians"]:
                                line.set_color('k')      
                        resmeans = [np.median(res) for res in result_list]
                        if Q_idx == 0:
                            label = 'H-PDEKF' if 'hybrid' in model else 'p-PDEKF' if 'physics' in model else 'n-PDEKF'
                            ax.plot(resmeans, color=color, label=label, alpha=0.5)
                        else:
                            ax.plot(resmeans, color=color, alpha=0.5)
                        ax.set_yscale('log')
                        ax.set_xlabel('j', fontsize=35)
                if model_i == 0:
                    set_layout_predictionplot(ax, model, plot_axes, var_i)

    if plot: 
        # plt.set_yscale('log')
        # axes[0].set_yscale('log')
        # axes[1].set_yscale('log')

        fix_legend(axes[0])
        # axes[0].legend
        
        def on_resize(event):
            # Update tick labels when resizing
            # yticks = ax.get_yticks()
            # yticks_new = [f"${round(tick, 2)}$" for tick in yticks]
            # ax.set_yticklabels(yticks_new, fontsize=30)

            plt.tight_layout()
            for ax in axes:
                ax.set_yscale('log')
        increase_plot_bounds(axes)
            
        plt.show()
        plt.close()

