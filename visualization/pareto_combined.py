"""
(Figure 7) - (created without plt.style.use(plotstylepath) in plot_util.py)
copy of read_pickled_results_time.py with following adaptation:
    - Iterate over different QNN to check (p-convergence vs QNN vs filtering error)
Two options of plotting types:
    1. Pareto plot. Plot p-convergence vs filtering error for multiple values (scatterplot) of QNN
    2. Plot p-convergence (y-axis1) and filtering error (y-axis2) both in function of QNN (x-axis)
    
A lot of the same variables as in time_evolution.py
- see comments there for more information on each variable's purpose if not clear.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import gc

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#%%
whichplot = "type2"         # "type2" plot Qw and p-error vs Q_alpha
whichplot = "type1"         # "type1": plot Qw vs p-error
subselect_n_ensembles = 1
selection_criteria = 'Qx'   # 'Qx' or 'innov'

global_model = True     # pretrained model is not trained on specific subset of data but on complete "global" dataset (default)
wandb_for_bb = True     # use wandb tuned parameters for blackbox model (default)

#%% load/save settings
load = True
save = False
assert(load == True), "Only load=True is implemented; create results with time_evolution script"
#%% which models
modeltypes = ['hybrid_PDEKF', 'blackbox_PDEKF', 'physics_PDEKF']
# modeltypes = ['hybrid_PDEKF', 'blackbox_PDEKF']

#%% what to plot
plot_axes = 'H'
plot_what = pareto_p, pareto_metric = [['H', 'beta', 'V'], 'Qx']
# plot_what = pareto_p, pareto_metric = [['H', 'beta', 'V'], 'eps_pred']
Q_weight_added = 1e-9   # Q_NN is always = Q_weight_added * Q_NN_added_rel; Q_weight_added = 1e-9 is default (used everywhere)
#%%
import pathmagic as pathmagic
PlotFolder, ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()
savefolder = os.path.join(PlotFolder, f'time_evol_{selection_criteria}')

import setup_src
from plot_util import get_selected_results, dict_to_pareto, set_plot_bounds, fix_legend_pareto, increase_plot_bounds_pareto, dict_to_QNN_on_x_axis
#%% Config Object
from configuration import MyConfig as MC 
from configuration import update_MC, configurate_MC

MC.global_model = global_model

fig, axes = plt.subplots(1)

if whichplot == "type2": axes2 = axes.twinx() # create duplicate axes for second y-axis

if global_model: savefolder = savefolder.replace('time_evol', 'time_evol_global') 
Q_NN_added_rel_opts = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
## for [0.000001, 0.00001] I don't have results for hybrid model [0.000001, 0.00001] * 1e-9 = [1e-15, 1e-14]
## for [0.000001, 0.00001] I only have results for blackbox model [0.000001, 0.00001] * 1e-7 = [1e-13, 1e-12]
    # however they differ not much from 0 and 0.0001 (*1e-7) so just skip them
        # ensures I can plot with same # elements for all models
        # alternative: run hybrid for [0.000001, 0.00001] * 1e-7
if plot_what[1] == 'eps_pred':
    ## for eps_pred I took blackbox method till 1 (not higher) --> to have same # elements for all models include 1e-6 and 1e-5 again
        # --> 1e-6 and 1e-5 not in hybrid method
        # --> 10, 100 not in blackbox method 
            # --> they have same real Q_alpha (because they have a shift of *100 in 1e-9 vs 1e-7)
    Q_NN_added_rel_opts = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# Q_NN_added_rel_opts = [0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
y_res = None
y_res2 = None
x_res = None

for model_idx, modeltype in enumerate(modeltypes):
    model, KFtype = modeltype.split('_')
    #%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
    configurate_MC(MC)
    update_MC(MC, which_model=model, which_kalman_opt=KFtype, wandb_results=0)
    #%% get pathing to results based on selected MC
    EnsembleSave = setup_src.get_ensemble_save_path(ResultFolder, MC)

    for plot_what_idx, plot_what_ii in enumerate(plot_what[0]):
        if plot_what_ii == 'beta': 
            plot_axes = 'beta'
        else:
            plot_axes = 'H'
            
        if plot_axes == 'beta':
            var_opts = MC.beta
        elif plot_axes == 'H':
            var_opts = MC.H
        plot_what_i = [plot_what_ii, plot_what[1]]
        
        import time
        t1 = time.time()

        for var_idx, var_i in enumerate(var_opts):
            x = []
            y = []
            y2 = []
            y_std = []
            y2_std = []
            print(f"Creating plot for {plot_axes} = {var_i} ")
            for Q_NN_idx, Q_NN_added_rel in enumerate(Q_NN_added_rel_opts):
                if model == 'blackbox' and wandb_for_bb == True:
                    if global_model:
                        # (compensated later) loading from 1e-7 * Q_NN_added_rel instead of previously 1e-9 * Q_NN_added_rel
                        Q_NN_added_rel_i = Q_NN_added_rel
                    else:
                        Q_NN_added_rel_i = Q_NN_added_rel / 100
                        if Q_NN_added_rel_i == 1.0 or Q_NN_added_rel_i == 0.0: 
                            Q_NN_added_rel_i = int(Q_NN_added_rel_i)
                elif model == 'physics':
                    Q_NN_added_rel_i = 0
                else:
                    Q_NN_added_rel_i = Q_NN_added_rel
                if Q_NN_added_rel_i != 0:
                    MC.Q_NN_added_rel = Q_NN_added_rel_i
                    EnsembleSave_Qsetting = EnsembleSave + f'_Q_NN_added_rel_{Q_NN_added_rel_i}/'  
                else:
                    EnsembleSave_Qsetting = EnsembleSave          
                # print(f"Creating results for Q_NN_added_rel = {Q_NN_added_rel_i} ")

                if model == 'blackbox' and wandb_for_bb == True:
                    # loading from 1e-7 * Q_NN_added_rel instead of previously 1e-9 * Q_NN_added_rel
                    Q_weight_added = 1e-7   # Q_NN is always = Q_weight_added * Q_NN_added_rel; Q_weight_added = 1e-9 is default (used everywhere)
                    EnsembleSave_Qsetting = EnsembleSave_Qsetting.replace('/results/', '/results_wandb1/')
                    savefolder_i = savefolder.replace('/plots', '/plots_wandb1') 
                else:
                    Q_weight_added = 1e-9   # Q_NN is always = Q_weight_added * Q_NN_added_rel; Q_weight_added = 1e-9 is default (used everywhere)
                    savefolder_i = savefolder
                try:            
                    if whichplot == "type1":
                        xi, yi = dict_to_pareto({}, model=modeltype, Q_NN_added_rel=Q_NN_added_rel_i, Q_NN_idx=Q_NN_idx, ax=axes, var_i=var_i, plot_axes=plot_axes, plot_what=plot_what_i,
                                                            selection_criteria=selection_criteria, subselect_n_ensembles=subselect_n_ensembles, 
                                                            whichplot=whichplot, load=load, save=save, savefolder=savefolder_i)
                        x.append(xi)
                        y.append(yi)
                    elif whichplot == "type2":
                        xi, yi, yi2, yi_std, yi2_std, colors = dict_to_QNN_on_x_axis(model=modeltype, Q_NN_added_rel=Q_NN_added_rel_i, Q_NN_idx=Q_NN_idx, ax=axes, ax2=axes2, var_i=var_i, plot_axes=plot_axes, plot_what=plot_what_i,
                                                        selection_criteria=selection_criteria, subselect_n_ensembles=subselect_n_ensembles, 
                                                        load=load, save=save, savefolder=savefolder_i, Q_weight_added=Q_weight_added)
                        x.append(xi)
                        y.append(yi)
                        y2.append(yi2)
                        y_std.append(yi_std)
                        y2_std.append(yi2_std)
                except:
                    # print(f"Error for {modeltype} with Q_NN_added_rel = {Q_NN_added_rel_i}")
                    continue
                gc.collect()
            
            if y_res is None:
                y_res = np.zeros((len(modeltypes), len(pareto_p), len(MC.H), len(x)))
                y_res2 = np.zeros((len(modeltypes), len(pareto_p), len(MC.H), len(x)))
                x_res = np.zeros((len(modeltypes), len(pareto_p), len(MC.H), len(x)))
                
            if whichplot == "type2":
                y = np.array(y)
                y_std = np.array(y_std)
                y2 = np.array(y2)
                y2_std = np.array(y2_std)

                y_res[model_idx, plot_what_idx, var_idx] = y
                y_res2[model_idx, plot_what_idx, var_idx] = y2
                x_res[model_idx, plot_what_idx, var_idx] = x
            elif whichplot == "type1":
                y_res[model_idx, plot_what_idx, var_idx] = y
                x_res[model_idx, plot_what_idx, var_idx] = x
if x_res.flatten().shape[0] == 0:
    import sys
    sys.exit("No results found for selected parameters")
    
if whichplot == "type2": 
    assert(np.all([np.all(x_res[i] - x_res[i,0,0] == 0) for i in range(len(x_res))])), "All x-axisses should be equal for a single model (over the different variables)"
    x_res = [x_res[i,0,0] for i in range(len(x_res))]
    result_variables = [y_res, y_res2]
elif whichplot == "type1":
    result_variables = [x_res, y_res]

result_vars = []
geometric_mean = True
for var in result_variables:
    # y_res.shape = (models x variables [H,B,V] x selection (H=0.03, H=0.05) x Q_NN_added_rel)

    if geometric_mean:
        # give single total mean --> mean over third dimension
        var_res_geom = np.prod(var, axis=2)**(1/var.shape[2])
        
        # calculate relative error for each variable and then take mean (over each model and the complete x_axis [QNN added])
        var_res_mean_geom = np.prod(var_res_geom, axis=1)**(1/var_res_geom.shape[1])
        
        var_flattened = var.reshape((var.shape[0], -1, var.shape[3]))
        var_res_mean_geom2 = np.prod(var_flattened, axis=1)**(1/var_flattened.shape[1])
        assert(np.allclose(var_res_mean_geom, var_res_mean_geom2)), "Geometric mean order should not matter"
        
        var_res_rel_mean_over_all_vars = var_res_mean_geom
    else:
        # give single total mean --> mean over third dimension
        var_res = np.mean(var, axis=2)  

        # calculate relative error for each variable and then take mean (over each model and the complete x_axis [QNN added])
        var_res_mean = np.mean(np.mean(var_res, axis=-1), axis=0)

        ## below returns (n_variables [H,B,V] x n_models x n_Q_NN_added_rel) 
        var_res_rel = np.array([(var_res[:,i,:] - var_res_mean[i]) / var_res_mean[i] for i in range(len(var_res_mean))])

        # average out over multiple variables --> returns (n_models x n_Q_NN_added_rel)
        var_res_rel_mean_over_all_vars = np.mean(var_res_rel, axis=0)
        
    result_vars.append(var_res_rel_mean_over_all_vars)

for model_idx, modeltype in enumerate(modeltypes):
    model, KFtype = modeltype.split('_')
    color = colors[0] if 'hybrid' in model else colors[2] if 'blackbox' in model else colors[1]
    if whichplot == "type2":
        y_res_rel_i = result_vars[0][model_idx,:]
        y_res2_rel_i = result_vars[1][model_idx,:]
        x_res_i = x_res[model_idx]
        x_adjusted = [xi if xi != 0 else 1e-14 for xi in x_res_i] # avoid log(0) dissappearing from plot
    elif whichplot == "type1":
        x_res_i = result_vars[0][model_idx,:]
        y_res_rel_i = result_vars[1][model_idx,:]
    
    # marker = 'v' if 'hybrid' in model else 'x' if 'blackbox' in model else 'o'
    marker = 'o' if 'hybrid' in model else 'x' if 'blackbox' in model else 'o'
    if whichplot == "type2":
        axes.plot(x_adjusted, y_res_rel_i, color=colors[0], marker=marker)
        axes2.plot(x_adjusted, y_res2_rel_i, color=colors[1], marker=marker)
    elif whichplot == "type1":
        model_label = 'H-PDEKF' if 'hybrid' in model else 'n-PDEKF' if 'blackbox' in model else 'p-PDEKF'
        axes.plot(x_res_i, y_res_rel_i, marker=marker, label=model_label, color=color)
        for point in range(len(x_res_i) - 1):
            if point >= 4:
                ## draw directional arrow
                begin = (x_res_i[point], y_res_rel_i[point])
                end = (x_res_i[point+1], y_res_rel_i[point+1])
                annotate_begin = ((x_res_i[point] + x_res_i[point+1])/2, (y_res_rel_i[point] + y_res_rel_i[point+1])/2)
                annotate_end = ((x_res_i[point] + x_res_i[point+1])/2, (y_res_rel_i[point] + y_res_rel_i[point+1])/2)
                if plot_what[1] == 'eps_pred':
                    begin = np.log(begin)
                    end = np.log(end)
                    annotate_begin = np.log(annotate_begin)
                    annotate_end = np.log(annotate_end)
                # plt.annotate('', xy=annotate_end, xytext=begin, size=30, arrowprops=dict(arrowstyle="->", color='black', lw=3, alpha=1.0))
                # plt.annotate('', xy=annotate_end,xytext=begin, size=30, arrowprops=dict(arrowstyle="->", color='black'), alpha=1.)
                plt.annotate('', xy=annotate_end,xytext=begin, size=40, arrowprops=dict(arrowstyle="->", color=color), alpha=1.)
                
if plot_what[1] == 'eps_pred':
    plt.yscale('log')
    
xticks = axes.get_xticks()
axes.set_xticks(xticks)
# xticks_new = [f"${round(tick,2)}$" for tick in xticks]
xticks_new = [f"${round(tick,2):.2f}$" for tick in xticks]  # 2 float precision
axes.set_xticklabels(xticks_new)

yticks = axes.get_yticks()
yticks = yticks[::2]    # reduce number of yticks
axes.set_yticks(yticks)
# yticks_new = [f"${round(tick, 5)}$" for tick in yticks]

if whichplot == "type2":
    if plot_what[1] != 'eps_pred':
        yticks_new = [f"${round(tick, 5):.4f}$" for tick in yticks]
        axes.set_yticklabels(yticks_new, color=colors[0], fontsize=20)
    xticks = axes2.get_xticks()
    axes2.set_xticks(xticks)
    xticks_new = [f"${tick}$" for tick in xticks]
    axes2.set_xticklabels(xticks_new, fontsize=20)

    yticks = axes2.get_yticks()
    axes2.set_yticks(yticks)
    yticks_new = [f"${round(tick, 5)}$" for tick in yticks]
    axes2.set_yticklabels(yticks_new, color=colors[1], fontsize=20)
    # ax.set_yscale('log')
    axes2.set_xscale('log')
elif whichplot == "type1":
    axes.set_xticklabels(xticks_new, fontsize=30)
    if plot_what[1] != 'eps_pred':
        yticks_new = [f"${round(tick, 5):.4f}$" for tick in yticks]
        axes.set_yticklabels(yticks_new, fontsize=30)
    if plot_what[0] == ['H', "beta", "V"]:
        if geometric_mean:
            axes.set_xlabel(rf"Geometric Mean $H, \beta, V \ $ Error", fontsize=30)
        else:
            axes.set_xlabel(rf"Mean Relative $H, \beta, V \ $ Error", fontsize=30)
    if plot_what[1] == 'Qx':
        if geometric_mean:
            axes.set_ylabel(r"Geometric Mean $Q_{\omega}$", fontsize=30)
        else:
            axes.set_ylabel(r"Mean Relative $Q_{\omega}$", fontsize=30)
    elif plot_what[1] == 'eps_pred':
        if geometric_mean:
            axes.set_ylabel(r"Geometric Mean $MSE{{\epsilon_{{\omega, f}}}}$", fontsize=30)
        else:
            axes.set_ylabel(r"Mean Relative $MSE{{\epsilon_{{\omega, f}}}}$", fontsize=30)
    plt.title("")
    plt.legend(fontsize=30)    
t2 = time.time()
print(f"Time for creating plot = {t2-t1}")
if whichplot == "type1":
    pass
    # fix_legend_pareto(axes[0], where='lower', Q_NN_added_rel_opts=Q_NN_added_rel_opts)
plt.subplots_adjust(wspace=0.6)
# increase_plot_bounds_pareto(axes)
# set_plot_bounds(axes, result_metric)
def on_resize(event):
    # Update tick labels when resizing
    # for ax in axes:
    #     ax.tick_params(axis='both', which='major', labelsize=10)
    # for ax in axes2:
    #     ax.tick_params(axis='both', which='major', labelsize=10)
    # xlabel = axes[0].
    # ax.set_xlabel(r'$\mathrm{Time \ [s]}$', fontsize=fontsize)
        # ax.set_title(title, fontsize=fontsize)
    pass
    # plt.tight_layout()
    # fix_legend_metrics(axes[0], selection_criteria_opts, plotting_order=plotting_order, fig=fig)

fig.canvas.mpl_connect('resize_event', on_resize)   # when resizing, reformat the yticks
plt.show()
plt.close()

