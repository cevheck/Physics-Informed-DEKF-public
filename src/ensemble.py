"""
All functionalities related to handling the ensemble DEKF
- pathing
- saving
- plotting
- wandb result handling
- simulate ensemble
- subselect ensemble based on selected metric
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

def get_simulation_savepath(EnsembleSave, model, Kalman_method, p_true, V_test=None):
    if not os.path.exists(EnsembleSave): os.mkdir(EnsembleSave)
    modeltype = f"{model}_{Kalman_method}"
    ModelTypeSave = os.path.join(EnsembleSave, modeltype)
    if not os.path.exists(ModelTypeSave): os.mkdir(ModelTypeSave)
    if len(p_true['H']) > 0:    # H, beta, m given as arrays
        H = str(p_true['H'][0]).replace(".","_")
        beta = str(round(p_true['beta'][0],2)).replace(".","_")
        m = str(round(p_true['m'][0],2)).replace(".","_")
    else:                       # H, beta, m given as single values
        H = str(p_true['H']).replace(".","_")
        beta = str(round(p_true['beta'],2)).replace(".","_")
        raise Exception("Not implemented yet for m,V")
    savename = f"H{H}_beta{beta}_m{m}"
    SimulationSave = os.path.join(ModelTypeSave, savename)

    if isinstance(V_test, np.ndarray):
        V1 = str(round(V_test[0],2)).replace(".","_")
        V2 = str(round(V_test[1],2)).replace(".","_")
        EnsembleSave += f"_V{V1}_V{V2}"
    elif V_test:
        V = str(round(V_test,2)).replace(".","_")
        SimulationSave += f"_V{V}"
    return SimulationSave, ModelTypeSave

def make_new(model, Kalman_method, p_true, EnsembleSave, overwrite, plot_now, V_test, wandb_tuning=False):
    """
    Determine to make a new simulation for this setting or not
    - does it already exist
    - do i want to overwrite
    """
    if plot_now or overwrite or wandb_tuning: return True   # if I want to plot or overwrite, I need create now
    SimulationSave,ModelTypeSave = get_simulation_savepath(EnsembleSave, model, Kalman_method, p_true, V_test)
    if os.path.exists(SimulationSave) or os.path.exists(SimulationSave + '.pkl'): return False       # already created and overwrite = False
    return True

def save_ensemble(EnsembleSave, return_dict, MC, plotting_settings, V_test=None, overwrite=True, save_zipped=False, save_w_full=False):
    # save ensemble to file
    if not os.path.exists(EnsembleSave): os.mkdir(EnsembleSave)
    SimulationSave, ModelTypeSave = get_simulation_savepath(EnsembleSave, MC.model, MC.Kalman_method, MC.p_true, V_test)
    if os.path.exists(SimulationSave) and not overwrite: raise Exception(f"Simulation result file {SimulationSave} already exists and overwrite = False!")
    if not os.path.exists(ModelTypeSave): os.mkdir(ModelTypeSave)
    if not os.path.exists(SimulationSave): os.mkdir(SimulationSave)

    additional_plotting_info = {}
    additional_plotting_info['real_y'] = MC.real_y
    additional_plotting_info['trainable_names'] = MC.trainable_names
    additional_plotting_info['p_true'] = MC.p_true
    
    # # save complete w as zip ; remove from return dict
    n_p = len(MC.trainable_names)
    # w_zip = return_dict['filtering']['w']
    # np.savez_compressed(savepath, w=w_array)
    # print(f"Weights Saved to {savepath}")
    
    ## remove abundant NN part of w from return dict
    return_dict['final_w'] = return_dict['filtering']['w'][-1]      # save final weight state (for prediction models to be loaded)
    if not save_w_full:
        return_dict['filtering']['w'] = return_dict['filtering']['w'][:,:, -n_p:,:]
    
    # save simulation
    pickle.dump(return_dict, open(os.path.join(SimulationSave, "return_dict.pkl"), "wb"))
    pickle.dump(additional_plotting_info, open(os.path.join(SimulationSave, "additional_plotting_info.pkl"), "wb"))
    pickle.dump(plotting_settings, open(os.path.join(SimulationSave, "plotting_settings.pkl"), "wb"))

def save_prediction(ResultSave, MSE, MC, V_test=None, overwrite=True, save_zipped=False):
    # save ensemble to file
    if not os.path.exists(ResultSave): os.mkdir(ResultSave)
    ResultSave, ModelTypeSave = get_simulation_savepath(ResultSave, MC.model, MC.Kalman_method, MC.p_true, V_test)
    ResultSave += '.pkl'
    if os.path.exists(ResultSave) and not overwrite: raise Exception(f"Ensemble file {ResultSave} already exists and overwrite = False!")
    if not os.path.exists(ModelTypeSave): os.mkdir(ModelTypeSave)

    # save simulation
    with open (ResultSave, "wb") as f:
        pickle.dump(MSE, f)

def process_ensemble(plot_ensemble_results, plot_results, NN_investigation, path, MC):
    ensemble_results = os.listdir(path)
    for result in ensemble_results:
        if not result.endswith(".pkl"): raise Exception("Not a pickle file")
        ensemble_file_path = os.path.join(path, result)
        if 'settings' in result:
            settings = pickle.load(open(ensemble_file_path, "rb"))
        elif 'additional_plotting_info' in result:
            additional_plotting_info = pickle.load(open(ensemble_file_path, "rb"))
        elif 'return_dict' in result:
            return_dict = pickle.load(open(ensemble_file_path, "rb"))
        else:
            raise Exception(f"Unknown pickle file {result}")
    MC.real_y = additional_plotting_info['real_y']
    results = plot_ensemble_results(return_dict, MC, settings, plot_results, NN_investigation)
    return results

def create_empty_MC():
    class MC:
        pass
    MC = MC()
    return MC

def plot_ensemble_results(return_dict, MC, plotting_settings, plot_results, NN_investigation):
    n_ensemble = return_dict['filtering']['x'].shape[1]
    timesamples = return_dict['filtering']['x'].shape[0]
    desired_nr_samples = 50
    ds_rate = int(timesamples/desired_nr_samples)
    model = MC.model
    
    ## mpl settings
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.close("all")
    
    cluster = False
    if cluster:
        beta = return_dict['filtering']['w'][:,:,-1,0]
        H = return_dict['filtering']['w'][:,:,-2,0]

        beta_end = return_dict['filtering']['w'][-1,:,-1]
        H_end = return_dict['filtering']['w'][-1,:,-2]
        plt.plot(beta) 
        plt.plot(H)
        from sklearn.cluster import KMeans
        best_MCD = 500
        for n_clusters in range(1,11):
            kmeans_beta = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=0)
            kmeans_H = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10, random_state=0)
            
            kmeans_beta.fit(beta_end)
            kmeans_H.fit(H_end)
            
            kmeans_beta.labels_
            kmeans_H.labels_
            print(f"equal clusters = {np.all(kmeans_beta.labels_ == kmeans_H.labels_)}")
            MCD = np.mean(kmeans.fit_transform(beta))   # mean cluster distance
            selected_cluster_means = kmeans.cluster_centers_[kmeans.labels_]
            MCD = np.mean(np.abs(beta-selected_cluster_means))
            print(f"n_clusters: {n_clusters}, MCD: {MCD}")
            if MCD < best_MCD:
                best_MCD = MCD
                best_n_clusters = n_clusters
        
        for b_idx in range(beta.shape[1]):
            print(kmeans_beta.labels_[b_idx])
            #plt.scatter(np.arange(len(beta)), beta[:,b_idx], c=np.repeat(kmeans_beta.labels_[b_idx], len(beta)), cmap='viridis')
            plt.scatter(np.arange(len(beta)), beta[:,b_idx], c=colors[kmeans_beta.labels_[b_idx]], cmap='viridis')

        for H_idx in range(H.shape[1]):
            print(kmeans_H.labels_[H_idx])
            #plt.scatter(np.arange(len(beta)), beta[:,b_idx], c=np.repeat(kmeans_beta.labels_[b_idx], len(beta)), cmap='viridis')
            plt.scatter(np.arange(len(H)), H[:,H_idx], c=colors[kmeans_H.labels_[H_idx]], cmap='viridis')


        plt.plot(np.arange(len(beta)), beta, c=kmeans_beta.labels_, cmap='viridis')

    if NN_investigation and plot_results:
        fig, axes = plt.subplots(2,2)
        ax11, ax22, ax33, ax44 = axes.flatten()
    elif NN_investigation:
        fig, (ax22, ax33, ax44) = plt.subplots(1,3)


    if NN_investigation:
        if NN_investigation: plt.suptitle('Evolution of errors i.f.o. time')
        if model != 'hybrid': raise Exception("NN_investigation only relevant for hybrid model")
        NN_param_size_list = []
        NN_param_list = []
        for DEKF_idx in range(n_ensemble): 
            total_sinapses, n_p = return_dict['dekf_param']
            NN_param = np.array(return_dict['filtering']['w'])[:,DEKF_idx,:-n_p,0]
            NN_param_size = np.abs(NN_param)
            NN_mean_param_size = np.mean(NN_param_size, axis=1)     # average over all weights (not over time dimension)
            NN_relative_size = NN_mean_param_size - NN_mean_param_size[0]
            NN_procentual_size = NN_mean_param_size / NN_mean_param_size[0]

            NN_param_size_list.append(NN_mean_param_size)    
                
            #NN_param_size_list.append(NN_relative_size)        
            #NN_param_size_list.append(NN_procentual_size)        
        # Create a color map based on NN_param_size        
        norm = plt.Normalize(np.min(NN_param_size_list), np.max(NN_param_size_list))
        cmap = plt.cm.get_cmap('viridis')

    # settings
    if plot_results:
        plot_weights, plot_P, plot_K, plot_deriv, verbose = plotting_settings
        Kalman_method = MC.Kalman_method
        
        # data (same for all simulations)
        history_x, history_u, filter_x, filter_u, future_x, future_u = MC.history_x, MC.history_u, MC.filter_x, MC.filter_u, MC.future_x, MC.future_u
        try:
            filter_xreal = MC.filter_xreal
        except:
            pass

        real_y = MC.real_y
        T = MC.T
        T_history = T[:len(history_x)]
        T_KF = T[len(history_x):len(history_x)+len(filter_x)]
        T_future = T[-len(future_x):] if len(future_x) > 0 else []
        # p_H = MC.p_true['H'][np.arange(len(history_x), len(history_x)+len(filter_x))]
        # p_beta = MC.p_true['beta'][np.arange(len(history_x), len(history_x)+len(filter_x))]
        
        ### ----- PLOTS ------ ###
        ## plotting kalman filtering trajectory
        state_names = ['theta', 'omega']
        fig1, axes = plt.subplots(1,len(state_names))
        for state_idx in range(len(state_names)):
            ## plot real values / measurements
            if not history_x is None:
                if not real_y is None:
                    #if len(real_y) != len(T_KF):
                    ## complete real_y is given, including history and future
                    real_y_history = real_y[:len(history_x)]
                    real_y_KF = real_y[len(history_x):len(history_x)+len(filter_x)]
                    real_y_future = real_y[len(history_x)+len(filter_x):len(history_x)+len(filter_x)+len(future_x)]

                    l1, = axes[state_idx].plot(T_history, real_y_history[:, state_idx], linestyle=':', color='olive', label='real_y')
                    axes[state_idx].plot(T_KF, real_y_KF[:,state_idx], linestyle=':', color='olive')
                    axes[state_idx].plot(T_future, real_y_future[:, state_idx], linestyle=':', color='olive')
                    #else:
                        #real_y_KF = real_y[len(history_x):len(history_x)+len(filter_x)]
                        #raise Exception("Validate if real_y_KF correctly calculated here with correct Time aswell")
                l2 = axes[state_idx].scatter(T_history, history_x[:, state_idx], marker='x', c='k', s=5, label='measurements')
                axes[state_idx].scatter(T_KF, filter_x[:, state_idx], marker='x', c='k', s=5)

                ## plot predictions (different for each DEKF)
                state_history_ensemble = []
                Pxk_history_ensemble = []
                state_ensemble = []
                state_future_ensemble = []
                Pxk_ensemble = []
                eps_innov_ensemble = []
                Qx_ensemble = []
                for DEKF_idx in range(n_ensemble):  
                    # history (=initial state)  
                    initial_state = return_dict['initial_condition']['xk_ini'][DEKF_idx,state_idx,:]
                    initial_state_expanded = np.ones_like(history_x[:,state_idx])*initial_state
                    try:
                        initial_Px = return_dict['initial_condition']['Pxk_ini'][DEKF_idx, state_idx, state_idx]
                    except:
                        initial_Px = return_dict['initial_condition']['Pxk_ini'][DEKF_idx, state_idx,0]
                    initial_Px_expanded = np.ones_like(history_x[:,state_idx])*initial_Px
                    l3, = axes[state_idx].plot(T_history, initial_state_expanded, alpha=0.3, c=colors[3])
                    #l3, = axes[state_idx].scatter(T_history, initial_state_expanded, marker='o', s=0.8, c=colors[3])

                    # during KF
                    X = np.array(return_dict['filtering']['x'][:,DEKF_idx,:])
                    state = X[:, state_idx, 0]
                    l4, = axes[state_idx].plot(T_KF, state, alpha=0.3, c=colors[1])

                    Pxk = np.array(return_dict['filtering']['Px'][:,DEKF_idx])
                    uncertainty = Pxk[:, state_idx, state_idx]  # diagonal elements

                    # after KF: future
                    try:
                        X_future = np.array(return_dict['future']['x'][:,DEKF_idx])
                        state_next = X_future[:, state_idx, 0]
                        l5, = axes[state_idx].plot(T_future, state_next, alpha=0.3, c=colors[2], label=f'prediction: {state_names[state_idx]}')
                        state_future_ensemble.append(state_next)
                    except:
                        X_future = []                        
                    # innovation error
                    try:
                        eps_innov = np.array(return_dict['filtering']['x_error'][:,DEKF_idx])
                        eps_innov_ensemble.append(eps_innov)
                    except:
                        pass
                    
                    try:
                        Qx = np.array(return_dict['filtering']['Qx'][:,DEKF_idx])
                        Qx_ensemble.append(Qx)
                    except:
                        pass
                    # add to ensemble predictions
                    state_history_ensemble.append(initial_state_expanded)
                    Pxk_history_ensemble.append(initial_Px_expanded)
                    state_ensemble.append(state)
                    Pxk_ensemble.append(uncertainty)

                mean_state_history = np.mean(state_history_ensemble, axis=0)
                mean_uncertainty_history = np.mean(Pxk_history_ensemble, axis=0)
                mean_state = np.mean(state_ensemble, axis=0)
                mean_uncertainty = np.mean(Pxk_ensemble, axis=0)
                if state_future_ensemble != []: mean_state_future = np.mean(state_future_ensemble, axis=0)
                l6, = axes[state_idx].plot(T_history, mean_state_history, linestyle='-', label=f'history', c=colors[3])
                l7, = axes[state_idx].plot(T_KF, mean_state, linestyle='-', label=f'filtering', c=colors[1])
                if state_future_ensemble != []: l8, = axes[state_idx].plot(T_future, mean_state_future, linestyle='-', label=f'future', c=colors[2])
                
                # Add uncertainty boundaries, however don't let it change the y-dimension of the plot
                # 95% confidence = mean +- 2*sigma with sigma = std = sqrt(variance)
                ylim = axes[state_idx].get_ylim()
                axes[state_idx].fill_between(T_history, mean_state_history-2*np.sqrt(mean_uncertainty_history), mean_state_history+2*np.sqrt(mean_uncertainty_history), alpha=0.1, color=colors[3])
                axes[state_idx].fill_between(T_KF, mean_state-2*np.sqrt(mean_uncertainty), mean_state+2*np.sqrt(mean_uncertainty), alpha=0.1, color=colors[1])
                axes[state_idx].set_ylim(ylim)
                handles = [l1, l2, l6, l7] if state_future_ensemble == [] else [l1, l2, l6, l7, l8]
                axes[state_idx].legend(handles=handles)
                axes[state_idx].set_title(f'{state_names[state_idx]}')
                
                # log tracking error (mean)
                if state_idx == 1:   # focus on omega tracking
                    plt.suptitle(f"Filtering: {model}_{Kalman_method}")     # subtitle from filtering figure
                    true = real_y_KF[:, state_idx]
                    noisy = filter_x[:, state_idx]
                    
                    eps_filter_data = [state_ensemble_i - true for state_ensemble_i in state_ensemble]
                    if len(eps_innov_ensemble) > 0:
                        eps_title = "Innovation error"
                        try:
                            Qx_data = [np.abs(Qx_ensemble_i[:,state_idx,state_idx]) for Qx_ensemble_i in Qx_ensemble]
                            eps_innov_data = [np.abs(eps_innov_ensemble_i[:,state_idx,0]) for eps_innov_ensemble_i in eps_innov_ensemble]
                        except:
                            eps_innov_data = [np.abs(eps_innov_ensemble_i[:,0,0]) for eps_innov_ensemble_i in eps_innov_ensemble]
                        # eps_plot = eps_innov_data
                        eps_plot = Qx_data
                    else:
                        eps_title = "Filtering error"
                        eps_plot = eps_filter_data
                    
                    if NN_investigation:
                        for NN_param_size, predicted_state, eps_plot_i in zip(NN_param_size_list, state_ensemble, eps_plot):
                            ## downsampling (computation time)
                            NN_param_size_ds = NN_param_size[::ds_rate]
                            predicted_state_ds = predicted_state[::ds_rate]
                            real_state_ds = true[::ds_rate]
                            T_KF_ds = T_KF[::ds_rate]
                            eps_plot_ds = eps_plot_i[::ds_rate]
                            for (time_i, param_size, pred, real_state, eps_plot_i_i) in zip(T_KF_ds, NN_param_size_ds, predicted_state_ds, real_state_ds, eps_plot_ds):
                                color = cmap(norm(param_size))
                                #ax22.scatter(time_i, np.abs(pred-real_state), color=color)
                                #ax22.scatter(time_i, np.abs(eps_innov_i[state_idx]), color=color)
                                if eps_plot_i_i != eps_plot_ds[0]:    # first one really scews the scale
                                    ax22.scatter(time_i, (eps_plot_i_i), color=color)
                        ax22.set_xlabel(f"Sample")
                        ax22.set_ylabel(eps_title)
                    else:
                        fig22, ax22 = plt.subplots()
                        for predicted_state, eps_plot_i in zip(state_ensemble, eps_plot):
                            ## downsampling (computation time)
                            predicted_state_ds = predicted_state[::ds_rate]
                            real_state_ds = true[::ds_rate]
                            T_KF_ds = T_KF[::ds_rate]
                            eps_plot_ds = eps_plot_i[::ds_rate]
                            for (time_i, pred, real_state, eps_plot_i_i) in zip(T_KF_ds, predicted_state_ds, real_state_ds, eps_plot_ds):
                                if eps_plot_i_i != eps_plot_ds[0]:    # first one really scews the scale
                                    ax22.scatter(time_i, (eps_plot_i_i), color='b')
                        ax22.set_xlabel(f"Sample")
                        ax22.set_ylabel(eps_title)
                    predicted = mean_state
                    eps_noisy = predicted - noisy
                    eps_filter = predicted - true
                    groundtruth_error = noisy-true                

                    selection_criteria = np.array(Qx_data)
                    # selection_criteria = np.array(eps_innov_data)
                    if len(eps_innov_ensemble) > 0: 
                        print(f"Mean Innovation error: {np.mean(eps_innov_data)}")
                        mean_innovs_each = [np.mean(eps_innov_i) for eps_innov_i in eps_innov_data]
                        mean_innovs_each = np.where(np.isnan(mean_innovs_each), 1e8, mean_innovs_each)
                        
                        ones_correct_shape = np.ones_like(np.array(eps_innov_data))
                        idxs = np.array([ones_correct_shape[i]*i for i in range(len(ones_correct_shape))])
                        best_idxs, _ = get_best_x_ensembles(idxs, selection_criteria, 1)
                        best_innov, _ = get_best_x_ensembles(np.array(eps_innov_data), selection_criteria, 1)

                        best_filter_idx = np.argmin(mean_innovs_each)
                        print(f"Best Innovation error: {np.min(mean_innovs_each)} for filter {best_filter_idx}")
                        print(f"Best Innovation error at each time: {np.mean(best_innov)}")
                        
                        eps_best = [np.mean(np.abs(state_ensemble_i - true)) for state_ensemble_i in state_ensemble]
                        best_filter_idx2 = np.nanargmin(eps_best)
                        eps_best = np.nanmin(eps_best)
                        print(f"Best Filtering error: {eps_best} for filter {best_filter_idx2}")                               
                    print(f"Mean Filtering error: {np.mean(np.abs(eps_filter_data))}")
                    omega_future = real_y_future[:, state_idx]
                    if state_future_ensemble != []: eps_future = mean_state_future - omega_future
                    #print(f"Future Prediction Error: {np.mean(np.abs(eps_future))}")
            
        if return_dict['initial_condition']['xk_ini'].shape[1] > 2:    # more than 2 states
            # also filtering Tm
            Tm_ensemble = []
            Pxk_ensemble = []
            fig, ax_Tm = plt.subplots()
            for DEKF_idx in range(n_ensemble):
                X = np.array(return_dict['filtering']['x'][:,DEKF_idx,:])
                Tm = X[:, 2, 0]
                Tm_ensemble.append(Tm)
                ax_Tm.plot(T_KF, Tm, alpha=0.3, c=colors[0])
                Pxk = np.array(return_dict['filtering']['Px'][:,DEKF_idx])
                uncertainty = Pxk[:, 2, 2]  # diagonal elements
                Pxk_ensemble.append(uncertainty)
            mean_uncertainty_history = np.mean(Pxk_ensemble, axis=0)
            mean_Tm = np.mean(Tm_ensemble, axis=0)
            ax_Tm.plot(T_KF, mean_Tm, linestyle='-', label=f'Tm', c=colors[1])
            ax_Tm.fill_between(T_KF, mean_Tm-2*np.sqrt(mean_uncertainty_history), mean_Tm+2*np.sqrt(mean_uncertainty_history), alpha=0.1, color=colors[1])
            ax_Tm.legend()
            ax_Tm.set_title("Filtered Tm value")
                
        if plot_P or plot_K: raise NotImplementedError("Plotting of P and K not implemented yet")
    try:
        omega_innov_all  = [np.abs(eps_innov_ensemble_i[1:,1,0]) for eps_innov_ensemble_i in eps_innov_ensemble][0]
        omega_innov_all_ds  = [np.abs(eps_innov_ensemble_i[1::ds_rate,1,0]) for eps_innov_ensemble_i in eps_innov_ensemble]
    except:
        omega_innov_all  = [np.abs(eps_innov_ensemble_i[1:,0,0]) for eps_innov_ensemble_i in eps_innov_ensemble][0]
        omega_innov_all_ds  = [np.abs(eps_innov_ensemble_i[1::ds_rate,0,0]) for eps_innov_ensemble_i in eps_innov_ensemble]
    outliers = np.where(omega_innov_all > np.max(np.sort(omega_innov_all)[:int(0.9*len(omega_innov_all))]))
    omega_innov_all_no_outliers = np.delete(omega_innov_all, outliers)
    omega_innov_windows = [np.mean(omega_innov_all_no_outliers[i:i+ds_rate]) for i in range(0, len(omega_innov_all_no_outliers), ds_rate)]
    norm_ens = plt.Normalize(np.min(omega_innov_windows), np.max(omega_innov_windows))
    norm_ens = plt.Normalize(0.009, 0.03)       ## hardcode
    cmap_ens = plt.cm.get_cmap('viridis')
            
    ## physical parameter tracking
    if True:    #model != 'blackbox':
        if plot_results: 
            n_p = len(MC.trainable_names)
            if n_p == 2:
                fig3, axes = plt.subplots(1,2)
                figbest, axesbest = plt.subplots(1,2)
                fig_ensemble_select, axes_ens_sel = plt.subplots(1,2)
            elif n_p == 4:
                fig3, axes = plt.subplots(2,2)
                figbest, axesbest = plt.subplots(2,2)
                fig_ensemble_select, axes_ens_sel = plt.subplots(2,2)
            elif n_p == 3:
                fig3, axes = plt.subplots(1,3)
                figbest, axesbest = plt.subplots(1,3)
                fig_ensemble_select, axes_ens_sel = plt.subplots(1,3)
            elif n_p == 5:
                fig3, axes = plt.subplots(1,5)
                figbest, axesbest = plt.subplots(1,5)
                fig_ensemble_select, axes_ens_sel = plt.subplots(1,5)
            else:
                raise Exception("Not implemented yet for different than 2, 3, 4 or 5 trainable parameters")
            axes = axes.flatten()
            axes_ens_sel = axes_ens_sel.flatten()
            axesbest = axesbest.flatten()
        
        p_dict = {key: [] for key in MC.trainable_names}
        p_best_dict = {key: [] for key in MC.trainable_names}
        handles_dict = {}

        p_all_param_all = np.array(return_dict['filtering']['w'])[:,:, -n_p:,0]
        for i, key in enumerate(MC.trainable_names):
            p_all = p_all_param_all[:,:,i].T
            p_best, _ = get_best_x_ensembles(p_all, selection_criteria, 1)
            p_best_dict[key] = p_best[0]
            lbest, = axesbest[i].plot(T_KF, p_best_dict[key], label= f'{key}_{DEKF_idx}')
        for DEKF_idx in range(n_ensemble): 
            p_param = np.array(return_dict['filtering']['w'])[:,DEKF_idx, -n_p:,0]
            
            try:
                w_innov = np.abs(np.array(return_dict['filtering']['x_error'])[:,DEKF_idx,1,0])
            except:
                w_innov = np.abs(np.array(return_dict['filtering']['x_error'])[:,DEKF_idx,0,0])
                
            
            if plot_results:
                for i, key in enumerate(MC.trainable_names):
                    p_i = p_param[:,i]
                    # li, = axes[i].plot(T_KF, p_i, alpha=0.3, c=colors[0], label= f'{key}_i')
                    li, = axes[i].plot(T_KF, p_i, alpha=0.3, label= f'{key}_i')
                    handles_dict[key] = li
                    for j in range(0, len(p_i), ds_rate):
                        w_innov_window = np.mean(w_innov[j:j+ds_rate])
                        param_window = np.mean(p_i[j:j+ds_rate])

                        # Create a color map based on NN_param_size        
                        color = cmap_ens(norm_ens(w_innov_window))
                        axes_ens_sel[i].scatter(T_KF[j], param_window, color=color, s=2)
                    p_dict[key].append(p_i)
                
                    
        p_dict = {key: np.array(val) for key,val in p_dict.items()}
        p_errors = {}
        for key in MC.trainable_names:
            if key in MC.p_true:
                p_true = MC.p_true[key]
                p_error = np.array([np.abs(p_dict[key][i,:] - p_true) for i in range(n_ensemble)])
                best_p_error = np.abs(p_true - p_dict[key][best_filter_idx])
                best_p_error2 = np.abs(p_true - p_dict[key][best_filter_idx2])
                print(f"Mean {key} error: {np.mean(p_error)}")
                print(f"best filter mean {key} error: {np.mean(best_p_error)}")
                print(f"best filter2 mean {key} error: {np.mean(best_p_error2)}")
                p_errors[key] = p_error
        ylim_pdict = {}
        ylim_pdict['H'] = [0.005, 0.075]
        ylim_pdict['beta'] = [np.pi/4, 7*np.pi/4]
        # ylim_pdict['Ra'] = [0.5, 1.1]
        # ylim_pdict['D'] = [0, 0.2]
        if plot_results:
            for i, (key, val) in enumerate(p_dict.items()):
                p_mean = np.mean(val, axis=0)
                lmean, = axes[i].plot(T_KF, p_mean, c=colors[1], label=f'{key}_mean')
                if key in MC.p_true: 
                    ltrue, = axes[i].plot(T_KF, MC.p_true[key], c='k', linestyle='--', label=f'{key}_true')
                    axes_ens_sel[i].plot(T_KF, MC.p_true[key], c='k', linestyle='--', label=f'{key}_true')
                    axesbest[i].plot(T_KF, MC.p_true[key], c='k', linestyle='--', label=f'{key}_true')
                    axes[i].legend(handles=[handles_dict[key],lmean,ltrue])
                    p_mean_error = np.abs(MC.p_true[key] - p_mean)
                    # print(f"{key} mean Prediction Error: {np.mean(p_mean_error)}")
                else:
                    axes[i].legend(handles=[handles_dict[key],lmean])
                if key in ylim_pdict: axes[i].set_ylim(ylim_pdict[key])
                if key in ylim_pdict: axes_ens_sel[i].set_ylim(ylim_pdict[key])
                if key in ylim_pdict: axesbest[i].set_ylim(ylim_pdict[key])
                axes[i].set_ylabel(f'{key}')
                axes_ens_sel[i].set_ylabel(f'{key}')

            if plot_weights:
                fig, ax_weights = plt.subplots(1,1)
                for i in range(n_ensemble): 
                    weights = return_dict['filtering']['w'][:,i, :-n_p]
                    for j in range(weights.shape[1]):
                        if j == weights.shape[1] - 1 or j == weights.shape[1] - 2: continue
                        weight_i = weights[:,j, -1]
                        ax_weights.plot(T_KF, weight_i, color=colors[i], alpha=0.3, linestyle=':')
                ax_weights.set_title("NN weights")  
                
            if MC.plot_extras:
                z = return_dict['filtering']['z'][:,:,0,0]
                fig, ax_NNoutp = plt.subplots(1,1)
                for i in range(n_ensemble): 
                    z_i = z[:,i]
                    ax_NNoutp.plot(z_i, color=colors[i])
                    ax_NNoutp.plot(return_dict['filtering']['x'][:,0,1,0], color=colors[i])
                ax_NNoutp.set_title("NN output and omega")
                
                xk_pred = return_dict['filtering']['xk_pred'][:,:,1,0]
                xk_real = MC.real_y[:,1]
                fig, ax_xk_pred = plt.subplots(1,1)
                for i in range(n_ensemble): 
                    xk_pred_i = xk_pred[:,i]
                    ax_xk_pred.plot(xk_pred_i, color=colors[i])
                    ax_xk_pred.plot(return_dict['filtering']['x'][:,i,1,0], color=colors[i], linestyle=':')
                ax_xk_pred.plot(xk_real, color="black", linestyle='--')
                ax_xk_pred.set_title("xk_pred")
            
                fric_rest = return_dict['filtering']['r'][:,:,0]
                fig, ax_r = plt.subplots(1,1)
                for i in range(n_ensemble): 
                    ax_r_i = fric_rest[:,i]
                    ax_r.plot(ax_r_i, color=colors[i])
                ax_r.set_title("r")
                
                Tl_eq = return_dict['filtering']['Tl_eq'][:,:,0]
                fig, ax_Tl_eq = plt.subplots(1,1)
                for i in range(n_ensemble): 
                    Tl_eq_i = Tl_eq[:,i]
                    ax_Tl_eq.plot(Tl_eq_i, color=colors[i])
                ax_Tl_eq.set_title("Tl_eq")
                
                J_eq = return_dict['filtering']['J_eq'][:,:,0]
                fig, ax_J_eq = plt.subplots(1,1)
                for i in range(n_ensemble): 
                    J_eq_i = J_eq[:,i]
                    ax_J_eq.plot(J_eq_i, color=colors[i])
                ax_J_eq.set_title("J_eq")
                
                try:
                    Ts_and_NN = return_dict['filtering']['Ts_and_NN'][:,:,0]
                    fig, ax_Ts_and_NN = plt.subplots(1,1)
                    for i in range(n_ensemble): 
                        Ts_and_NN_i = Ts_and_NN[:,i]
                        ax_Ts_and_NN.plot(Ts_and_NN_i, color=colors[i])
                    ax_Ts_and_NN.set_title("Ts_and_NN")
                except:
                    pass
                        
            if not verbose:
                plt.close("all") 

    if NN_investigation:
        ax44_twin = ax44.twinx()
        # Plot the evolutions of H_error with color map
        for DEKF_idx, (NN_param_size, eps_plot_i) in enumerate(zip(NN_param_size_list, eps_plot)):
            p_error_i = {key: val[DEKF_idx] for key,val in p_errors.items()}
            ## downsampling (computation timeà
            NN_param_size = NN_param_size[::ds_rate]
            p_error_i = {key: val[::ds_rate] for key, val in p_error_i.items()}
            T_KF_ds = T_KF[::ds_rate]
            eps_plot_i_ds = eps_plot_i[::ds_rate]
            for time_idx, (time_i, param_size, eps_plot_i_i) in enumerate(zip(T_KF_ds, NN_param_size, eps_plot_i_ds)):
                color = cmap(norm(param_size))
                for key,val in p_error_i.items():
                    if key == 'H':
                        ax = ax11
                        axx = ax44
                    elif key == 'beta':
                        ax = ax33
                        axx = ax44_twin
                    else:
                        continue
                        raise Exception("Not implemented yet for different than H,Beta trainable parameters (quite close though but now too lazy)")
                    ax.scatter(time_i, p_error_i[key][time_idx], color=color)
                    if eps_plot_i_i != eps_plot_i_ds[0]:    # first one really scews the scale
                        if key == 'beta': 
                            axx.scatter(eps_plot_i_i, p_error_i[key][time_idx], marker='x', color=color)
                        elif key == 'H':
                            axx.scatter(eps_plot_i_i, p_error_i[key][time_idx], color=color)
                        else:
                            continue
                            raise Exception("Not implemented yet for different than H,Beta trainable parameters (quite close though but now too lazy)")
        # Set labels and title
        ax11.set_xlabel('Time [s]')
        ax22.set_xlabel('Sample')
        ax33.set_xlabel('Sample')
        ax44.set_xlabel(eps_title)

        ax11.set_ylabel('|H error|')
        ax33.set_ylabel('|Beta error|')
        ax44.legend(["H error"])
        ax44_twin.legend(["Beta error"])
        # ------- show NN_prediction -------
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        
        [axes_ens_seli.set_xlabel('Time [s]') for axes_ens_seli in axes_ens_sel]
        [axes_i.set_xlabel('Time [s]') for axes_i in axes]
        # [axes_ens_seli.legend() for axes_ens_seli in axes_ens_sel]
        axes_ens_lines = [len(axes_ens_seli.get_lines()) for axes_ens_seli in axes_ens_sel]
        [axes_ens_seli.legend() for (axes_ens_seli, axes_ens_line) in zip(axes_ens_sel, axes_ens_lines) if axes_ens_line > 0]        # checks for empty legends first

        # Add a color bar for the color map
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=[ax11,ax22,ax33,ax44])        
        cbar.set_label('NN_param_size')
        
        
        # Add a color bar for the color map
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm_ens, cmap=cmap_ens), ax=axes_ens_sel)        
        cbar.set_label('innovation error')
    
def WandB_ensemble(return_dict, MC):    
    import wandb
    n_ensemble = return_dict['filtering']['x'].shape[1] if 'filtering' in return_dict else None

    results = {}
    p_H, p_beta = MC.p_true['H'], MC.p_true['beta']

    try:
        p_H = p_H[0]
        p_beta = p_beta[0]
    except:
        pass
    
    history_x, history_u, filter_x, filter_u, future_x, future_u = MC.history_x, MC.history_u, MC.filter_x, MC.filter_u, MC.future_x, MC.future_u

    T_history = MC.T[:len(history_x)]
    T_KF = MC.T[len(history_x):len(history_x)+len(filter_x)]
    T_future = MC.T[-len(future_x):]

    ### ----- PLOTS ------ ###
    ## plotting kalman filtering trajectory
    state_names = ['theta', 'omega']
    for state_idx in range(len(state_names)):
        if not history_x is None:
            if not MC.real_y is None:
                #if len(real_y) != len(T_KF):
                # complete real_y is given, including history and future
                real_y_history = MC.real_y[:len(history_x)]
                real_y_KF = MC.real_y[len(history_x):len(history_x)+len(filter_x)]
                real_y_future = MC.real_y[len(history_x)+len(filter_x):len(history_x)+len(filter_x)+len(future_x)]
                #else:   
                #    real_y_KF = real_y[len(history_x):len(history_x)+len(filter_x)]
            ## plot predictions (different for each DEKF)
            state_history_ensemble = []
            Pxk_history_ensemble = []
            state_ensemble = []
            state_future_ensemble = []
            Pxk_ensemble = []
            eps_innov_ensemble = []
            Qx_ensemble = []
            try:
                for DEKF_idx, simulation in sorted(return_dict.items()):  
                    # history (=initial state)  
                    initial_state = simulation['initial_condition']['xk_ini'][state_idx,:]
                    initial_state_expanded = np.ones_like(history_x[:,state_idx])*initial_state
                    try:
                        initial_Px = simulation['initial_condition']['Pxk_ini'][state_idx, state_idx]
                    except:
                        initial_Px = simulation['initial_condition']['Pxk_ini'][state_idx,0]
                    initial_Px_expanded = np.ones_like(history_x[:,state_idx])*initial_Px

                    # during KF
                    try:
                        X = np.array(simulation['filtering']['x'])
                    except:
                        X = np.array(simulation['x'])
                    state = X[:, state_idx, 0]

                    try:
                        Pxk = np.array(simulation['filtering']['Px'])
                    except:
                        Pxk = np.array(simulation['Px'])
                    uncertainty = Pxk[:, state_idx, state_idx]  # diagonal elements

                    try:
                        eps_innov = np.array(return_dict['filtering']['x_error'][:,DEKF_idx])
                        eps_innov_ensemble.append(eps_innov)
                    except:
                        pass
                    
                    # after KF: future
                    try:
                        X_future = np.array(simulation['future']['x'])
                    except:
                        X_future = np.array(simulation['x_next'])
                    state_next = X_future[:, state_idx, 0]

                    # add to ensemble predictions
                    state_history_ensemble.append(initial_state_expanded)
                    Pxk_history_ensemble.append(initial_Px_expanded)
                    state_ensemble.append(state)
                    Pxk_ensemble.append(uncertainty)
                    state_future_ensemble.append(state_next)
            except:
                for DEKF_idx in range(n_ensemble):  
                    simulation = return_dict
                    # history (=initial state)  
                    initial_state = simulation['initial_condition']['xk_ini'][DEKF_idx, state_idx,:]
                    initial_state_expanded = np.ones_like(history_x[:,state_idx])*initial_state
                    try:
                        initial_Px = simulation['initial_condition']['Pxk_ini'][DEKF_idx, state_idx, state_idx]
                    except:
                        initial_Px = simulation['initial_condition']['Pxk_ini'][DEKF_idx, state_idx,0]
                    initial_Px_expanded = np.ones_like(history_x[:,state_idx])*initial_Px

                    # during KF
                    X = np.array(simulation['filtering']['x'])[:,DEKF_idx]
                    state = X[:, state_idx, 0]

                    Pxk = np.array(simulation['filtering']['Px'])[:,DEKF_idx]
                    uncertainty = Pxk[:, state_idx, state_idx]  # diagonal elements

                    try:
                        eps_innov = np.array(return_dict['filtering']['x_error'][:,DEKF_idx])
                        eps_innov_ensemble.append(eps_innov)
                    except:
                        pass
                    try:
                        Qx = np.array(return_dict['filtering']['Qx'][:,DEKF_idx])
                        Qx_ensemble.append(Qx)
                    except:
                        pass
                    
                    # after KF: future
                    try:
                        X_future = np.array(simulation['future']['x'])
                        raise Exception("Not checked for ensemble")
                        state_next = X_future[:, state_idx, 0]
                    except:
                        state_next = []
                    # add to ensemble predictions
                    state_history_ensemble.append(initial_state_expanded)
                    Pxk_history_ensemble.append(initial_Px_expanded)
                    state_ensemble.append(state)
                    Pxk_ensemble.append(uncertainty)
                    state_future_ensemble.append(state_next)
            mean_state_history = np.mean(state_history_ensemble, axis=0)
            mean_uncertainty_history = np.mean(Pxk_history_ensemble, axis=0)
            mean_state = np.mean(state_ensemble, axis=0)
            mean_uncertainty = np.mean(Pxk_ensemble, axis=0)
            mean_state_future = np.mean(state_future_ensemble, axis=0)

            # log tracking error (mean)
            if state_idx == 1:   # focus on omega tracking
                #plt.suptitle(f"Filtering: {model}_{Kalman_method}")     # subtitle from filtering figure
                true = real_y_KF[:, state_idx]
                noisy = filter_x[:, state_idx]
                
                predicted = mean_state
                eps_noisy = predicted - noisy
                eps_filter = predicted - true
                groundtruth_error = noisy-true

                if len(eps_innov_ensemble) > 0:
                    eps_title = "Innovation error"
                    try:
                        Qx_data = [np.abs(Qx_ensemble_i[:,state_idx,state_idx]) for Qx_ensemble_i in Qx_ensemble]
                        eps_innov_data = [np.abs(eps_innov_ensemble_i[:,state_idx,0]) for eps_innov_ensemble_i in eps_innov_ensemble]
                    except:
                        eps_innov_data = [np.abs(eps_innov_ensemble_i[:,0,0]) for eps_innov_ensemble_i in eps_innov_ensemble]
                    # eps_plot = eps_innov_data
                    eps_plot = Qx_data
                        
                selection_criteria = np.array(Qx_data)

                if len(eps_innov_ensemble) > 0: 
                    ### selects single filter over whole filter traj to be "the best"
                    # eps_innov_data = [np.abs(eps_innov_ensemble_i[:,state_idx,0]) for eps_innov_ensemble_i in eps_innov_ensemble]
                    # mean_innov = np.mean(eps_innov_data)
                    # mean_innovs_each = [np.mean(eps_innov_i) for eps_innov_i in eps_innov_data]
                    # mean_innovs_each = np.where(np.isnan(mean_innovs_each), 1e8, mean_innovs_each)
                    # best_filter_idx = np.argmin(mean_innovs_each)
                    # innov_best = np.min(mean_innovs_each)
                    # print(f"Mean Innovation error: {mean_innov}")
                    # print(f"Best Innovation error: {innov_best} for filter {best_filter_idx}")      # select single filter
                    
                    # eps_each = [np.mean(np.abs(state_ensemble_i - true)) for state_ensemble_i in state_ensemble]
                    # mean_eps = np.mean(eps_each)
                    # best_filter_idx2 = np.nanargmin(eps_each)
                    # eps_best = np.nanmin(eps_each)
                    # print(f"Mean Filtering error: {mean_eps}")
                    # print(f"Best Filtering error: {eps_best} for filter {best_filter_idx2}")
                    
                    ## selects at each time the best filter over whole filter traj (Selection S=Q_omega)
                    ones_correct_shape = np.ones_like(np.array(eps_innov_data))
                    idxs = np.array([ones_correct_shape[i]*i for i in range(len(ones_correct_shape))])
                    best_idxs, _ = get_best_x_ensembles(idxs, selection_criteria, 1)
                    best_innov, _ = get_best_x_ensembles(np.array(eps_innov_data), selection_criteria, 1)
                    print(f"Best Innovation error at each time: {np.mean(best_innov)}")
                omega_future = real_y_future[:, state_idx]
                eps_future = mean_state_future - omega_future
    
    # plot tracking/prediction error
    results['mean_innov'] = np.mean(best_innov)           # mean innov
    results['end_innov'] = np.mean(best_innov[:,-100:])
    
    ## physical parameter tracking
    p_dict = {key: [] for key in MC.trainable_names}
    p_best_dict = {key: [] for key in MC.trainable_names}
    n_p = len(MC.trainable_names)

    p_all_param_all = np.array(return_dict['filtering']['w'])[:,:, -n_p:,0]
    for i, key in enumerate(MC.trainable_names):
        p_all = p_all_param_all[:,:,i].T
        p_best, _ = get_best_x_ensembles(p_all, selection_criteria, 1)
        p_best_dict[key] = p_best[0]

    for key in MC.trainable_names:
        if key in MC.p_true:
            p_true = MC.p_true[key]
            p_error = np.abs(p_best_dict[key] - p_true)
            print(f"Mean {key} error: {np.mean(p_error)}")
                        
            results[f'{key}_error'] = np.mean(p_error)
            results[f'{key}_error_end'] = np.mean(p_error[-100:])
    return results

def log_WandB(results_list, log=True, verbose=False):
    import wandb
    keys = list(results_list[0].keys())
    for key in keys:
        try:
            maxlength = np.max([len(results[key]) for results in results_list])
            
            ## if I want to calculate mean of all lasting lists
            means = []
            stds = []
            for i in range(maxlength):
                vals = []
                for results in results_list:
                    try:
                        vals.append(results[key][i])
                    except:
                        pass
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            avg_result = means

            ## if I want to calculate the mean of all lists, where finished lists are padded with their ending value
            #padded_metric_results = np.array([list(result_i[key]) + [result_i[key][-1]] * (maxlength - len(result_i[key])) for result_i in results_list])
            #avg_result = np.mean([results[key] for results in padded_metric_results], axis=0)
            
            if log: 
                wandb.log({key: avg_result})
                print(f"Logged {key}")
            elif verbose:
                print(f"{key}: {avg_result}")
            else:
                raise Exception("No action specified")
        except:
            try:
                avg_result = np.mean([results[key] for results in results_list], axis=0)
                if log:
                    wandb.log({key: avg_result})
                    print(f"Logged {key}")
                elif verbose:
                    print(f"{key}: {avg_result}")
                else:
                    raise Exception("No action specified")
            except:
                print(f"Could not log {key}")
                
def simulate_ensemble(models, MC, n, which_dim=1, sz=50, modulo=None, verbose=0, plot=True):
    """ 
    get ensemble prediction given a set of models on a given dataset
    """
    x0 = list(MC.filter_x[0])
    if n['x'] == 3: x0.append(0.6)
    if plot:
        plt.figure()
        plt.plot(MC.filter_x[:,which_dim], 'k', label='measurements')
        ylim = plt.ylim()
    n_ensemble = len(models)
    Xsims_list = []
    for i in range(n_ensemble):
        u_sz = []
        x0_sz = []
        for k in range(int(len(MC.filter_u)/sz)):
            u_test_k = MC.filter_u[sz*k:sz*(k+1)]
            x_test_k = MC.filter_x[sz*k:sz*(k+1)]
            if n['x'] == 3: 
                x0_k = np.append(x_test_k[0], 0.6)
            else:
                x0_k = x_test_k[0]
            u_sz.append(u_test_k)
            x0_sz.append(x0_k)
            
        X_sim, Z = models[i].simulation(x0_sz, u_sz, jit=True, modulo=list(modulo), return_z=True, verbose=verbose)  # make simulation
        Xsims = np.concatenate(X_sim)
        if plot: plt.plot(Xsims[:,which_dim])
        Xsims_list.append(Xsims)
    Xsims = np.array(Xsims_list)
    if plot:
        plt.legend()
        plt.show()
    MSE = [np.mean((Xsims[i, :,which_dim] - MC.filter_x[:Xsims.shape[1],which_dim])**2) for i in range(n_ensemble)]
    MSE = np.mean(MSE)
    return MSE


def get_best_x_ensembles(camsimu_results, camsimu_innov_error, subselect_n_ensembles, transpose=False, mw_len=1000, return_best_idxs=False):
    """
    camsimu.shape = n_ensemble x time
    camsimu_innov_error.shape = n_ensemble x time
    
    returns 
    best_array.shape: n_ensemble_selected x time
    worst_array.shape: n_ensemble - n_ensemble_selected x time
    
    
    if transpose then all the shapes are transposed: i.e (time x ...)
    """
    if transpose:
        if len(camsimu_results.shape) > 2:
            if len(camsimu_results.shape) > 3:
                raise Exception("Not implemented for >3D input")
            camsimu_results = np.transpose(camsimu_results, axes=(1,0,2))
        else:
            camsimu_results = camsimu_results.T
        camsimu_innov_error = camsimu_innov_error.T
    
    if True:
        best_array = []
        worst_array = []
        mw = mw_len
        mw_list = []
        ## get moving average (over 50 timesteps back and forward) of innovation error
        for i in range(0, camsimu_innov_error.shape[1]):
            if i < mw/2:
                mw_innov = camsimu_innov_error[:,:int(i+mw/2)]
            else:
                mw_innov = camsimu_innov_error[:,int(i-mw/2):i+int(mw/2)]
            mw_innov = np.mean(mw_innov, axis=1)
            mw_list.append(mw_innov)
        # select best x ensemble members based on moving average of innovation error
        camsimu_innov_error_mw = np.array(mw_list).T
        for timestep in range(camsimu_innov_error_mw.shape[1]):
            mw_innov = camsimu_innov_error_mw[:,timestep]
            sorted_idxs = np.argsort(mw_innov)
            # sorted_idxs = np.argsort(camsimu_innov_error[:,timestep])
            best_idxs = sorted_idxs[:subselect_n_ensembles]
            worst_idxs = sorted_idxs[subselect_n_ensembles:]
            
            best_array.append(camsimu_results[best_idxs, timestep])
            worst_array.append(camsimu_results[worst_idxs, timestep])
        if not transpose:   # with transposed inputs, output should not be transposed and vice versa
            best_array = np.array(best_array).T
            worst_array = np.array(worst_array).T
        else:
            best_array = np.array(best_array)
            worst_array = np.array(worst_array)
        return best_array, worst_array
    