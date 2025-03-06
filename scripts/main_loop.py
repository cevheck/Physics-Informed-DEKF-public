"""
main.py but looping over multiple settings of Q_NN
"""
#%% imports 
from multiprocessing import Process, Queue
import time
import numpy as np
import matplotlib.pyplot as plt
import gc

#--------- Settings --------- #
#%% which model
hybrid = True
bb = True
physics = True
modelopts = []
if hybrid: modelopts.append('hybrid')
if bb: modelopts.append('blackbox')
if physics: modelopts.append('physics')
kalman_opts = ['PDEKF']     #['PDEKF', 'DEKF] 

#%% variables to set
overwrite = False
plot_now = False
save_now = True
save_w_full = False     # True: save the results with the full weight vector (takes a lot of memory on disc); False: Only save physical part of weight vector
verbose = 1             # verbosity of filtering loop

# Other settings are found in the object "MC"

#--------- Do not change below --------- #
#%% pathing
import pathmagic as pathmagic
ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()

#%% Config Object -- functions as config file but implemented as class. Initializes mainly fixed things
from configuration import MyConfig as MC 
from configuration import update_MC, configurate_MC, prepare_MC_for_filtering

#%% update MC object to correct model (custom choises made here!) -- programmed in src to be easily imported at multiple locations
configurate_MC(MC)

#%% custom imports
import setup_src
from load_data import get_data, get_filter_data, load
make_new, dekf, main, plot_ensemble_results, WandB_ensemble, save_ensemble, log_WandB = setup_src.custom_imports()

Q_NN_added_rel_opts = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for Q_NN_added_rel in Q_NN_added_rel_opts:
    #%% load data
    X_real, U_real = load(DataFolder)
    #%% subsample data
    X_real_sampled = [X[::int(MC.dt/MC.dt_original),:] for X in X_real]
    U_real_sampled = [U[::int(MC.dt/MC.dt_original),:] for U in U_real]

    ## no noise added because noise is inherent to the data in experimental setting
    X = X_real_sampled.copy()
    U = U_real_sampled.copy()

    #%% hybrid model settings
    MC.make_model = setup_src.make_model
    MC.get_cf = setup_src.get_cf
    #%% initialization of KF
    ## Initial state guess
    MC.IC = np.reshape(np.array([np.pi, 10.0]), newshape=(-1,1))     # initial guess: [theta_ini, omega_ini]

    ## measurement matrix
    MC.C = np.array([[1, 0], [0, 1]])  # fully observed
    #MC.C = np.array([[1, 0], [0, 0]])  # only theta observed
    # MC.C = np.array([[0, 0], [0, 1]])  # only w observed (relative encoder)

    #%% Perform KF simulation + future prediction
    camvoltages = np.unique([U_simu[0,0] for U_simu in U])
    cammasses = np.unique([U_simu[0,1] for U_simu in U])
    camheights = np.unique([U_simu[0,2] for U_simu in U])
    cambetas = np.unique([U_simu[0,3] for U_simu in U])

    #%%
    # total size of each simulation
    total_samples = int(MC.multipletest / MC.dt)
    ### Start main loop
    for camheight_test in camheights:
        for cambeta_test in cambetas:
            for cammass_test in cammasses:   
                testX_set = [X[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]                          # noisy measurements
                testU_set = [U[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]                          # noisy measurements
                realX_set = [X_real_sampled[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]             # real groundtruth
                
                V_list = np.unique([U_simu[0,0] for U_simu in testU_set])
                for V_i in V_list:
                    testX, testU, realX = get_data(MC, testX_set, testU_set, realX_set, V_i, total_samples)
                    total_samples = testX.shape[0]
                    
                    # move H and beta to p-vector
                    testV = testU[:,0]                              # get ground truth V-value
                    testm = testU[:,1]                              # get ground truth m-value
                    testH = testU[:,2]                              # get ground truth H-value
                    testbeta = testU[:,3]                           # get ground truth beta-value
                    p_true = {'H': testH, 'beta': testbeta, 'm': testm, 'V': testV}

                    for model in modelopts:
                        for Kalman_method in kalman_opts:
                            # update hyperparam of MC dependent on type of filter and model
                            wandb_results = 1 if model == 'blackbox' else 0
                            update_MC(MC, which_model=model, which_kalman_opt=Kalman_method, wandb_results=wandb_results)   
                            MC.Q_NN_added_rel = Q_NN_added_rel

                            EnsembleSave = setup_src.get_ensemble_save_path(ResultFolder, MC)
                            if Q_NN_added_rel != 0 and model != 'physics':
                                EnsembleSave = EnsembleSave + f'_Q_NN_added_rel_{Q_NN_added_rel}/'
                            if not make_new(model, Kalman_method, p_true, EnsembleSave, overwrite, plot_now, V_i): continue
                            print(f'Running {model} model with {Kalman_method} method for H={camheight_test}, beta={cambeta_test}, m={cammass_test}, V={V_i} (Q_NN_added_rel={Q_NN_added_rel})')

                            # update filtering matrices, dekf objects, filtering data, pathing for current model and settings
                            prepare_MC_for_filtering(MC, model, Kalman_method, setup_src, get_filter_data, ResultFolder, dekf, p_true, testX, testU, realX, total_samples)
  
                            t1 = time.time()                    
                            processes = []
                            if True:  # perform with multiprocessing (ensures each dekf instance correctly releases memory)
                                queue = Queue()
                                p = Process(target=main, args=[queue, MC, verbose], kwargs={})
                                p.start()
                                return_dict = queue.get()
                                p.join()
                            else:   # perform without multiprocessing
                                return_dict = main(None, MC, verbose)
                            t2 = time.time()
                            if MC.n_ensemble > 1: print(f'Time for ensemble simulation: {t2 - t1}s') 


                            plotting_settings = plot_weights, plot_P, plot_K, plot_deriv, verbose =  True, False, False, False, 1
                            return_dict = dict(return_dict) # convert manager.dict (proxy object) to python dict
                                                            # ensure this dict does not contain JAX arrays because then JAX is used in this script 
                                                                # and multiprocessing does not work anymore on future calls
                            
                            MC.real_y = realX
                            if plot_now:
                                plot_results = True
                                NN_investigation = True if model == 'hybrid' else False
                                plot_ensemble_results(return_dict, MC, plotting_settings, plot_results, NN_investigation)
                                plt.show()
                            if save_now:  ## save the results (merging offline)
                                results = save_ensemble(EnsembleSave, return_dict, MC, plotting_settings, V_test=V_i, overwrite=overwrite, save_w_full=save_w_full)
                            del return_dict
                            gc.collect()
                                
#%%
print("End of code")