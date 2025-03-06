"""
File to load in certain H,B,m,V values (trajectory) and evaluate the converged model from this setting on the entire simulation.
- Requires the full weights to be saved of the filtering procedure (main.py or main_loop.py with save_w_full=True)
- I afterwards zipped the results for disk memory reasons. Zipped files can be found in results_zipped folder
    Unzip these results to make according predictions again
Prediction results saved to results/predict_...
"""
#%% imports 
from multiprocessing import Process, Queue
import time
import numpy as np

predict_eval = True     # goal of this file

#%% which model
hybrid = False
bb = True
physics = False
modelopts = []
if hybrid: modelopts.append('hybrid')
if bb: modelopts.append('blackbox')
if physics: modelopts.append('physics')
kalman_opts = ['PDEKF']        

#%%
if bb: assert(not physics and not hybrid)   # don't run them together
Q_NN_added_rel_opts = [0.01, 100] if bb else [0]    

#%% variables to set
overwrite = True       # overwrite existing results True/False
save_now = False        # save results True/False
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
make_new, dekf, main, plot_ensemble_results, WandB_ensemble, save_prediction, log_WandB = setup_src.custom_imports(predict_eval=predict_eval)

#%%
plot_now = False    # parameter required from re-using functions from main.py (make_new function). Set fixed to False in this script
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

for Q_NN_added_rel in Q_NN_added_rel_opts:
    ### Start main loop
    for camheight_test in camheights:
        weight_row = 0
        for cambeta_test in cambetas:
            for cammass_test in cammasses:   
                testX_set = [X[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]                          # noisy measurements
                testU_set = [U[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]                          # noisy measurements
                realX_set = [X_real_sampled[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]             # real groundtruth
                V_list = np.unique([U_simu[0,0] for U_simu in testU_set])   # get all unique V-values of the present subset
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
                            ResultSave = setup_src.get_ensemble_save_path(ResultFolder, MC, predict_eval=predict_eval)
                            if Q_NN_added_rel != 0 and model != 'physics': 
                                EnsembleSave = EnsembleSave + f'_Q_NN_added_rel_{Q_NN_added_rel}/'
                                ResultSave = ResultSave + f'_Q_NN_added_rel_{Q_NN_added_rel}/'
                            MC.EnsembleSave = EnsembleSave
                            if not make_new(model, Kalman_method, p_true, ResultSave, overwrite, plot_now, V_i): 
                                weight_row += 1
                                continue

                            # update filtering matrices, dekf objects, filtering data, pathing for current model and settings
                            prepare_MC_for_filtering(MC, model, Kalman_method, setup_src, get_filter_data, ResultFolder, dekf, p_true, testX, testU, realX, total_samples)
  
                            t1 = time.time()                    
                            processes = []
                            if True:  # call in a separate process to avoid memory leak
                                queue = Queue()
                                p = Process(target=main, args=[queue, weight_row, MC], kwargs={})
                                p.start()
                                MSE = queue.get()
                                p.join()
                            else:
                                return_dict = main(None, weight_row, MC, )
                            t2 = time.time()
                            if MC.n_ensemble > 1: print(f'Time for ensemble prediction: {t2 - t1}s') 

                            if save_now:  ## save the results (merging offline)
                                results = save_prediction(ResultSave, MSE, MC, V_test=V_i, overwrite=overwrite)
                            weight_row += 1

                                
#%%
print("End of code")