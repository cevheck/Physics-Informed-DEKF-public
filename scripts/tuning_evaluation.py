"""
tuning file copy to check/compare results of custom parameters on the tuning task
"""
#%% imports 
from multiprocessing import Process, Queue
import time
import numpy as np

#--------- Settings --------- #
#%% which model
hybrid = True
bb = False
physics = False
modelopts = []
if hybrid: modelopts.append('hybrid')
if bb: modelopts.append('blackbox')
if physics: modelopts.append('physics')
kalman_opts = ['PDEKF']   

# Other settings are found in the object "MC"

#--------- Do not change below --------- #
#%% pathing
import pathmagic as pathmagic
ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()

#%% Config Object
from configuration import MyConfig as MC 
from configuration import update_MC, configurate_MC, prepare_MC_for_filtering

#%% initialize MC object (custom choises made here!) -- programmed in src to be easily imported at multiple locations
configurate_MC(MC)

#%% custom imports
import setup_src
from load_data import get_filter_data, sample_data, load
make_new, dekf, main, plot_ensemble_results, WandB_ensemble, save_ensemble, log_WandB = setup_src.custom_imports()

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

camvoltages = np.unique([U_simu[0,0] for U_simu in U])
cammasses = np.unique([U_simu[0,1] for U_simu in U])
camheights = np.unique([U_simu[0,2] for U_simu in U])
cambetas = np.unique([U_simu[0,3] for U_simu in U])

#%% get data for tuning (N samplesets)
N = 20
resultingX_sets, resultingU_sets, resultingrealX_sets, HB_combinations = sample_data(N, camheights, cambetas, cammasses, camvoltages, X, U, X_real_sampled, MC)

#%%
def train():
    #%% start iterating over tuning sets
    results_list = []
    for idx in range(N):
        HBcombo = HB_combinations[idx]
        camheight_test, cambeta_test = HBcombo
        
        testX = resultingX_sets[idx]
        testU = resultingU_sets[idx]
        realX = resultingrealX_sets[idx]
        total_samples = testX.shape[0]

        # move H and beta to p-vector
        testV = testU[:,0]                            # get ground truth H-value
        testm = testU[:,1]                          # get ground truth beta-value
        testH = testU[:,2]                            # get ground truth H-value
        testbeta = testU[:,3]                         # get ground truth beta-value
        p_true = {'H': testH, 'beta': testbeta, 'm': testm, 'V': testV}     

        for model in modelopts:
            for Kalman_method in kalman_opts:
                # update hyperparam of MC dependent on type of filter and model
                update_MC(MC, which_model=model, which_kalman_opt=Kalman_method, wandb_results=0)   

                # update filtering matrices, dekf objects, filtering data, pathing for current model and settings
                prepare_MC_for_filtering(MC, model, Kalman_method, setup_src, get_filter_data, ResultFolder, dekf, p_true, testX, testU, realX, total_samples)

                t1 = time.time()                    
                if True:    # perform with multiprocessing (ensures each dekf instance correctly releases memory)
                    queue = Queue()
                    p = Process(target=main, args=[queue, MC.n_ensemble, MC, False, False], kwargs={})
                    p.start()
                    return_dict = queue.get()
                    p.join()

                t2 = time.time()
                if MC.n_ensemble > 1: print(f'Time for ensemble simulation: {t2 - t1}s') 


                return_dict = dict(return_dict) # convert manager.dict (proxy object) to python dict
                                                # ensure this dict does not contain JAX arrays because then JAX is used in this script 
                                                    # and multiprocessing does not work anymore on future calls
                
                MC.real_y = realX
                results = WandB_ensemble(return_dict, MC) 
                results_list.append(results)
    print("---"*10)
    print("Done with all simulations -- logging to WandB")
    print("---"*10)
    log_WandB(results_list, log=False, verbose=True)    
# %%
train()