"""
- Creates/Loads and simulates ensemble DEKF on a single data-trajectory
- Uses final/converged solution to predict/evaluate on whole trajectory
"""

import numpy as np
import jax.numpy as jnp
import copy
import os
from ensemble import simulate_ensemble, get_simulation_savepath

import pickle
def main(queue, weight_row, MC):
    # ensure same seed is used for repeats
    if os.path.exists("rng_seed.pkl"):
        rng_seed = pickle.load(open("rng_seed.pkl", "rb"))
    else:
        rng_seed = np.random.get_state()
        pickle.dump(rng_seed, open("rng_seed.pkl", "wb"))
    np.random.set_state(rng_seed)
    
    debug = False   # set to True to disable jit for debugging purposes
    if debug:
        from jax import config
        config.update("jax_disable_jit", True)
        
    #%% load config of camfollower model and Neural Network structure
    cf_i = MC.cf
    
    #%% set dtype (provided as string)
    if MC.dtype == 'float64':
        from jax.config import config
        config.update("jax_enable_x64", True)
        dtype = jnp.float64
    elif MC.dtype == 'float32':
        dtype = jnp.float32
    else:
        raise Exception("dtype must be either 'float64' or 'float32'")

    #%% HybModel(s) dimensions (n)
    n, _, _ = cf_i.config()
                
    #%% loop over ensemble of H,Beta values and load corresponding pretrained models
    models = []

    if len(np.unique(MC.p_true['V'])) != 1: raise Exception("V should be unique for this to work")
    if len(np.unique(MC.p_true['H'])) != 1: raise Exception("H should be unique for this to work")
    if len(np.unique(MC.p_true['beta'])) != 1: raise Exception("beta should be unique for this to work")
    V_test = np.unique(MC.p_true['V'])[0]
    H_test = np.unique(MC.p_true['H'])[0]
    beta_test = np.unique(MC.p_true['beta'])[0]
    
    SimulationSave, ModelTypeSave = get_simulation_savepath(MC.EnsembleSave, MC.model, MC.Kalman_method, MC.p_true, V_test)

    # get physical param (saved fully)
    return_dict = os.path.join(SimulationSave, "return_dict.pkl")
    with open(return_dict, "rb") as f:
        results = pickle.load(f)
        weights_p = results['filtering']['w']
        n_p = weights_p.shape[-2]
        Qx = results['filtering']['Qx']
    
    if MC.model != 'physics':
        zipped_weights_loc = os.path.join((ModelTypeSave).replace("results", "results_zipped", 1), f"w_H{str(H_test).replace('.', '_')}.npz")
        unzipped_weights_path = zipped_weights_loc.replace(".npz", "/w.npy")

        # get full NN weights --  first unzip (could be automated, manual for now)
        if not os.path.exists(unzipped_weights_path):
            raise Exception(f"Unzipped weights not found at {unzipped_weights_path}: \n Create first using main.py or main_loop.py with save_w_full = True. \n Afterwards zip using visualization/remove_w_from_disk.py and unzip manually. \n Had to be done one by one due to memory constraints (not all weights could be saved at the same time).")
        unzipped_weights = np.load(unzipped_weights_path, mmap_mode='r')
        weights = unzipped_weights[weight_row]
        
        # check if correct array is loaded
        assert(np.all(weights[:,:,-n_p:] == weights_p))
    else:
        weights = weights_p
        

    # subselect on ensemble based on Qx_end! --> get single model for predictions 
    Qx_end = Qx[-1]
    weights_end = weights[-1]
    Qx_omega = Qx_end[:,1,1]
    lowest_Qx = np.argmin(Qx_omega)
    weights = weights_end[lowest_Qx]
    
    #%% create model with normal load (to get scaler and working KF instance)
    P = copy.deepcopy(MC.p_trainset)
    hybMod = MC.make_model(dt=MC.dt_sub, P=P, n=n, f=cf_i.f, g=cf_i.g, cfg_eta=MC.cfg_eta, dtype=dtype) #, method="RK")
    if MC.model == 'physics': assert(hybMod.NNmodel.model == None)

    #%% save p dictionary
    p_preload = copy.deepcopy(hybMod.p)
    #%% load initial models (for scaler)
    if MC.model != 'physics':
        n_layers, n_hidden, activ = len(MC.cfg_eta['n_hidden']), MC.cfg_eta['n_hidden'][0], MC.cfg_eta['activation_hid'][0]
        layer_folder = os.path.join(MC.path, f"layers_{n_layers}")
        activ_folder = os.path.join(layer_folder, f"activ_{activ}")
        neuron_folder = os.path.join(activ_folder, f"neurons_{n_hidden}")
        path_i = os.path.join(neuron_folder, f"hybMod")                
    elif MC.model == 'physics': 
        path_i = os.path.join(MC.path, f"hybMod")
    # load model ## https://github.ugent.be/cevheck/HMT - contact cevheck.vanheck@ugent.be to discuss access
    hybMod.load_from(path=path_i)
    
    for key,val in hybMod.p.items():
        if p_preload[key] == val: continue  # matches both before and after loading -- no need to examine
        if MC.HB_trainable and (key in ['H', 'beta']): continue         # these are already changed by grid
        if MC.m_trainable and (key in ['m'])         : continue         # these are already changed by grid
        if MC.V_trainable and (key in ['V'])         : continue         # these are already changed by grid
        if "trainable_param" in str(p_preload[key]):
            if "trainable_param" in str(val):
                p_preload[key].val = val.val
            elif len(val.shape) > 0:
                # take mean value of trained param as initial guess of trainable param // alternative to taking random guess in p_init
                meanval = np.mean(val)
                p_preload[key].val = meanval
        else:
            if p_preload[key].astype("float32") == val.astype("float32"): continue  # matches both before and after loading -- no need to examine
            raise Exception(f"Parameter {key} is not equal before and after loading model for unknown reason")
    
    # update hybmodel with values calculated in loop above (e.g. Ra --> mean(Ras))
    hybMod.change_p(p_preload, compile=True)
        
    KF = MC.dekf(MC.Qx, MC.R, state_dim=len(MC.IC), Pw_ini=MC.Pw_ini, Qw=MC.Qw, 
                input_dim=n['u'], models=[hybMod], xk_ini=MC.IC, C=MC.C, dt=MC.dt_KF, 
                trainable_names=MC.trainable_names, p_true=MC.p_true, MC=MC)

    #%% update to converged model weights    
    NNweights, p_weights = KF.wk_array_to_NNinput(wk=weights)
    hybMod.NNmodel.weights = NNweights
    updated_p = hybMod.p | p_weights
    hybMod.p = updated_p

    #%% Perform KF simulation + future prediction
    ## DUAL KALMAN FILTER    
    modulo = np.zeros(n['x'])
    modulo[0] = 2*np.pi
    models = [hybMod]
    MSE = simulate_ensemble(models, MC, n, modulo=modulo, sz=1000, verbose=0, plot=0)

    try:
        queue.put(MSE)
        return
    except:
        return MSE
    

