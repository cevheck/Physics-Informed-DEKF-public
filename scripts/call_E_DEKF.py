"""
Creates/Loads and simulates ensemble DEKF on a single data-trajectory
"""
import numpy as np
import jax.numpy as jnp
import jax
import copy
import os
from ensemble import simulate_ensemble
import pickle

def main(queue, MC, verbose=1):
    # ensure same seed is used for repeats
    if os.path.exists("rng_seed.pkl"):
        rng_seed = pickle.load(open("rng_seed.pkl", "rb"))
    else:
        rng_seed = np.random.get_state()
        pickle.dump(rng_seed, open("rng_seed.pkl", "wb"))
    np.random.set_state(rng_seed)
    
    debug = False   # set to True to disable jit for debugging purposes --> really slow, so do NOT use for full simulations
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
    
    #%% create n initial conditions on the parameter vector p
    grid = []
    chosenHBs = []
    for g_i in range(MC.n_ensemble):
        p_dict_i = {}
        if MC.HB_trainable:
            chosenH = [0.01, 0.03, 0.05, 0.07][g_i%4]
            beta_values = [2*np.pi/4, 3*np.pi/4, 4*np.pi/4, 5*np.pi/4]
            beta_values_repeat_for_each_H = [item for item in beta_values for _ in range(2)]
            chosenB = beta_values_repeat_for_each_H[g_i%(len(beta_values_repeat_for_each_H))]
            while (chosenH, chosenB) in chosenHBs:        # avoid duplicates
                g_j = np.random.randint(0, len(beta_values_repeat_for_each_H))
                chosenB = beta_values_repeat_for_each_H[g_j]
            p_dict_i['H'] = [chosenH, 0.005, 0.075]
            p_dict_i['beta'] =  [chosenB, np.pi/4, 7*np.pi/4]
            chosenHBs.append((chosenH, chosenB))
        if MC.m_trainable:
            m_values = [0.5, 0.65, 0.7325, (0.7325+0.9379)/2, 0.9379, (0.9379+1.1238)/2, 1.1238, (1.1238+1.3067)/2, 1.3067, 1.6]
            p_dict_i['m'] = [m_values[g_i], 0.0, 2.0]
        if MC.V_trainable:
            V_values = [3.6, 4.35789474, 5.11578947, 5.87368421, 6.25263158, 6.63157895, 7.01052632, 7.76842105, 8.14736842, 8.90526316]
            p_dict_i['V'] = [V_values[g_i], 3, 10]
        grid.append(p_dict_i)
                
    #%% loop over ensemble of H,Beta values and load pretrained models
    models = []
    for idx in range(MC.n_ensemble):
        #%% make p trainable
        P = copy.deepcopy(MC.p_trainset)
        for key,val in grid[idx].items():
            P[key] = val
            
        #%% create models
        hybMod = MC.make_model(dt=MC.dt_sub, P=P, n=n, f=cf_i.f, g=cf_i.g, cfg_eta=MC.cfg_eta, dtype=dtype, rng_seed=idx*10) #, method="RK")
        if MC.model == 'physics': assert(hybMod.NNmodel.model == None)  # physics model should not have a NNmodel

        #%% save p dictionary
        p_preload = copy.deepcopy(hybMod.p)
        #%% load pretrained models
        # get correct pathing
        if MC.model != 'physics':
            n_layers, n_hidden, activ = len(MC.cfg_eta['n_hidden']), MC.cfg_eta['n_hidden'][0], MC.cfg_eta['activation_hid'][0]
            layer_folder = os.path.join(MC.path, f"layers_{n_layers}")
            activ_folder = os.path.join(layer_folder, f"activ_{activ}")
            neuron_folder = os.path.join(activ_folder, f"neurons_{n_hidden}")
            path_i = os.path.join(neuron_folder, f"hybMod")            
        elif MC.model == 'physics': 
            path_i = os.path.join(MC.path, f"hybMod_H{P['H'][0]}_beta{round(P['beta'][0],2)}")
        
        # load model ## https://github.ugent.be/cevheck/HMT - contact cevheck.vanheck@ugent.be to discuss access
        hybMod.load_from(path=path_i)
        
        # ensure that trainable parameters equal those loaded for H,Beta
        if MC.HB_trainable:
            assert((hybMod.p['H'].val == P['H'][0]) and (hybMod.p['H'].min_val == P['H'][1]) and (hybMod.p['H'].max_val == P['H'][2]))                      # this should not be overwritten by load_from
            assert((hybMod.p['beta'].val == P['beta'][0]) and (hybMod.p['beta'].min_val == P['beta'][1]) and (hybMod.p['beta'].max_val == P['beta'][2]))    # this should not be overwritten by load_from
        
        # update p with values from grid
        if MC.m_trainable or MC.V_trainable:
            for key,val in grid[idx].items():
                P[key] = val
        
        # additional checks/updates for each element in p
        for key,val in hybMod.p.items():
            if p_preload[key] == val: continue  # matches both before and after loading -- no need to examine
            if MC.HB_trainable and (key in ['H', 'beta']): continue         # these are already changed by grid
            if MC.m_trainable and (key in ['m'])         : continue         # these are already changed by grid
            if MC.V_trainable and (key in ['V'])         : continue         # these are already changed by grid
            if "trainable_param" in str(p_preload[key]):    # trainable but not in grid (e.g. Ra, D)
                if "trainable_param" in str(val):       # is a "trainable_param" object -> has a .val attribute giving the current estimate
                    p_preload[key].val = val.val
                elif len(val.shape) > 0:                # is a list of estimates for the corresponding list of training trajectories
                    # take mean value of trained param as initial guess of trainable param // alternative to taking random guess in p_init
                    meanval = np.mean(val)
                    p_preload[key].val = meanval
            else:   
                if p_preload[key].astype("float32") == val.astype("float32"): continue  # matches both before and after loading -- no need to examine
                raise Exception(f"Parameter {key} is not equal before and after loading model for unknown reason")
        
        
        # update hybmodel with values calculated in loop above (e.g. Ra --> mean(Ras))
        hybMod.change_p(p_preload, compile=True)
        
        # add to ensemble
        models.append(hybMod)
    
    #%% Perform KF simulation + future prediction
    ## DUAL KALMAN FILTER
    KF = MC.dekf(MC.Qx, MC.R, state_dim=len(MC.IC), Pw_ini=MC.Pw_ini, Qw=MC.Qw, 
                 input_dim=n['u'], models=models, xk_ini=MC.IC, C=MC.C, dt=MC.dt_KF, 
                 trainable_names=MC.trainable_names, p_true=MC.p_true, MC=MC)        

    KF.simulate(filter_y=MC.filter_x, filter_u=MC.filter_u, history_y=MC.history_x, history_u=MC.history_u, 
                future_y=MC.future_x, future_u=MC.future_u, T=MC.T, verbose=verbose, cf=cf_i)

    try:
        queue.put(KF.simulation)    # queue for multiprocessing
        return
    except:
        return KF.simulation        # normal return
    

