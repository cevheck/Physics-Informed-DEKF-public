"""
Config Object for repeatable initialization in different scripts. 
Similar purpose as config files
"""

import numpy as np
import copy

def configurate_MC(MC):        
    #%% choose model
    MC.nq = 6

    MC._HB_trainable = True
    MC._m_trainable = False
    MC._V_trainable = True
    MC.include_friction_term = False    # include D*w in hybrid model yes/no

    #%% choose filter
    MC.n_ensemble = 10                  # number of ensemble members
    MC.adaptive = True                  # adaptive DEKF: update Q and R
    MC.transfer_learning = None         # None or "method1"      # method1 = freeze all but last layer of NN ; None is update as normal
    if MC.adaptive: MC.window = 30      # moving window to update Q and R
    if MC.adaptive: MC.delta = 1/1000   # delta update parameter for adaptive Q and R

def update_MC(MC, which_model, which_kalman_opt="PDEKF", wandb_results=0):
    hyperparam_update(MC, which_model, wandb_results)    
    
    MC.HB_trainable = MC._HB_trainable  
    MC.m_trainable = MC._m_trainable
    MC.V_trainable = MC._V_trainable

    if which_model == 'hybrid':
        pass                            # already correctly set
    elif which_model == 'blackbox':
        MC.transfer_learning = False    # default = no transfer learning for blackbox
        MC.Q_NN_rel = 10                # overwrite default from hybrid model; we want NN to be trainable
        MC.Q_NN_added_rel = 10          # overwrite default from hybrid model; we want NN to be trainable
    else:
        assert(which_model == 'physics')
        
    if which_kalman_opt == 'DEKF':
        # for non pretrained: we want NN to be adaptable and not bound to initial model
        MC.Q_NN_rel = 10
        MC.Q_NN_added_rel = 10
        MC.transfer_learning = False
        
def hyperparam_update(MC, which_model, wandb_results):
    if wandb_results == 0:
        ### Manual tuning ###
            # those outperformed (for hybrid and physical models) the WandB best parameters and are hence prefered!
        MC.Qx_init = 0.05
        MC.weight_uncertainty = 0.0001
        MC.weight_uncertainty_added = 1e-9
        
        # relative scaling
        MC.Q_NN_rel = 100
        MC.Q_NN_added_rel = 100
        MC.Q_NN_added_rel = 0.1
        
        MC.Q_NN_rel = 0
        MC.Q_NN_added_rel = 0

        MC.QH_rel = 0
        MC.QH_added_rel = 0.5
        
        MC.Qbeta_rel = 0
        MC.Qbeta_added_rel = 100
        
        MC.QRa_rel = 1/10
        MC.QRa_added_rel = 1/10
        
        MC.QD_rel = 1/100
        MC.QD_added_rel = 1/100
        
        MC.Qm_rel = 100
        MC.Qm_added_rel = 100
        
        MC.QV_rel = 10
        MC.QV_added_rel = 100
    elif wandb_results == 1:
        assert(which_model == 'blackbox')
        ### WandB parameters ###
        MC.Qx_init = 0.0005
        MC.weight_uncertainty = 0.00001
        MC.weight_uncertainty_added = 0.0000001
        
        # relative scaling
        MC.Q_NN_rel = 0.01
        MC.Q_NN_added_rel = 0.01

        MC.QH_rel = 1
        MC.QH_added_rel = 0
        
        MC.Qbeta_rel = 100
        MC.Qbeta_added_rel = 0.01
        
        MC.QRa_rel = 1
        MC.QRa_added_rel = 1000
        
        MC.QD_rel = 0.01
        MC.QD_added_rel = 10000
        
        MC.QV_rel = 10000
        MC.QV_added_rel = 1
    else:
        raise NotImplementedError

class MyConfig:
    """
    Class to initialize configuration parameters. Things that are fixed for all simulations will be set here.
    Things that are variable are set in the functions defined above
        - configurate_MC: set the model type (what is trainable, adaptive yes/no, transfer learning yes/no, GA-DEKF yes/no, etc.)
        - update_MC: update hyperparam based on model type (hybrid, blackbox, physics) and Kalman method (PDEKF, DEKF)
        - hyperparam_update: set hyperparameters based on WandB results or manual tuning (used in update_MC)
    """
    np.random.seed(42)
        
    wandb = False
    
    dt = 0.005
        
    R_opt = 0.001
    
    plot_extras = False
    
    ### Model parameters ###
    activ = 'mish'
    n_layers = 1 #2
    n_neurons = 32
    dtype = 'float32'
    #dtype = 'float64'

    cfg_eta = {'n_hidden': [n_neurons]*n_layers,    # number of neurons per layer --> e.g: 2 layers = [16,32]
        'activation_hid': [activ]*n_layers,         # activation in each layer    --> e.g: 2 layers = ['tanh','relu']
        'activation_out': 'linear',                 # activation of output layer
        #'dropout': [0.01]
        }    
    
    ### Data Parameters ###
    dt_original = 0.0005            # measurement dt
    freq_original = 1/dt_original

    multipletest = 60 # concatenate simulations until a total of 30s
    assert(multipletest)

    ### Filtering Parameters ###
    # which param to look at
    H = [0.01, 0.03, 0.05, 0.07]
    beta = [np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]

    ### exit handler (stops all subprocesses at exit of code) ###
    import atexit
    @atexit.register
    def byebye():
        try:
            for process in processes:
                process.terminate()
            print('Clossed all processes at shutdown')
        except:
            print('Clossed without any processes running')
            
def prepare_MC_for_filtering(MC, model, Kalman_method, setup_src, get_filter_data, ResultFolder, dekf, p_true, testX, testU, realX, total_samples):
    MC.model = model
    MC.Kalman_method = Kalman_method
    MC.path = setup_src.getmodelpath(model, Kalman_method, ResultFolder, MC.include_friction_term, nq=MC.nq)                
    MC.cf = MC.get_cf(model, filter_Tm=False, HB_trainable=MC.HB_trainable, m_trainable=MC.m_trainable, V_trainable=MC.V_trainable, include_friction_term=MC.include_friction_term, nq=MC.nq)
    MC.p_trainset = MC.cf.p_initial()

    # update KF uncertainties
    setup_src.update_uncertainties(MC)   
    setup_src.create_uncertainty_matrices(MC)
    
    testU_i = copy.deepcopy(testU)
    if (MC.HB_trainable): testU_i = np.delete(testU_i, [2,3], axis=1)    # delete H in input array -- want it in the p-vector
    if (MC.m_trainable): testU_i = np.delete(testU_i, [1], axis=1)       # delete m in input array -- want it in the p-vector
    if (MC.V_trainable): testU_i = np.delete(testU_i, [0], axis=1)       # delete V in input array -- want it in the p-vector

    # R the KF uses
    MC.R = np.diag([MC.R_opt * MC.dt, MC.R_opt])
    
    MC.history_x, MC.history_u, MC.filter_x, MC.filter_u, MC.filter_xreal, MC.future_x, MC.future_u, MC.T = get_filter_data(testX, testU_i, realX, total_samples, MC.dt)

    ## additional settings that need to be passed
    MC.dt_sub = MC.dt
    MC.dt_KF = MC.dt
    MC.dekf = dekf
    MC.p_true = p_true
