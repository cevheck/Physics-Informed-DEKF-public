"""
Config object for WandB hyperparameter tuning purposes
For full commentated code, reference to "configuration.py"
"""
import numpy as np
import copy

class MyConfig:
    np.random.seed(42)
    
    wandb = True            # is the setting for tuning with WandB
    plot_extras = False     
    overwrite = False
    plot_now = False
    plot = False
    verbose = False
    n_ensemble = 10  # amount of ensemble members

    dt = 0.005
    R_opt = 0.001

    ### WandB parameters ###
    parameters_dict = {
                    'model': {'values': ['hybrid','physics','blackbox']}, 
                    # 'kalman_method': {'values': ['PDEKF', 'DEKF']}, 
                    'Kalman_method': {'values': ['PDEKF']}, 
                    'transfer_learning': {'values': ['method1', False]}, 
                    'Qx_init': {'values': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]},
                    'weight_uncertainty': {'values': [1e-7, 1e-6, 1e-5, 1e-4]},
                    'weight_uncertainty_added': {'values': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]},
                    'Q_NN_rel': {'values': [0, 1/100, 1/10, 1, 10, 100]},
                    'Q_NN_added_rel': {'values': [0, 1/100, 1/10, 1, 10, 100]},
                    'QH_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'QH_added_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'Qbeta_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'Qbeta_added_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'QRa_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'QRa_added_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'QD_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'QD_added_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'QV_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                    'QV_added_rel': {'values': [0, 1/100, 1/10, 1, 10, 100, 1000, 10000]},
                   }
    
    ### Model parameters ###
    activ = 'mish'
    n_layers = 1 #2
    n_neurons = 32
    dtype = 'float32'

    cfg_eta = {'n_hidden': [n_neurons]*n_layers,  # number of neurons per layer --> e.g: 2 layers = [16,32]
        'activation_hid': [activ]*n_layers,  # activation in each layer    --> e.g: 2 layers = ['tanh','relu']
        'activation_out': 'linear',  # activation of output layer
        }    
    

    ### Data Parameters ###
    dt_original = 0.0005
    freq_original = 1/dt_original

    ### Filtering Parameters ###
    # which param to look at
    H = [0.01, 0.03, 0.05, 0.07]
    beta = [np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4]

    multipletest = 60 # concatenate simulations until a total of 30s
    assert(multipletest)
    
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
            
def configurate_MC(MC):
    MC.nq = 6

    MC._HB_trainable = True
    MC._m_trainable = False
    MC._V_trainable = True
    MC.include_friction_term = False
    # assert(sum([HB_trainable, m_trainable, V_trainable]) <= 1)

    #%% choose filter
    MC.n_ensemble = 10
    MC.adaptive = True
    MC.transfer_learning = "method1" # method1 = freeze all but last layer of NN
    if MC.adaptive: MC.window = 30
    if MC.adaptive: MC.delta = 1/1000

def update_wandb_hyperparam(MC, config):
    MC.model = config.model
    MC.Kalman_method = config.Kalman_method
    
    MC.transfer_learning = config.transfer_learning
    MC.Qx_init = config.Qx_init
    
    MC.weight_uncertainty = config.weight_uncertainty
    MC.weight_uncertainty_added = config.weight_uncertainty_added
    MC.Q_NN_rel = config.Q_NN_rel
    MC.Q_NN_added_rel = config.Q_NN_added_rel
    MC.QH_rel = config.QH_rel
    MC.QH_added_rel = config.QH_added_rel
    MC.Qbeta_rel = config.Qbeta_rel
    MC.Qbeta_added_rel = config.Qbeta_added_rel
    MC.QRa_rel = config.QRa_rel
    MC.QRa_added_rel = config.QRa_added_rel
    MC.QD_rel = config.QD_rel
    MC.QD_added_rel = config.QD_added_rel
    MC.QV_rel = config.QV_rel
    MC.QV_added_rel = config.QV_added_rel
    
def prepare_MC_for_filtering(MC, model, Kalman_method, setup_src, get_filter_data, ResultFolder, dekf, p_true, testX, testU, realX, total_samples, switching_times):
    MC.model = model
    MC.Kalman_method = Kalman_method
    MC.path = setup_src.getmodelpath(model, Kalman_method, ResultFolder, MC.Ra, MC.include_friction_term, MC.global_model, nq=MC.nq)                
    MC.cf = MC.get_cf(model, MC.data, filter_Tm=False, HB_trainable=MC.HB_trainable, m_trainable=MC.m_trainable, V_trainable=MC.V_trainable, Ra=MC.Ra, include_friction_term=MC.include_friction_term, nq=MC.nq)
    MC.p_trainset = MC.cf.p_initial()

    # update KF uncertainties
    setup_src.update_uncertainties(MC)   
    setup_src.create_uncertainty_matrices(MC)
    
    testU_i = copy.deepcopy(testU)
    if (MC.HB_trainable): testU_i = np.delete(testU_i, [2,3], axis=1)       # delete H in input array -- want it in the p-vector
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
    MC.switching_times = switching_times
    
