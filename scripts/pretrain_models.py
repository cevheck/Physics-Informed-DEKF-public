"""
Pretrain models for a (defined) range of 
    - hidden layers
    - hidden neurons
    - activation func
    - starting data (H, beta) or global modem
and save them in folder for further usage
"""
#%%
print('starting main file')

# debug
from jax import config
config.update("jax_disable_jit", False)
    
#%% create which model
hybrid = True
bb = False
physics = False
assert(hybrid + bb + physics == 1)  # only one model at a time
model = "hybrid" if hybrid else "physics" if physics else "blackbox"

#%%
pretrain = True     # True: create pretrained models; False create non-pretrained models
Kalman_method = "PDEKF" if pretrain else "DEKF" 

#%% which models to train
save = False
resave = True      # if True, will overwrite existing models
HB_trainable = False    # if True, H and beta are trainable. If False, they are given
m_trainable = False     # if True, m is trainable. If False, it is given
V_trainable = False     # if True, V is trainable. If False, it is given
include_friction_term = False   # if True, friction term is included in the model dynamics
nq = 6                      # number of inputs to the neural network, c.f.r. scripts/CF_exp/CamFollower... .py

#%% other settings
verb = True         # verbosity
testing = True     # if True, will test the model after training
show_x = True      # if True, will show the x-predictions of testing
if show_x: assert(testing)      # can only show if testing is True
if testing: assert(pretrain)    # only test if pretrain is True
seperated_vars = ['H', 'beta', 'D', 'Ra']   # variables that are fitted seperate for each traj. If not in seperated vars, a single value is fit for all simu's

#%% imports
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

dtype = jnp.float32
if dtype == jnp.float64: jax.config.update("jax_enable_x64", True)

#%% pathing // custom code imports
import pathmagic as pathmagic
ProjectFolder, DataFolder, ResultFolder = pathmagic.magic()
import setup_src
from load_data import load

#%% load data
dt_original = 0.0005
freq_original = 1/dt_original

X_real, U_real = load(DataFolder)

# load model ## https://github.ugent.be/cevheck/HMT - contact cevheck.vanheck@ugent.be to discuss access
import HybTool_JAX as HMT
import src_JAX as src
import Data as DataClass

cf = setup_src.get_cf(model, HB_trainable=HB_trainable, m_trainable=m_trainable, V_trainable=V_trainable, include_friction_term=include_friction_term, nq=nq)
 
DataObj = DataClass.Data()
#%% subsample data
dt = 0.005
X_real_sampled = [X[::int(dt/dt_original),:] for X in X_real]
U_real_sampled = [U[::int(dt/dt_original),:] for U in U_real]

X = X_real_sampled.copy()
U = U_real_sampled.copy()

#%% create HybModel(s)
n, names, units = cf.config()

def make_model(dt=1/1000, P=None, n=None, f=None, g=None, cfg_eta=None, l2=None, **kwargs):
    hybMod = HMT.HybLayer(g=g, P=P, f=f, n=n, cfg_eta=cfg_eta, dt=dt, **kwargs)           
    loss = src.WeightedLoss([0, 1], l2=l2).mse_weighted
    hybMod.compile(**kwargs, loss_func=loss, custom_training=True)
    return hybMod

# options to iterate over
n_hidden_opt = [2, 4, 8, 16, 32, 64, 128]
n_layers_opt = [1,2]
activ_opt = ['mish', 'relu', 'tanh']

# little variance was spotted over the hyperparameters of the neural network. During paper taking fixed to the below
n_hidden_opt = [32]
n_layers_opt = [1]
activ_opt = ['mish']

# options for H and beta
H_select = np.unique([U[i][0,2] for i in range(len(U))])
beta_select = np.unique([U[i][0,3] for i in range(len(U))])
    
for n_hidden in n_hidden_opt:
    for n_layers in n_layers_opt:
        for activ in activ_opt:
            cfg_eta = {'n_hidden': [n_hidden]*n_layers,  # number of neurons per layer --> e.g: 2 layers = [16,32]
            'activation_hid': [activ]*n_layers,  # activation in each layer    --> e.g: 2 layers = ['tanh','relu']
            'activation_out': 'linear',  # activation of output layer
            }
                
            ##%% save pathing
            resultFolder = os.path.join(ProjectFolder, "results")
            if os.path.exists(resultFolder) == False: os.mkdir(ModelSave)
            ModelSave = setup_src.getmodelpath(model, Kalman_method, resultFolder, include_friction_term, nq)                

            if os.path.exists(ModelSave) == False: os.mkdir(ModelSave)
            if model != 'physics':
                layer_folder = os.path.join(ModelSave, f"layers_{n_layers}")
                if os.path.exists(layer_folder) == False: os.mkdir(layer_folder)
                activ_folder = os.path.join(layer_folder, f"activ_{activ}")
                if os.path.exists(activ_folder) == False: os.mkdir(activ_folder)
                neuron_folder = os.path.join(activ_folder, f"neurons_{n_hidden}")
                if os.path.exists(neuron_folder) == False: os.mkdir(neuron_folder)
                FinalModelSave = neuron_folder
            else:
                FinalModelSave = ModelSave
                                     
            assert(len(H_select) == 4)      # --> ensure dataset has full 4 options for H
            assert(len(beta_select) == 4)   # "" for beta
            savepath = os.path.join(FinalModelSave, f"hybMod")                            
            if save:
                if os.path.exists(f"{savepath}.pkl") and not resave: 
                    print(f"Model already exists at {savepath}, continuing")
                    continue
                else:
                    print(f"Creating new model at {savepath}")
            else:
                print(f"Not saving model, hence just plotting")

            ## physics
            p_train = cf.p_initial()

            X_train = X
            U_train = U
            H_train = [U[i][:,2:3] for i in range(len(X))]   # get ground truth H-value   // 1:2 to make sure its a txn array instead of txNone
            beta_train = [U[i][:,3:4] for i in range(len(X))]   # get ground truth H-value   // 1:2 to make sure its a txn array instead of txNone

            if (HB_trainable): U_train = [np.delete(U_traini, [2,3], axis=1) for U_traini in U_train]       # delete H in input array -- want it in the p-vector
            if (m_trainable): U_train = [np.delete(U_traini, [1], axis=1) for U_traini in U_train]       # delete H in input array -- want it in the p-vector
            if (V_trainable): U_train = [np.delete(U_traini, [0], axis=1) for U_traini in U_train]       # delete H in input array -- want it in the p-vector
            P = copy.deepcopy(p_train)

            if model != 'physics':
                # get statistics (mean, std) on q for standardscaler in NeuralNetwork
                P_init = {key: value[0] if isinstance(value, list) else value for key, value in P.items()}
                x_all, u_all = DataObj.q_input_format(X=X_train, U=U_train, n=n)
                q = cf.g(x_all, u_all,p=P_init)
                
                scaler = 'StandardScaler'
                q_mean = np.mean(q, axis=0)
                q_std = np.std(q, axis=0)
                cfg_eta['scaler'] = {"type": 'StandardScaler', 'mean': q_mean, 'std': q_std}
            
            lr=0.0001   # learning rate 
            l2 = None   # l2 regularization
            
            seperated_P = {key:val for key,val in P.items() if key in seperated_vars}
            groups_train = jnp.arange(len(X_train)) # each trajectory is a seperate group
            shuffle = True if seperated_P else False  

            # Hybrid model ## https://github.ugent.be/cevheck/HMT - contact cevheck.vanheck@ugent.be to discuss access
            hybMod = make_model(lr=lr, dt=dt, P=P, n=n, f=cf.f, g=cf.g, cfg_eta=cfg_eta,seperated_P=seperated_P, l2=l2)
            
            ##%% (pre)training
            epochs = 2000
            batch_size = 1024
            if pretrain:
                trainProcess = hybMod.train(X_train, U_train, epochs=epochs, batch_size=batch_size, groups_train=groups_train, verbose=verb, ES=None, shuffle=shuffle, batch_shuffle=False, validation_split=0.3)  # train
                
            if testing:
                # -- testing data -- #
                X0 = [x[0] for x in X_train]  # load initial states of each trajectory
                X_sim, Z_sim = hybMod.simulation(X0, U_train,verbose=verb, groups=groups_train, modulo=[2*np.pi, 0], return_z=True)  # make simulation

                MSE_test_trajectory = np.mean([np.mean(np.square(xsim[:1000,1] - X_t[:1000,1])) for xsim, X_t in zip(X_sim, X_train)])
                print("MSE test trajectory: ", MSE_test_trajectory)

                if show_x:
                    ## plot simulations -- testing data
                    for j in range(min(len(X_sim), 20)):
                        plt.plot(X_sim[j][:, 1])
                        plt.plot(X_train[j][:, 1], '--k')
                    plt.xlabel('samples')
                    plt.title("Simulation results")
                    plt.show()

                plt.close()
            ##%% save (pretrained) models
            if save:
                hybMod.save(savepath)
                print(f"Saved model at path {savepath}")

# %%
