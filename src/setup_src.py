"""
All functionalities related to handling the ensemble DEKF
- pathing
- saving
- plotting
- wandb result handling
- simulate ensemble
- subselect ensemble based on selected metric
"""

import os
import copy
import numpy as np

import HybTool_JAX as HMT
import src_JAX as src
    
def get_ensemble_save_path(ResultFolder, MC, predict_eval=False):   
    EnsembleSave = os.path.join(ResultFolder, "ensembles_experimental_results")  

    if not(MC.HB_trainable):
        EnsembleSave += "_noHB"
    if not(MC.V_trainable):
        EnsembleSave += "_noV"
        
    if MC.nq == 3: 
        EnsembleSave = EnsembleSave.replace("ensembles", "ensembles_noHBV")
    elif MC.nq == 4: 
        EnsembleSave = EnsembleSave.replace("ensembles", "ensembles_noHB")
    else:
        if MC.nq != 6: raise Exception(f"MC.nq = {MC.nq} not known")
        
    if predict_eval: EnsembleSave = EnsembleSave.replace("ensembles", "predict_ensembles")
    return EnsembleSave

#%% HM setup
def make_model(dt=1/1000, P=None, n=None, f=None, g=None, cfg_eta=None, **kwargs):
        ## https://github.ugent.be/cevheck/HMT - contact cevheck.vanheck@ugent.be to discuss access
        hybMod = HMT.HybLayer(g, P, f, n, cfg_eta, dt=dt, **kwargs)     
        loss = src.WeightedLoss([0, 1]).mse_weighted                    
        hybMod.compile(**kwargs, loss_func=loss)
        return hybMod
    
def custom_imports(predict_eval=False):
    """
    Left out a lot of alternative options, that can be found in the previous folder dualKF_v2
        e.g. iteratedEKF, other CFmodels, adaptiveQ,W,R, 
    """
    from adaptive_dekf_ensemble import E_dekf as dekf  
    from ensemble import WandB_ensemble, make_new, log_WandB, plot_ensemble_results

    if not predict_eval:
        from call_E_DEKF import main
        from ensemble import save_ensemble as saver
    else:
        from predict_E_DEKF import main
        from ensemble import save_prediction as saver
    
    return make_new, dekf, main, plot_ensemble_results, WandB_ensemble, saver, log_WandB

def get_cf(model, filter_Tm=False, HB_trainable=False, m_trainable=False, V_trainable=False, include_friction_term=False, nq=None):
    CF_folder = 'CF_exp'
    import_file = 'CamFollower'
    
    if model == 'hybrid':
        pass
    elif model == 'physics':
        import_file += '_physical'
    elif model =='blackbox':
        import_file += '_blackbox'
    else:
        raise Exception(f'model {model} not known')
    
    ## perform import
    import_statement = f'from {CF_folder}.{import_file} import CamFollower as cf'
    exec_namespace = {}
    exec(import_statement, exec_namespace)
    cf = exec_namespace['cf']

    if model == 'hybrid' or model == 'blackbox':
        CamF = cf(HB_trainable, m_trainable, V_trainable, include_friction_term, nq=nq)
    elif model == 'physics':
        include_friction_term = True
        CamF = cf(HB_trainable, m_trainable, V_trainable, include_friction_term)
    return CamF

def getmodelpath(model, Kalman_method, resultFolder, include_friction_term, nq=None):
    ModelSave = os.path.join(resultFolder, "models")
    if model == 'hybrid':
        if Kalman_method == "PDEKF":
            path = os.path.join(ModelSave, "hybMod")     
        elif Kalman_method == 'DEKF':
            path = os.path.join(ModelSave, "no_pretrain_hybMod")                     
    elif model == 'physics':
        if Kalman_method == "PDEKF":
            path = os.path.join(ModelSave, "hybMod_p")                        
        elif Kalman_method == 'DEKF':
            path = os.path.join(ModelSave, "no_pretrain_hybMod_p")                        
    elif model == 'blackbox':
        if Kalman_method == "PDEKF":
            path = os.path.join(ModelSave, "hybMod_bb")                        
        elif Kalman_method == 'DEKF':
            path = os.path.join(ModelSave, "no_pretrain_hybMod_bb")    
    if include_friction_term: path += "_friction" 
    if model != 'physics':
        if nq == 3: 
            path = path.replace("hybMod", "hybMod_noHBV")
        elif nq == 4: 
            path = path.replace("hybMod", "hybMod_noHB")
        else:
            if nq != 6: raise Exception(f"nq = {nq} not known")
    return path

def update_uncertainties(MC):
    Qx1 = MC.Qx_init * MC.dt    # Qx0 on theta
    Qx2 = MC.Qx_init            # Qx0 on omega
    Qx = np.zeros(shape=(2, 2))
    np.fill_diagonal(Qx, [Qx1, Qx2])
    MC.Qx = Qx
                                
    weight_uncertainty = MC.weight_uncertainty                              # P0,w
    weight_uncertainty_added = MC.weight_uncertainty_added                  # Qw
    MC.NN_uncertainty = weight_uncertainty*MC.Q_NN_rel                      # P0,alpha
    MC.NN_uncertainty_added = weight_uncertainty_added*MC.Q_NN_added_rel    # Q_alpha
    
    if MC.HB_trainable:
        MC.H_uncertainty = weight_uncertainty*MC.QH_rel                             # P0,H
        MC.H_uncertainty_added = weight_uncertainty_added*MC.QH_added_rel           # QH
        MC.beta_uncertainty = weight_uncertainty*MC.Qbeta_rel                       # P0,beta
        MC.beta_uncertainty_added = weight_uncertainty_added*MC.Qbeta_added_rel     # Qbeta
    if MC.m_trainable:
        MC.m_uncertainty = weight_uncertainty*MC.Qm_rel                             # P0,m
        MC.m_uncertainty_added = weight_uncertainty_added*MC.Qm_added_rel           # Qm
    if MC.V_trainable:
        MC.V_uncertainty = weight_uncertainty*MC.QV_rel                             # P0,V
        MC.V_uncertainty_added = weight_uncertainty_added*MC.QV_added_rel           # QV       
        
    MC.Ra_uncertainty = weight_uncertainty*MC.QRa_rel                               # P0,Ra
    MC.Ra_uncertainty_added = weight_uncertainty_added*MC.QRa_added_rel             # QRa
    MC.D_uncertainty = weight_uncertainty*MC.QD_rel                                 # P0,D
    MC.D_uncertainty_added = weight_uncertainty_added*MC.QD_added_rel               # QD
    
def create_uncertainty_matrices(MC):
    Pw_ini = []
    Qw = []
    if MC.model == 'hybrid' or MC.model == 'blackbox':
        Pw_ini.extend([MC.NN_uncertainty])
        Qw.extend([MC.NN_uncertainty_added])
    MC.trainable_names = ['Ra', 'D']
    Pw_ini.extend([MC.Ra_uncertainty, MC.D_uncertainty])
    Qw.extend([MC.Ra_uncertainty_added, MC.D_uncertainty_added])
    if MC.HB_trainable: 
        Pw_ini.extend([MC.H_uncertainty, MC.beta_uncertainty])
        Qw.extend([MC.H_uncertainty_added, MC.beta_uncertainty_added])
        MC.trainable_names.extend(['H', 'beta'])
    if MC.m_trainable: 
        Pw_ini.extend([MC.m_uncertainty])
        Qw.extend([MC.m_uncertainty_added])
        MC.trainable_names.extend(['m'])
    if MC.V_trainable: 
        Pw_ini.extend([MC.V_uncertainty])
        Qw.extend([MC.V_uncertainty_added])
        MC.trainable_names.extend(['V'])
    MC.Pw_ini = Pw_ini
    MC.Qw = Qw