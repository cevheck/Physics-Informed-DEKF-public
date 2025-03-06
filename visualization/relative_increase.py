import numpy as np
### Calculate relative increase in parameter tracking accuracy ###
"""
print the end value of the mean parameter error of all simulations.
    - by running time_evolution.py with "print_end_mean" set to True

Do this for all values of H, beta, H for the tracking of H, beta, V.
Put them in single list (below) to calculate mean % increase
"""
print("Parameter tracking accuracy:")
print("-"*40)
wandb_for_bb = True
H_PDEKF_Qrel = 0.001

H_PDEKF = [[8.068885654211044e-05,0.0007581431418657303,0.00013982877135276794,0.0007329210638999939],    # H = [0.01, 0.03, 0.05, 0.07]
           [0.02510666847229004, 0.02947211265563965, 0.025105953216552734, 0.0017707347869873047],       # beta = [pi/2, 3pi/4, pi, 5pi/4]
           [0.03342147544026375, 0.07444275705154332, 0.12928450107574463, 0.08662940283953133]]           # V for H = [0.01, 0.03, 0.05, 0.07]

p_PDEKF = [[0.0026637474074959755,0.005003189668059349,0.006178852170705795,0.00023784488439559937],   # H = [0.01, 0.03, 0.05, 0.07]
           [0.21027398109436035, 0.22864985466003418, 0.31291699409484863, 0.26665449142456055],          # beta = [pi/2, 3pi/4, pi, 5pi/4]
           [0.4780348055064678, 0.0012771194264040156, 0.4399069408575694, 0.43782850039207327]]          # V for H = [0.01, 0.03, 0.05, 0.07]


n_PDEKF = [[0.02131533809006214,0.009132923558354378,0.004536937922239304,0.014761719852685928],      # H = [0.01, 0.03, 0.05, 0.07]
           [0.9391294717788696, 0.41899561882019043, 0.1715860366821289, 1.2108747959136963],              # beta = [pi/2, 3pi/4, pi, 5pi/4]
           [0.32373857994874317, 0.5539127204377773, 0.22299509048461913, 0.665478140620862]]           # V for H = [0.01, 0.03, 0.05, 0.07]

n_PDEKF_wandb_update = [[0.0009019793942570686,0.0017536487430334091,0.002405703067779541,0.00296146422624588],   # H = [0.01, 0.03, 0.05, 0.07]
           [0.015576839447021484, 0.019866466522216797, 0.07769989967346191, 0.08061099052429199],          # beta = [pi/2, 3pi/4, pi, 5pi/4]
           [0.11061001569032669, 0.14385357953734318, 0.1850546956062317, 0.1433006100735422]]          # V for H = [0.01, 0.03, 0.05, 0.07]

if wandb_for_bb:
    n_PDEKF = n_PDEKF_wandb_update
    
H_PDEKF_mean_per_var = np.mean(H_PDEKF, axis=1)
p_PDEKF_mean_per_var = np.mean(p_PDEKF, axis=1)
n_PDEKF_mean_per_var = np.mean(n_PDEKF, axis=1)

perc_H_p = [(p_PDEKF_mean_per_var[i] - H_PDEKF_mean_per_var[i])/p_PDEKF_mean_per_var[i] for i in range(len(H_PDEKF_mean_per_var))]
perc_H_n = [(n_PDEKF_mean_per_var[i] - H_PDEKF_mean_per_var[i])/n_PDEKF_mean_per_var[i] for i in range(len(H_PDEKF_mean_per_var))]

print(f"Mean % increase Hybrid vs physics: {np.mean(perc_H_p)}")
print(f"Mean % increase Hybrid vs blackbox: {np.mean(perc_H_n)}")

### REPEAT FOR MULTISTEP PREDICTION ACCURACY ###
print("Multistep prediction accuracy:")
print("-"*40)
"""
print the end value of the model prediction error of all simulations.
    - by running model_prediction_errorbars.py with "print_end_mean" set to True
"""
wandb_for_bb = True
H_PDEKF_Qrel = 0.001

H_PDEKF = [0.018780205936664903,0.22797912877668017,0.8675439905835602,11.446910166700283]    # H = [0.01, 0.03, 0.05, 0.07]

p_PDEKF = [0.025798496859473344,0.5960242103462114,3.9919800863610084,15.897366090342297]   # H = [0.01, 0.03, 0.05, 0.07]

n_PDEKF_wandb_update = [61.59491738553652,433.4694222561134,211.37143056844252,438.4297148910152]   # H = [0.01, 0.03, 0.05, 0.07]

if wandb_for_bb:
    n_PDEKF = n_PDEKF_wandb_update
else:
    raise NotImplementedError
    
H_PDEKF_mean_per_var = H_PDEKF
p_PDEKF_mean_per_var = p_PDEKF
n_PDEKF_mean_per_var = n_PDEKF_wandb_update

perc_H_p = [(p_PDEKF_mean_per_var[i] - H_PDEKF_mean_per_var[i])/p_PDEKF_mean_per_var[i] for i in range(len(H_PDEKF_mean_per_var))]
perc_H_n = [(n_PDEKF_mean_per_var[i] - H_PDEKF_mean_per_var[i])/n_PDEKF_mean_per_var[i] for i in range(len(H_PDEKF_mean_per_var))]

print(f"Mean % increase Hybrid vs physics: {np.mean(perc_H_p)}")
print(f"Mean % increase Hybrid vs blackbox: {np.mean(perc_H_n)}")
