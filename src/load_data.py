"""
All functionalities related to loading in the data
"""
import numpy as np
import math
import copy
from itertools import product

def load(DataFolder):
    X_real = []
    U_real = []
    X_save = np.load(f'{DataFolder}/X.npz')
    U_save = np.load(f'{DataFolder}/U.npz')
    for key in X_save.keys():
        X_real.append(X_save[key])
        U_real.append(U_save[key])
    return X_real, U_real

def get_data(MC, X, U, X_real_sampled, V_test, total_samples):
    """
    for a single V settings (single simulation), repeat the simulation to obtain #total samples and return in a concatenated array
        Ensures that each trajectory is 60s long, e.g. a trajectory of 14s is repeated 5 times and cut off
        For the repetitions a best point of repetition is found by minimizing the distance in the state space between end and start of the concatenated parts
    """
    assert(MC.multipletest)
    multiple_testX = [X[i] for i in range(len(X)) if np.allclose(U[i][0, 0], V_test)]                          # noisy measurements
    multiple_testU = [U[i] for i in range(len(X)) if np.allclose(U[i][0, 0], V_test)]                          # inputs
    multiple_realX = [X_real_sampled[i] for i in range(len(X)) if np.allclose(U[i][0, 0], V_test)]             # real groundtruth
    if len(multiple_testX) != 1:            # start with single traj
        assert(len(multiple_testX) == 0)    # else; should be None found     
        return None
    
    ## find best point of concatenation
    dist_in_statespace = np.abs(multiple_testX[0] - multiple_testX[0][0])
    mean_speed = np.mean(multiple_testX[0][:,1])
    one_circle_seconds = 2*np.pi/mean_speed
    one_circle_samples = int(one_circle_seconds/MC.dt)
    closest_idx_in_last_circle = np.argmin(dist_in_statespace[-one_circle_samples:,0])      # argmin in theta direction on last cycle
    closest_idx_from_end = one_circle_samples - closest_idx_in_last_circle
    
    ## concatenation array that flows smoothly in the next one
    multiple_testX = [multiple_testX[0][:-closest_idx_from_end]]
    multiple_testU = [multiple_testU[0][:-closest_idx_from_end]]
    multiple_realX = [multiple_realX[0][:-closest_idx_from_end]]
    total_len = len(multiple_testX[0])

    repeats = math.ceil(total_samples/total_len)
    multiple_testX = [multiple_testX[0] for _ in range(repeats)]
    multiple_testU = [multiple_testU[0] for _ in range(repeats)]
    multiple_realX = [multiple_realX[0] for _ in range(repeats)]
    
    testX = np.concatenate(multiple_testX)[:total_samples]
    testU = np.concatenate(multiple_testU)[:total_samples]
    realX = np.concatenate(multiple_realX)[:total_samples]        
    return testX, testU, realX

def get_filter_data(testX, testU, realX, total_samples, dt):
    """
    convert to history, filter and future parts to be used in the filter
    """
    # filter_start = int(1/dt)              # start filtering after 1s
    filter_start = int(0/dt)                # filter from start onwards
    history_x = testX[:filter_start]        # history part (=pre-filter)
    history_u = testU[:filter_start]
    filter_end = total_samples              # end of filtering stage

    filter_x = testX[filter_start:filter_end]          # filtering part
    filter_u = testU[filter_start:filter_end]
    filter_xreal = realX[filter_start:filter_end]           

    end = testX.shape[0] * dt   # amount of samples * time/sample = total time
    total_end = int(end/dt)
    T = np.linspace(0, end, total_end)

    future_x=testX[filter_end:]                   # future part (=post-filter)
    future_u=testU[filter_end:]

    return history_x, history_u, filter_x, filter_u, filter_xreal, future_x, future_u, T

def sample_data(N, camheights, cambetas, cammasses, camvoltages, X, U, X_real_sampled, MC):
    """
    Sample N different data sets from the given data (for tuning purposes)
    """
    HB_combinations = list(product(camheights, cambetas))

    occurance_per_mass = int(N/len(cammasses))
    occurance_per_V = int(N/len(camvoltages))

    sampled_masses = []
    sampled_voltages = []
    while len(sampled_masses) < 2*N:    # just take too much because there can still fall out
        all_masses = copy.deepcopy(cammasses) 
        np.random.shuffle(all_masses)
        sampled_masses.extend(all_masses)
    while len(sampled_voltages) < 2*N:
        all_voltage = copy.deepcopy(camvoltages) 
        np.random.shuffle(all_voltage)
        sampled_voltages.extend(all_voltage)
        
    total_samples = 100000
    resultingX_sets = []
    resultingU_sets = []
    resultingrealX_sets = []
    for HBcombo, cammass_test, camvoltage_test in zip(HB_combinations, sampled_masses, sampled_voltages):
        if len(resultingX_sets) >= N: break
        camheight_test, cambeta_test = HBcombo
        testX_set = [X[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]                          # noisy measurements
        testU_set = [U[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]                          # noisy measurements
        realX_set = [X_real_sampled[i] for i in range(len(X)) if (U[i][0, 2] == camheight_test and np.allclose(U[i][0,3], cambeta_test) and cammass_test == U[i][0,1])]             # real groundtruth
        
        try:
            testX, testU, realX = get_data(MC, testX_set, testU_set, realX_set, camvoltage_test, total_samples)
        except Exception as e:
            Vs = [testU_set_i[0,0] for testU_set_i in testU_set]
            assert(camvoltage_test not in Vs)
            continue
        resultingX_sets.append(testX)
        resultingU_sets.append(testU)
        resultingrealX_sets.append(realX)
    return resultingX_sets, resultingU_sets, resultingrealX_sets, HB_combinations