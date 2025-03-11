# Physics-Informed Ensembles of DEKF
Repository Accompanying the paper: "Physics-Informed Ensembles of Dual Extended Kalman Filters for Online State and Parameter Estimation of Mechatronic Systems"
DOI: https://doi.org/10.1016/j.ymssp.2025.112469

# Paper "Physics-Informed Ensembles of Dual Extended Kalman Filters for Online State and Parameter Estimation of Mechatronic Systems"

This README is an attempt to help people trying to use my code implementation of an Ensemble of Hybrid Dual Extended Kalman Filters (HDEKF). This repository summarizes the code utilized in the creation of the paper: "Physics-Informed Ensembles of Dual Extended Kalman Filters for Online State and Parameter Estimation of Mechatronic Systems".
DOI: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4979179. 

The paper can be found in the folder "Paper". Additionally seperate files can be used to create your own (Hybrid) Ensemble of Dual Extended Kalman Filters. The raw data and results (for visualization) can be obtained upon request (too high file size to share on Github). In this folders results/ and visualization/ we added a subsampled version thereof that allows to reproduce the figures in the paper.  
In what follows a quick overview is provided of the folder structure. This Github is sufficient to recreate all of the results. Steps for recreating the paper are summarized at the end of this README. In case of additional questions, feel free to contact me, Cedric Van Heck, personally or by mail: cevheck.vanheck@ugent.be // cedric.vh@live.be


# Hybrid Dual Extended Kalman Filter
![HDEKF](https://github.com/user-attachments/assets/4c297050-c668-49a7-a638-ca7faa0b2f05)

# Variant methods
## _Hybrid_ model (Neural Network Augmented Physics)
![HybridModel](https://github.com/user-attachments/assets/1641d4d2-5ec6-4fdb-8cc9-f07ef86cf952)

## _Neural_ model
![NeuralModel](https://github.com/user-attachments/assets/532c9dd1-7fa8-4942-859a-c9cc5368184f)

## _Physics_ model
![PhysicalModel](https://github.com/user-attachments/assets/e5555770-6a16-4245-a647-053e4793de27)

# All Folders

## data
Folder containing the state and control measurement data of the Cam Follower system:

	- X.npy: state (\theta, \omega)
	- U.npy: control (V, m, H, \beta)


## plots
### paper_plots
Contains all the plots used in the paper

### additional_plots
Contains additional plots, sometimes refered to within the paper but not depicted therein.
## scripts
Contains the main files that are used to create and save results of the filtering and prediction precedured.

	- CF_exp: Folder that contains the ODE models for the hybrid-, physical- and neural formulation.
	- call_E_DEKF.py: Creates/Loads and simulates ensemble DEKF on a single data-trajectory. Called by main.py and main_loop.py
	- main.py and main_loop.py: Main files to choose the simulations that are being made. Iterates over the data grid (Table 1) to create results on each trajectory. main_loop.py is an extension to addtionally loop over different values of the parameter Q_alpha.
	- model_evaluation.py: similar to main_loop.py, but loads the converged model from the corresponding trajectory and performs trajectory prediction (Fig 4b & Fig B9)
	- pathmagic.py: File to fix pathing from absolute to relative frame
	- predict_E_DEKF.py: Loads converged ensemble DEKF and predicts the MSE on a single data-trajectory. Called by model_evaluation.py
	- pretrain_models.py: Pretrains and saves an initial model to results/models
	- tuning_evaluation.py: For a given setting, evaluates the tuning metrics
	- tuning.py: Tuning loop, using WandB (https://wandb.ai/)

## src
Contains functions and classes utilized by the scripts in the "scrips/" folder.

	- adaptive_dekf_ensemble.py: The main filter class that initializes an ensemble of DEKF (init). And can perform filtering (simulate).
	- configuration.py: Class that is used to configurate basic settings in order to allow for identical usage over multiple scripts. Serves as a config file. 
	- configuration_WandB.py: Config class used when tuning (tuning.py)
	- ensemble.py: Functions related to plotting/saving/simulating the ensembles DEKF.
	- load_data.py: Functions related to data handling
	- setup_src.py: Additional functions used for setting up the filtering problem (pathing/conversion to correct matrix format/correct function loading dependent on hybrid/physics/neural, ...)
	
## visualization
Contains all files used for creating the Figures and more. General way of working is converting the results into specific form for the plot. The data after conversion can be saved (save=True) and later immediately loaded (load=True) for speed up of the plot creation. We also created the pickle files (by running with save=True), creating the folder and results at location visualization/plots & visualization/plots_wandb1. The resulting files were however too big to push to Github, therefore we provide a subsampled version (factor 100) of the results, which still works with all of the code in the visualization folder. The complete (non-subsampled) data is available on request. With these files (or after running yourself, following the step-by-step plan below under "paper recreation"), running the code with load=True should recreate the plots. In order to use the provided files, the folders (and all subfolders) should first be unzipped.

	- time_evolution.py: Fig 4a and Fig 5
	- time_evolution_different_criteria.py: Fig 6
	- model_prediction_errorbars.py: Fig. 4b
	- model_prediction_errorbars_ifo_n.py: Fig. B9
	- pareto_combined.py: Fig 7
	- pathmagic.py: File to fix pathing
	- plot_style.txt: File used for cleaner plots (imported in util.py, except for creation of Fig 5)
	- plot_util.py: Contains all of the functions related to plotting and handling of the pickle result files. Functionalities will be called upon from other files.
	- relative_increase.py: file where the % increases of the models are calculated

## results
The results can be recreated using the main scripts (mostly "main_loop.py"). The results are used in the folder visualization/, where they are converted to smaller pickle files, required for each plot. These smaller files are present in this repository and the results in this folder are therefore abundant for plotting purposes and only required when going through the whole result making procedure. The files provided in visualization/ are created based on the non-downsampled results that are also the ones that will be made by running the scripts yourself. This results in a ~240GB folder (~60GB after zipping) and is therefore not suited to push to Github, hence the downsampled version (factor 100) is provided here. When using this zipped file, ensure the file structure is as follows (e.g.) results/ensembles_experimental_results/... This ensures compatibility with usage in other files. 

Three main results are created
	- ensembles_***: (main result) Results of filtering 
	- predict_***: Results of predicting with the converged models obtained in the filtering procedures
	- models: pretrained models created by scripts/pretrain_models.py

# Paper recreation
In order to recreate the paper one needs to go through a handful of steps. These will be listed here.

0. (optional) Redo the parameter tuning, by running scripts/tuning.py, the tuning grid is defined in src/configuration_WandB.py
1. Create pretrained models by running scripts/pretrain_models.py for each of the models once and with save=True.
2. Run scripts/main_loop.py with all models (hybrid, bb, physics) = True and save_now = True. Hyperparameters are set in src/configuration.py. The result files in result/ensembles_*** should be created.
3. Run visualization/time_evolution.py with save=True (and load=False), "Q_NN_added_rel_opts = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]" and "plot_whats = ['H', 'V', 'beta', 'Qx', 'eps']". This converts all of the result files into the plotting pickle files than can now be loaded with the variable load=True. As mentioned before, we provide these results already in the github repository to allow for recreating the plots without having to rerun all of the experiments. 
4. Run visualization/pareto.py to evaluate and choose a trade-off on the hyperparamater Q_{alpha} (Fig 7). 
5. Run visualization/time_evolution.py and visualization/time_evolution_different_criteria.py for the desired trade-off values and the parameter which you want to plot. This creates Figures 4a and 5.
6. Run visualization/time_evolution_different_criteria.py to do the same for different criteria, creating Fig 6.
7. Model predictions (with the converged models) can be performed by running scripts/model_evaluation.py. Result files are created in results/predict_***.
8. Run visualization/model_prediction_evaluation.py to create Fig 4b.
9. Run time_evolution.py with "print_end_mean" = True to print the mean end error for all physical parameters, given a certain model as filter. The results of this are hardcoded in the file "relative_increase.py" and used to calculate percentage increases in performance.
