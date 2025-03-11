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