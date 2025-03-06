The results can be recreated using the main scripts (mostly "main_loop.py"). The results are used in the folder visualization/, where they are converted to smaller pickle files, required for each plot. This smaller files are present in this repository and the results in this folder are therefore abundant for plotting purposes and only required when going through the whole result making procedure. The files provided in visualization/ are created based on the non-downsampled results that are also the ones that will be made by running the scripts yourself. This results in a ~240GB folder (~60GB after zipping) and is therefore not suited to push to Github, hence the downsampled version (factor 100) is provided here. When using this zipped file, ensure the file structure is as follows (e.g.) results/ensembles_experimental_results/... This ensures compatibility with usage in other files. 

Three main results are created

	- ensembles_***: (main result) Results of filtering 
	- predict_***: Results of predicting with the converged models obtained in the filtering procedures
	- models: pretrained models created by scripts/pretrain_models.py

