# idea-lab-flare-forecast


[![DOI](https://zenodo.org/badge/604852892.svg)](https://zenodo.org/badge/latestdoi/604852892)


Probabilistic flare forecasting on single time-frame historical magnetograms.

To set up the environment using conda, run the following:  

    conda env create -f environment_[os].yml  
where [os] is your operating system, either windows or linux.  

To run the whole pipeline including data downloading, processing, and training the model, you can use [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html). Edit the snakemake_config.yaml file and the experiment_config.yml to customize the pipeline, then run:  

    snakefile --cores n
where n is the number of cores.

The snakemake pipeline is able to download MDI and HMI data, however, the historical data currently needs to be downloaded separately from the Harvard Dataverse (https://dataverse.harvard.edu/dataverse/solarmagnetograms). 

It takes a long time to index and clean the data (around 6 hours per year of data for HMI, which takes the longest since it is the highest resolution). This can be reduced by using multiple cores (nworkers in the snakemake_config file). 

To just train the models, after data is downloaded and processed, modify the experiment_config.yml file and run the train_models_classification.py script. There is a flag in the config model for regression which should be set to false. To train a single model on a  8GB GeForce RTX 3070 Ti GPU takes about 1.5 hours. Using pretraining and the ensemble of 5 models will then take around 15 hours.

### Regression onto max X-ray Irradiance

To train the CNN ensemble to forecast the max X-ray irradiance instead of a probability of flaring requires some additional data preprocessing. It is recommended to first run the Snakemake pipeline and download historical magnetograms so the data is already indexed and cleaned, and there are trained classification models to use for pretraining the regression models.

First download the GOES data by running:

    python src/data_preprocessing/download_goes.py [goesdir]
where [goesdir] is the path to store the GOES data at.

Then label the data:

    python src/data_preprocessing/regression_labeller.py Data/index_all_smoothed.csv [out_file] [goesdir] -w [window]
with [out_file] the name of the labels file to be created, [goesdir] the path to the GOES data and [window] the desired forecast window(s) in hours.

Now modify the experiment_config.yml appropriately and run something like the train_models_regression.py script. There is a regression flag in the config file that should be set to true. You can set the test to either 'test_a','test_b' or an empty string to toggle between tests. To train on a zarr dataset set the use_zarr_dataset flag. 
