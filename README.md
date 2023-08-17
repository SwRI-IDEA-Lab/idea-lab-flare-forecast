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

To train a single model on a  8GB GeForce RTX 3070 Ti GPU takes about 1.5 hours. Using pretraining and the ensemble of 5 models will then take around 15 hours.



