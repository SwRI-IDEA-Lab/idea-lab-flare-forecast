# idea-lab-flare-forecast
Flare forecasting code

To set up the environment using conda, run the following:  

    conda env create -f environment_[os].yml  
where [os] is your operating system, either windows or linux.  

To run the whole pipeline including data downloading, processing, and training the model, you can use [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html). Edit the snakemake_config.yaml file and the experiment_config.yml to customize the pipeline, then run:  

    snakefile --cores 1



