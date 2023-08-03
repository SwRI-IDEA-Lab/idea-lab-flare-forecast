configfile: "snakemake_config.yaml"
configfile: "experiment_config.yml"

wildcard_constraints:
    instrument = "[MHS5mhsa].*"

rule all:
    input: 
        # expand("{savedir}/{instrument}",savedir=config['save_dir'],instrument=config['instrument_download']),
        # expand("{savedir}/index_{instrument}.csv",savedir=config['save_dir'],instrument=config['instrument_process']),
        # expand("{savedir}/labels_{instrument}.csv",savedir=config['save_dir'],instrument=config['instrument_process']),
        # "{sd}/index_all_smoothed.csv".format(sd=config['save_dir']),
        # "{sd}/labels_all_smoothed.csv".format(sd=config['save_dir']),
        config['data']['data_file']
    shell:
        "python src/train_models.py"

rule download:
    output:
        path = directory("{savedir}/{instrument}/")
    shell:
        "python src/downloader.py --email={config[email]} -sd={config[start_date]} -ed={config[end_date]} -wl={config[wavelength]} -i={wildcards.instrument} -c={config[cadence]} -f={config[format]} -p={output.path} -dlim={config[dlim]}"

rule indexandclean:
    wildcard_constraints:
        instrument = "[MHS5mhs].*"
    input:
        "{savedir}/{instrument}/"
    output:
        "{savedir}/index_{instrument}.csv"
    shell:
        "python src/data_preprocessing/index_clean_magnetograms.py {wildcards.instrument} -r {config[save_dir]} -n {config[newdir]} -w {config[nworkers]}"

rule mergeindices:
    input:
        ["{sd}/index_{instrument}.csv".format(sd=config['save_dir'],instrument=instrument) for instrument in config['instrument_process']]
    output:
        "{savedir}/index_all_smoothed.csv"
    run:
        from src.data_preprocessing.index_clean_magnetograms import merge_indices_by_date
        import pandas as pd
        df_merged = merge_indices_by_date(config['save_dir'],config['instrument_process'])
        df_merged.to_csv(output[0],index=False)

rule label:
    input:
        index_file = "{savedir}/index_{instrument}.csv"
    output:
        labels_file = "{savedir}/labels_{instrument}.csv"
    shell:
        "python src/data_preprocessing/label_dataset.py {input.index_file} {output.labels_file} -w {config[forecast_windows]}"

