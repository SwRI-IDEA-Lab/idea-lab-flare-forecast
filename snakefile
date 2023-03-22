configfile: "snakemake_config.yaml"

rule download:
    output:
        path = directory(expand("{savedir}/{i}",savedir=config['save_dir'],i=config['instrument']))
    shell:
        "python src/downloader.py --email={config[email]} -sd={config[start_date]} -ed={config[end_date]} -wl={config[wavelength]} -i={config[instrument]} -c={config[cadence]} -f={config[format]} -p={output.path} -dlim={config[dlim]}"

rule indexandclean:
    input:
        expand("{savedir}/{i}",savedir=config['save_dir'],i=config['instrument'])
    output:
        expand("Data/index_{i}.csv",i=config['instrument'])
    shell:
        "python src/data_preprocessing/index_clean_magnetograms.py {config[instrument]} -r {config[save_dir]} -n {config[newdir]}"

# rule label:
#     input:
#         "Data/index_{instrument}.csv"
#     output:
#         "Data/labels_{instrument}.csv"
#     shell:
#         expand("python src/data_preprocessing/label_dataset.py {{input}} {{output}} -w={windows}",windows=config['forecast_windows'])