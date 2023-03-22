configfile: "snakemake_config.yaml"


wildcard_constraints:
    instrument = "[MHS5mhs].*"

rule all:
    input: 
        expand("{savedir}/{instrument}",savedir=config['save_dir'],instrument=config['instrument_download']),
        expand("{savedir}/index_{instrument}.csv",savedir=config['save_dir'],instrument=config['instrument_process']),
        expand("{savedir}/labels_{instrument}.csv",savedir=config['save_dir'],instrument=config['instrument_process'])
        
rule download:
    output:
        path = directory("{savedir}/{instrument}/")
    shell:
        "python src/downloader.py --email={config[email]} -sd={config[start_date]} -ed={config[end_date]} -wl={config[wavelength]} -i={wildcards.instrument} -c={config[cadence]} -f={config[format]} -p={output.path} -dlim={config[dlim]}"

rule indexandclean:
    input:
        "{savedir}/{instrument}/"
    output:
        "{savedir}/index_{instrument}.csv"
    shell:
        "python src/data_preprocessing/index_clean_magnetograms.py {wildcards.instrument} -r {config[save_dir]} -n {config[newdir]}"

rule label:
    input:
        index_file = "{savedir}/index_{instrument}.csv"
    output:
        labels_file = "{savedir}/labels_{instrument}.csv"
    shell:
        "python src/data_preprocessing/label_dataset.py {input.index_file} {output.labels_file}"