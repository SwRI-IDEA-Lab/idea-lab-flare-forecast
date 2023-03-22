configfile: "snakemake_config.yaml"


rule all:
    input: 
        expand("{savedir}/{instrument}",savedir=config['save_dir'],instrument=config['instrument_download']),
#         expand("{s}/index_{i}.csv",s=config['save_dir'],i=config['instrument_download']),
#         expand("{s}/labels_{i}.csv",s=config['save_dir'],i=config['instrument_download'])
        
rule download:
    output:
        path = directory("{savedir}/{instrument}")
    shell:
        "python src/downloader.py --email={config[email]} -sd={config[start_date]} -ed={config[end_date]} -wl={config[wavelength]} -i={wildcards.instrument} -c={config[cadence]} -f={config[format]} -p={output.path} -dlim={config[dlim]}"

# rule indexandclean:
#     input:
#         rules.download.output
#     output:
#         expand("{s}/index_{i}.csv",s=config['save_dir'],i=config['instrument_download'])
#     shell:
#         "python src/data_preprocessing/index_clean_magnetograms.py {config[instrument]} -r {config[save_dir]} -n {config[newdir]}"

# rule label:
#     input:
#         index_file = rules.indexandclean.output
#     output:
#         labels_file = expand("{s}/labels_{i}.csv",s=config['save_dir'],i=config['instrument_download'])
#     shell:
#         "python src/data_preprocessing/label_dataset.py {input.index_file} {output.labels_file}"