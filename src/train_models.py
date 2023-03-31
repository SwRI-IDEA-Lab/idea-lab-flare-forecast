# script for running multiple experiments
import yaml
import train_model

with open('experiment_config.yml') as config_file:
    config = yaml.safe_load(config_file.read())

for i in range(1):
    config['data']['val_split'] = i
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)
    
    train_model.main()