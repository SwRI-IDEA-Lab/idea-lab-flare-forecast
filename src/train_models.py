# script for running multiple experiments
import yaml
import train_model
import os

with open('experiment_config.yml') as config_file:
    config = yaml.safe_load(config_file.read())

highfluxmodels = ['sklmachn','rx977pm1','kdzgw1to','4f3zjhse','huzu3dhw']
flarefluxmodels = []

for i in range(5):
    config['data']['val_split'] = i

    # first pretrain
    config['data']['label'] = 'high_flux'
    config['data']['augmentation'] = 'conservative'
    config['model']['checkpoint_location'] = None
    config['model']['load_checkpoint'] = False
    config['testing']['eval'] = False
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)
    
    train_model.main()

    # # obtain run id and run train
    last_run = sorted(os.listdir('wandb'))[-1]
    run_id = last_run.split('-')[-1]
    # run_id = highfluxmodels[i]
    config['data']['label'] = 'flare'
    config['data']['augmentation'] = 'conservative'
    config['model']['checkpoint_location'] = 'model-'+run_id+':best_k'
    config['model']['load_checkpoint'] = True
    config['testing']['eval'] = True
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)

    train_model.main()