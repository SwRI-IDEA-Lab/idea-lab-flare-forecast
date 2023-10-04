# script for running multiple experiments
import yaml
from utils.analysis_helper import *
import train_model_regression
import os

with open('experiment_config.yml') as config_file:
    config = yaml.safe_load(config_file.read())

pseudotest_classifier_models = ['uyo4qwyp','qazuyg25','7yiwewej','7ns308th','avh2pkc1'] # wandb run ids of classifier models for pretrained weights
pseudotest_classifier_aiahmi_models = ['0d49hu0v','23u4f7nw','yquow5tx','ag90vus1','qkvaartc']
run_ids = []
val_splits = 5

for i in range(val_splits):
    config['data']['val_split'] = i

    if config['data']['use_zarr_dataset']:
        run_id = pseudotest_classifier_aiahmi_models[i]
    else:
        run_id = pseudotest_classifier_models[i]

    config['model']['checkpoint_location'] = config['meta']['user']+'/'+config['meta']['project']+'/model-'+run_id+':best_k'
    config['model']['load_checkpoint'] = True
    config['testing']['eval'] = True
    
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)

    train_model_regression.main()

    # save run id
    run_ids.append(sorted(os.listdir('wandb'))[-1].split('-')[-1])


# generate csv with metrics and performance plot for classification
if not os.path.exists(config['testing']['savedir']):
    os.makedirs(config['testing']['savedir'])

if config['testing']['eval']:
    df,df_trainval,_ = create_ensemble_df_regression(run_ids,
                        config['testing']['savedir']+'/metrics_'+config['testing']['savefile'],
                        rootdir='',pseudotest=False) # set pseudotest to True for evaluation on the hold-out, else set to False
