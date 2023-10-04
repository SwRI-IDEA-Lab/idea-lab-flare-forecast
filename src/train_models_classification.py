# script for running multiple experiments
import yaml
from utils.analysis_helper import *
import src.train_model_classification as train_model_classification
import os

with open('experiment_config.yml') as config_file:
    config = yaml.safe_load(config_file.read())

highfluxmodels = ['sklmachn','rx977pm1','kdzgw1to','4f3zjhse','huzu3dhw']
run_ids = []
val_splits = 5

for i in range(val_splits):
    config['data']['val_split'] = i

    # first pretrain
    config['data']['label'] = 'high_flux'
    config['model']['checkpoint_location'] = None
    config['model']['load_checkpoint'] = False
    config['testing']['eval'] = False
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)
    train_model_classification.main()

    # # obtain run id and run train
    last_run = sorted(os.listdir('wandb'))[-1]
    run_id = last_run.split('-')[-1]
    # run_id = highfluxmodels[i]

    config['data']['label'] = 'flare'
    config['model']['checkpoint_location'] = config['meta']['user']+'/'+config['meta']['project']+'/model-'+run_id+':best_k'
    config['model']['load_checkpoint'] = True
    config['testing']['eval'] = True
    
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)

    train_model_classification.main()

    # save run id
    run_ids.append(sorted(os.listdir('wandb'))[-1].split('-')[-1])


# generate csv with metrics and performance plot for classification
if not os.path.exists(config['testing']['savedir']):
    os.makedirs(config['testing']['savedir'])

if config['testing']['eval']:
    df,df_trainval = create_ensemble_df(run_ids,'',
                        config['testing']['savedir']+'/metrics_'+config['testing']['savefile'],
                        rootdir='',pseudotest=False)
    sns.set_theme(context='paper',style='whitegrid')
    plot_performance(df,nbins=8)
    plt.savefig(config['testing']['savedir']+'/'+config['testing']['savefile']+'_performance.png',dpi=300)
