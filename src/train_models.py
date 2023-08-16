# script for running multiple experiments
import yaml
from utils.analysis_helper import *
import train_model
import os

with open('experiment_config.yml') as config_file:
    config = yaml.safe_load(config_file.read())

highfluxmodels = ['sklmachn','rx977pm1','kdzgw1to','4f3zjhse','huzu3dhw']
pseudotest_classifier_models = ['uyo4qwyp','qazuyg25','7yiwewej','7ns308th','avh2pkc1']
run_ids = []
windows = [1,6,12,24]
val_splits = 5

for i in range(val_splits):
    config['data']['val_split'] = i

    # first pretrain
    # config['data']['label'] = 'high_flux'
    # config['data']['augmentation'] = 'conservative'
    # config['model']['checkpoint_location'] = None
    # config['model']['load_checkpoint'] = False
    # config['testing']['eval'] = False
    # with open('experiment_config.yml','w') as config_file:
    #     yaml.dump(config,config_file)
    
    # train_model.main()

    # # obtain run id and run train
    # last_run = sorted(os.listdir('wandb'))[-1]
    # run_id = last_run.split('-')[-1]
    run_id = pseudotest_classifier_models[i]
    config['data']['label'] = 'flare'
    config['data']['augmentation'] = 'conservative'
    config['model']['checkpoint_location'] = 'kierav/flare-forecast/model-'+run_id+':best_k'
    config['model']['load_checkpoint'] = True
    config['testing']['eval'] = True
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)

    train_model.main()

    # save run id
    run_ids.append(sorted(os.listdir('wandb'))[-1].split('-')[-1])


# generate csv with metrics and performance plot for classification
if not os.path.exists(config['testing']['savedir']):
    os.makedirs(config['testing']['savedir'])

if not config['data']['regression'] and config['testing']['eval']:
    df,df_trainval = create_ensemble_df(run_ids,'',
                        config['testing']['savedir']+'/metrics_'+config['testing']['savefile'],
                        rootdir='',pseudotest=False)
    sns.set_theme(context='paper',style='whitegrid')
    plot_performance(df,nbins=8)
    plt.savefig(config['testing']['savedir']+'/'+config['testing']['savefile']+'_performance.png',dpi=300)