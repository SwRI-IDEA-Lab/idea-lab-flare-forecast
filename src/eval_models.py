# script for running multiple experiments
import yaml
import eval_model

# Run IDs for pretraining experiment
run_ids = ['0yp1u3u7','rqjod26l','ccibb0m3','l43yawpd','kb7e4vjm',  # trained from scratch
        'yhhzi3ol','ckit3ek0','i1riak8a','ju5soz52','cx3n24vr',  # pre-trained on high flux
        'rmvre2nt','myhjb5c6','jkk8ysjw','gud9szjr','nnchky6w']  # pre-trained on flare-flux

with open('experiment_config.yml') as config_file:
    config = yaml.safe_load(config_file.read())

for run_id in run_ids:
    config['meta']['id'] = run_id
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)
    
    eval_model.main()