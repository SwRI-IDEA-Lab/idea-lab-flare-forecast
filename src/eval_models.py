# script for running multiple experiments
import yaml
import eval_model

# Run IDs for pretraining experiment - 192 h
run_ids = ['0yp1u3u7','rqjod26l','ccibb0m3','l43yawpd','kb7e4vjm',  # trained from scratch
        'yhhzi3ol','ckit3ek0','i1riak8a','ju5soz52','cx3n24vr',  # pre-trained on high flux
        'rmvre2nt','myhjb5c6','jkk8ysjw','gud9szjr','nnchky6w']  # pre-trained on flare-flux
# Run IDs for pretraining experiment - 24 h
run_ids = ['rpbdaar7','7rly07du','dvbv95z9','1qun5v4m','5yjfaisu', # trained from scratch
           'on2nqevh','bf031zcs','dczvlcbk','mmi8972m','y7afx68p', # pre-trained on high flux
           'd8smbbgj','jz02x97l','3xpiyftj','16cgp7dm','zrnoue78'] # pre-trained on flare-flux
# Run IDs for gating on 192 h
run_ids = ['sy7z0o3u','ti0z3try','8kntgxyp','q7gp8js4','0at1qp1f', # 72 hr
           '2x7ie8b3','9r32ir8c','z12k2weh','yew48cj2','ajqi853m']

with open('experiment_config.yml') as config_file:
    config = yaml.safe_load(config_file.read())

for run_id in run_ids:
    config['meta']['id'] = run_id
    with open('experiment_config.yml','w') as config_file:
        yaml.dump(config,config_file)
    
    eval_model.main()