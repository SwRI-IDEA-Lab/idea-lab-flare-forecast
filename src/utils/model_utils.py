from src.model_classifier import LitConvNet
from model_regressor import LitConvNetRegressor
import pandas as pd
from pathlib import Path
import numpy as np
import os
import glob
from utils.analysis_helper import print_metrics, print_regression_metrics
import wandb

def load_model(run,ckpt_path,model,litclass=LitConvNet,strict=True,download:bool=False):
    """
    Load model into wandb run by downloading and initializing weights

    Parameters:forecasting
        run:        wandb run object
        ckpt_path:  wandb path to download model checkpoint from
        model:      model instance
        litclass:   Lightning model class (must have a load_from_checkpoint function)
        download:   flag to force load to always download ckpt from wandb
    Returns:
        classifier: litclass object with loaded weights
    """
    #TODO: check local dir for model checkpoint if already downloaded
    print('Loading model checkpoint from ', ckpt_path)
    # check if already in local directory
    local_ckpt_path = glob.glob('artifacts/'+ckpt_path.split('/')[-1].strip(':best_k')+'*')
    if (len(local_ckpt_path)>0) and not download:
        artifact_dir = local_ckpt_path[0]
    else:
        artifact = run.use_artifact(ckpt_path,type='model')
        artifact_dir = artifact.download()
    classifier = litclass.load_from_checkpoint(Path(artifact_dir)/'model.ckpt',model=model,strict=strict)
    return classifier

def save_preds(preds,dir,fname,regression:bool=False):
    """
    Saves model predictions locally and to wandb run

    Parameters:
        preds:      list of model outputs
        dir:        local directory for saving
        fname:      filename to save as
        regression: if true then regression, else classification
    """
    file = []
    ytrue = []
    ypred = []
    for predbatch in preds:
        file.extend(predbatch[0])
        ytrue.extend(np.array(predbatch[1]).flatten())
        ypred.extend(np.array(predbatch[2]).flatten())
    if len(ytrue)>0 and not regression:  # no metrics if no data
        print_metrics(np.array(ypred)[:],np.array(ytrue)[:],True)
    elif len(ytrue)>0 and regression:  # no metrics if no data
        print_regression_metrics(np.array(ypred)[:],np.array(ytrue)[:],True)
    df = pd.DataFrame({'filename':file,'ytrue':ytrue,'ypred':ypred})
    df.to_csv(dir+os.sep+fname,index=False)
    wandb.save(fname)