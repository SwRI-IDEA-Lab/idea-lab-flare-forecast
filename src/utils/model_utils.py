from model import LitConvNet
import pandas as pd
from pathlib import Path
import numpy as np
import os
from utils.analysis_helper import print_metrics, print_regression_metrics
import wandb

def load_model(run,ckpt_path,model,strict=True):
    """
    Load model into wandb run by downloading and initializing weights

    Parameters:forecasting
        run:        wandb run object
        ckpt_path:  wandb path to download model checkpoint from
        model:      model class
    Returns:
        classifier: LitConvNet object with loaded weights
    """
    print('Loading model checkpoint from ', ckpt_path)
    artifact = run.use_artifact(ckpt_path,type='model')
    artifact_dir = artifact.download()
    classifier = LitConvNet.load_from_checkpoint(Path(artifact_dir)/'model.ckpt',model=model,strict=strict)
    return classifier

def save_preds(preds,dir,fname):
    """
    Saves model predictions locally and to wandb run

    Parameters:
        preds:      list of model outputs
        dir:        local directory for saving
        fname:      filename to save as
    """
    file = []
    ytrue = []
    ypred = []
    for predbatch in preds:
        file.extend(predbatch[0])
        ytrue.extend(np.array(predbatch[1]).flatten())
        ypred.extend(np.array(predbatch[2]).flatten())
    print_regression_metrics(np.array(ypred),np.array(ytrue),True)
    df = pd.DataFrame({'filename':file,'ytrue':ytrue,'ypred':ypred})
    df.to_csv(dir+os.sep+fname,index=False)
    wandb.save(fname)