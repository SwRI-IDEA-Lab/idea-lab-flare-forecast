""" Helper functions for plotting and analysis of model predictions """

import sys,os
sys.path.append(os.getcwd())

from datetime import datetime,timedelta 
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score,roc_auc_score,roc_curve,confusion_matrix,precision_recall_curve,r2_score
from src.probability_calibration import probability_calibration
import seaborn as sns
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a

# color-blind friendly palette https://davidmathlogic.com/colorblind/#%23004A98-%23C5C5C5-%23FF538C-%234DC6FF-%23FFA42C-%23C050FF-%2375FF84-%23C70000-%233D98C3-%235D5D5D
Clr = ['#004A98',
       '#C5C5C5',
       '#FF538C',
       '#4DC6FF',
       '#FFA42C',
       '#C050FF',
       '#75FF84',
       '#C70000'
       ]

def print_metrics(ypred,y,printresults=False):
    """
    Calculate metrics based on predictions. If there are no postive true
    labels, then will return NaNs
    
    Parameters:
        ypred (np array):       array of predicted probabilities
        ytrue (np array):       corresponding true outputs (1 - event, 0 - no event)
        printresults (bool):    flag to print out results or 
    
    Returns:
        results (list):         calculated metrics (MSE, BSS, APS, Gini, ECE, MCE, 
                                std, pos mean, pos_std, neg mean, neg_std, max output)
    """

    mse = (sum((ypred-y)**2))/len(ypred)
    std = np.std(ypred)
    pos_mean = np.mean(ypred[y==1])
    pos_std = np.std(ypred[y==1])
    neg_mean = np.mean(ypred[y==0])
    neg_std = np.std(ypred[y==0])
    max_output = np.max(ypred)

    if sum(y) == 0: # no positive data, return null results
        print('No positive events in dataset, returning NaNs for some results')
        bss = aps = gini = ece = mce = np.NaN
    else:
        bss = (sum((ypred-y)**2)-sum((sum(y)/len(y)-y)**2))/(-sum((sum(y)/len(y)-y)**2))
        aps = average_precision_score(y,ypred)
        gini = 2*roc_auc_score(y,ypred)-1
        ece,mce = reliability_diag(y,ypred,None,None,plot=False)
    
    results = [mse,bss,aps,gini,ece,mce,std,pos_mean,pos_std,neg_mean,neg_std,max_output]
    if printresults:
        print('MSE, BSS, APS, Gini, ECE, MCE, std, pos mean, pos_std, neg mean, neg_std, max output')
        print(results)
    return results

def print_regression_metrics(ypred,y,printresults=False):
    """
    Calculate metrics based on regression predictions
    
    Parameters:
        ypred (np array):       array of predicted outputs
        y (np array):           corresponding true outputs 
        printresults (bool):    flag to print out results or 
    
    Returns:
        results (list):         calculated metrics (MSE, BSS, APS, Gini, ECE, MCE, 
                                std, pos mean, pos_std, neg mean, neg_std, max output)
    """
    mse = (sum((ypred*6-y*6)**2))/len(ypred)
    mae = sum(abs(ypred*6-y*6))/len(ypred)
    r2 = r2_score(y*6,ypred*6)

    results = [mse,mae,r2]
    if printresults:
        print('MSE, MAE, R2')
        print(results)
    return results

def assemble_metrics(df,metricsfile,cal_label='yprob'):
    """
    Save metrics to csv for both uncalibrated and calibrated predictions for
    the model ensemble (5 members)

    Parameters:
        df (pd dataframe):      dataframe containing predictions (ypred) and true values (ytrue)
        metricsfile (str):      filename to save metrics to
        cal_label (str):        label for calibrated predictions (in case of using various calibrators)

    Returns:
        df_metrics (dataframe): metrics for uncalibrated predictions
        df_metrics_cal (dataframe):     metrics for calibrated predictions
    """
    metrics = []
    metrics_cal = []
    index = []
    
    for split in range(5):
        metrics.append(print_metrics(df['ypred'+str(split)],df['ytrue']))
        metrics_cal.append(print_metrics(df[cal_label+str(split)],df['ytrue']))
        index.append('Model '+str(split))
    metrics.append(print_metrics(df['ypred_median'],df['ytrue']))
    metrics_cal.append(print_metrics(df[cal_label+'_median'],df['ytrue']))
    index.append('Ensemble Median')

    df_metrics = pd.DataFrame(data=np.array(metrics),index=index,columns=['MSE','BSS','APS','Gini','ECE','MCE','std','pos mean','pos std','neg mean','neg std','max output'])
    df_metrics_cal = pd.DataFrame(data=np.array(metrics_cal),index=index,columns=['MSE','BSS','APS','Gini','ECE','MCE','std','pos mean','pos std','neg mean','neg std','max output'])
    df_metrics.to_csv(metricsfile+'.csv')
    df_metrics_cal.to_csv(metricsfile+'_cal.csv')

    return df_metrics,df_metrics_cal

def reliability_diag(ytrue,ypred,ax,label,nbins=10,plot=True,plot_hist=False,**kwargs):
    """
    Plots a reliability diagram (calibration curve) on a given axis and computes
    the calibration error metrics (expected calibration error and max calibration error)

    Parameters:
        ytrue (np array):       true values
        ypred (np array):       predicted values
        ax (axis object):       matplotlib axis for plotting
        label (str):            label for plot object
        nbins (int):            number of bins to compute the reliability diagram on
        plot (bool):            whether or not to plot
        plot_hist (bool):       whether or not to overlay a histogram of samples 
        **kwargs:               additional arguments for matplotlib plot function

    Returns:
        ece (float):            expected calibration error
        mce (float):            max calibration error
    """
    
    prob_true, prob_pred = calibration_curve(ytrue,ypred,n_bins=nbins)
    mce = max(abs(prob_true-prob_pred))

    if plot:
        ax.plot(prob_pred,prob_true,'.-',label=label,**kwargs)

    bin_edges = prob_pred[:-1] + np.diff(prob_pred)/2
    bin_edges = np.insert(bin_edges,0,0)
    bin_edges = np.append(bin_edges,1)
    ni = np.histogram(ypred,bins=bin_edges)[0]

    if plot_hist:
        sns.histplot(ypred,ax=ax,bins=bin_edges,alpha=0.6,stat='probability',label='_',**kwargs)

    ece = sum(ni*abs(prob_true-prob_pred))/sum(ni)

    return ece,mce

def calibrate_prob(run,nbinnings=20,pseudotest=True,rootdir='../'):
    """
    Loads model predictions and calibrates probabilites based on train/val dataset

    Parameters:
        run (str):          wandb run id
        nbinnings (int):    number of binnings for the probability calibration
        pseudotest (bool):  whether to load the pseudotest/holdout or the test set
        rootdir (str):      root dir to append when searching for wandb files
    Returns:
        df_test (dataframe): test data including filename, true, predicted and calibrated outputs
        df_trainval (dataframe):    trainval data
    """
    if pseudotest:
        test_file = sorted(glob.glob(rootdir+'wandb/*'+run+'/files/pseudotest_results.csv'))[-1]   #obtain last dir with matching id
    else:
        test_file = sorted(glob.glob(rootdir+'wandb/*'+run+'/files/test_results.csv'))[-1]   #obtain last dir with matching id
    trainval_file = sorted(glob.glob(rootdir+'wandb/*'+run+'/files/trainval_results.csv'))[-1]   #obtain last dir with matching id
    df_test = pd.read_csv(test_file)
    df_trainval = pd.read_csv(trainval_file)
    calibrator = probability_calibration(df_trainval['ypred'],df_trainval['ytrue'],df_test['ypred'])
    ypred_cal = calibrator.calibrateProbability(nbinnings)
    df_test['yprob'] = ypred_cal
    calibrator2 = probability_calibration(df_trainval['ypred'],df_trainval['ytrue'],df_trainval['ypred'])
    ypred_cal2 = calibrator2.calibrateProbability(nbinnings)
    df_trainval['yprob'] = ypred_cal2
    return df_test,df_trainval

def create_ensemble_df(run_ids,experiment,metricsfile,pseudotest=True,rootdir='../'):
    """
    Assembles a dataframe with all the ensemble member predictions
    Performs probability calibration so both calibrated (yprob) and uncalibrated
    (ypred) predictions are included. Saves metrics to file.

    Parameters:
        run_ids (list):     wandb run ids for the ensemble
        experiment (str):   descriptor of the experiment to save to dataframe
        metricsfile (str):  filename to save metrics to
        pseudotest (bool):  flag to assemble pseudotest or test results
        rootdir (str):      root dir to append when searching for wandb files

    Returns:
        df_ensemble (dataframe):    all ensemble predictions for pseudotest/test data
        df_trainval_ensemble (dataframe):   all ensemble predictions for train/val data
    """
    df_ensemble = pd.DataFrame()
    df_trainval_ensemble = pd.DataFrame()

    for run,i in zip(run_ids,range(len(run_ids))):
        df,df_trainval = calibrate_prob(run,rootdir=rootdir,pseudotest=pseudotest)
        df = df.rename(columns={'ypred':'ypred'+str(i),'yprob':'yprob'+str(i)})
        df_trainval = df_trainval.rename(columns={'ypred':'ypred'+str(i),'yprob':'yprob'+str(i)})
        if len(df_ensemble) == 0:
            df_ensemble = df
            df_trainval_ensemble = df_trainval
        else:
            df_ensemble = df_ensemble.merge(df,on=['filename','ytrue'])
            df_trainval_ensemble = df_trainval_ensemble.merge(df_trainval,on=['filename','ytrue'])
    
    df_ensemble['ypred_mean'] = df_ensemble.filter(regex='ypred[0-9]').mean(axis=1)
    df_ensemble['ypred_median'] = df_ensemble.filter(regex='ypred[0-9]').median(axis=1)
    df_ensemble['ypred_std'] = df_ensemble.filter(regex='ypred[0-9]').std(axis=1)
    df_ensemble['yprob_mean'] = df_ensemble.filter(regex='yprob[0-9]').mean(axis=1)
    df_ensemble['yprob_median'] = df_ensemble.filter(regex='yprob[0-9]').median(axis=1)
    df_ensemble['yprob_std'] = df_ensemble.filter(regex='yprob[0-9]').std(axis=1)
    df_ensemble['experiment'] = experiment

    assemble_metrics(df_ensemble,metricsfile,cal_label='yprob')

    return df_ensemble,df_trainval_ensemble

def create_ensemble_df_regression(run_ids,metricsfile,pseudotest:bool=True,rootdir:str='../'):
    """
    Assembles a dataframe with all the ensemble member regression predictions

    Parameters:
        run_ids (list):     wandb run ids for the ensemble
        metricsfile (str):  filename to save metrics to
        pseudotest (bool):  flag to assemble pseudotest or test results
        rootdir (str):      root dir to append when searching for wandb files

    Returns:
        df_ensemble (dataframe):    all ensemble predictions for pseudotest/test data
        df_trainval_ensemble (dataframe):   all ensemble predictions for train/val data
    """
    df_ensemble = pd.DataFrame()
    df_trainval_ensemble = pd.DataFrame()
    for j,run in enumerate(run_ids):
        trainval_file = sorted(glob.glob(rootdir+'wandb/*'+run+'/files/trainval_results.csv'))[-1]   #obtain last dir with matching id
        if pseudotest:
            test_file = sorted(glob.glob(rootdir+'wandb/*'+run+'/files/pseudotest_results.csv'))[-1]   #obtain last dir with matching id
        else:
            test_file = sorted(glob.glob(rootdir+'wandb/*'+run+'/files/test_results.csv'))[-1]   #obtain last dir with matching id
        df = pd.read_csv(test_file)
        df_trainval = pd.read_csv(trainval_file)
        df = df.rename(columns={'ypred':'ypred'+str(j)})
        df_trainval = df_trainval.rename(columns={'ypred':'ypred'+str(j)})
        if len(df_ensemble) == 0:
            df_ensemble = df
            df_trainval_ensemble = df_trainval
        else:
            df_ensemble = df_ensemble.merge(df,on=['filename','ytrue'])
            df_trainval_ensemble = df_trainval_ensemble.merge(df_trainval,on=['filename','ytrue'])
    
    df_ensemble['ypred_mean'] = df_ensemble.filter(regex='ypred[0-9]').mean(axis=1)
    df_ensemble['ypred_median'] = df_ensemble.filter(regex='ypred[0-9]').median(axis=1)
    df_ensemble['ypred_std'] = df_ensemble.filter(regex='ypred[0-9]').std(axis=1)

    metrics = []
    index = []
    for split in range(5):
        metrics.append(print_regression_metrics(df_ensemble['ypred'+str(split)],df_ensemble['ytrue']))
        index.append('Model '+str(split))
    metrics.append(print_regression_metrics(df_ensemble['ypred_median'],df_ensemble['ytrue']))
    index.append('Ensemble Median')
    metrics.append(print_regression_metrics(df_ensemble['ypred_mean'],df_ensemble['ytrue']))
    index.append('Ensemble Mean')

    df_metrics = pd.DataFrame(data=np.array(metrics),index=index,columns=['MSE','MAE','R2'])
    df_metrics.to_csv(metricsfile+'.csv')

    return df_ensemble,df_trainval_ensemble,df_metrics

def plot_reliability_diags(df,title):
    """
    Plot figure of reliability diagrams for an ensemble (5 members)
    
    Parameters:
        df (dataframe):     dataframe assembled from create_ensemble_df routine
        title (str):        title for plot
    """
    fig,ax = plt.subplots(1,2,figsize=(7,3.5))
    ax[0].plot([0,1],[0,1],'-k',linewidth=1,label='_')
    ax[1].plot([0,1],[0,1],'-k',linewidth=1,label='_')
    ax[0].plot([0,1],[sum(df['ytrue']/len(df)),sum(df['ytrue']/len(df))],'--k',linewidth=1,label='_')
    ax[1].plot([0,1],[sum(df['ytrue']/len(df)),sum(df['ytrue']/len(df))],'--k',linewidth=1,label='_')
    ax[0].plot([0,1],[sum(df['ytrue']/len(df))/2,sum(df['ytrue']/len(df))/2+0.5],'--k',linewidth=1,label='_')
    ax[1].plot([0,1],[sum(df['ytrue']/len(df))/2,sum(df['ytrue']/len(df))/2+0.5],'--k',linewidth=1,label='_')
    plt.suptitle(title)
    for model in range(5):
        reliability_diag(df['ytrue'],df['ypred'+str(model)],ax[0],label='Model '+str(model),nbins=5,color=Clr[model])
        reliability_diag(df['ytrue'],df['yprob'+str(model)],ax[1],label='Model '+str(model),nbins=5,color=Clr[model])
    reliability_diag(df['ytrue'],df['ypred_median'],ax[0],label='Median',nbins=5,color=Clr[5])
    reliability_diag(df['ytrue'],df['yprob_median'],ax[1],label='Median',nbins=5,color=Clr[5])
    ax[0].legend(loc='upper left')
    ax[0].set_title('Uncalibrated')
    ax[0].set_xlabel('Probability')
    ax[0].set_ylabel('Flare frequency')
    ax[1].legend(loc='upper left')
    ax[1].set_title('Calibrated')
    ax[1].set_xlabel('Probability')
    ax[1].set_ylabel('Flare frequency')
    plt.tight_layout()

def plot_pred_hist(df,datafortitle,cal_label='yprob_median',binwidth=0.05,kde=False):
    """
    Plots histogram of predictions for a model ensemble-median

    Parameters:
        df (dataframe):     dataframe assembled from create_ensemble_df routine
        datafortitle (str): string for appending to title
        cal_label (str):    string to specify label of calibrated probabilities
        binwidth (float):   width of bins
        kde (bool):         whether to plot the kernel density estimate 
    """
    fig,ax = plt.subplots(1,2,figsize=(6,3))
    sns.histplot(df[df['ytrue']==0],x='ypred_median',ax=ax[0],label='non-flaring',stat='count',binwidth=binwidth,binrange=(0,1),kde=kde)
    sns.histplot(df[df['ytrue']==1],x='ypred_median',ax=ax[0],label='flaring',stat='count',binwidth=binwidth,binrange=(0,1),kde=kde)    
    ax[0].set_title('Uncalibrated')
    ax[0].set_xlim([0,1])
    ax[0].set_xlabel('Output probability')
    ax[0].set_ylabel('Relative Frequency')
    sns.histplot(df[df['ytrue']==0],x=cal_label,ax=ax[1],label='non-flaring',stat='count',binwidth=binwidth,binrange=(0,1),kde=kde)
    sns.histplot(df[df['ytrue']==1],x=cal_label,ax=ax[1],label='flaring',stat='count',binwidth=binwidth,binrange=(0,1),kde=kde)    
    ax[1].set_title('Calibrated')
    ax[1].set_xlim([0,1])
    ax[1].set_xlabel('Output probability')
    ax[1].set_ylabel('Relative Frequency')
    ax[1].legend()
    plt.suptitle('Probabilities for '+datafortitle)
    plt.tight_layout()

def plot_performance(df,cal='yprob',nbins=5):
    """
    Plots a panel of figures illustrating model performance for an ensemble,
    reliability diagram, TPR vs FPR and precision vs. recall
    
    Parameters:
        df (dataframe):     dataframe assembled from create_ensemble_df routine
        cal (str):          label of the calibrated probabilities
        nbins (int):        number of bins for the reliability diagram
    """
    fpr,tpr,thresh = roc_curve(df['ytrue'],df[cal+'_median'])
    pr,re,thresh2 = precision_recall_curve(df['ytrue'],df[cal+'_median'])

    tss = []
    hss = [] 
    for t in np.arange(0,1,0.02):
        tn, fp, fn, tp = confusion_matrix(df['ytrue'], df[cal+'_median']>=t).ravel()
        tss.append((tp) / (tp + fn) - (fp) / (fp + tn))
        hss.append(2*(tp*tn-fp*fn)/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn)))
    
    fig,ax = plt.subplots(1,3,figsize=(9,3))

    ax[0].plot([0,1],[0,1],'-k',linewidth=1,label='_')
    ax[0].plot([0,1],[sum(df['ytrue']/len(df)),sum(df['ytrue']/len(df))],'--k',linewidth=1,label='_')
    ax[0].plot([0,1],[sum(df['ytrue']/len(df))/2,sum(df['ytrue']/len(df))/2+0.5],'--k',linewidth=1,label='_')
    for model in range(5):
        reliability_diag(df['ytrue'],df[cal+str(model)],ax[0],label='_',nbins=nbins,color=Clr[model])
    reliability_diag(df['ytrue'],df[cal+'_median'],ax[0],label='Median',nbins=nbins,color=Clr[5],marker='*',markersize=8)
    sns.histplot(df[cal+'_median'],ax=ax[0],stat='probability',binwidth=0.1,binrange=(0,1),color=Clr[5],alpha=0.5)
    ax[0].set_xlabel('Probability')
    ax[0].set_ylabel('Flare frequency')
    ax[0].legend()
    ax[0].set_ylim([-0.05,1.05])
    ax[0].set_xlim([-0.05,1.05])

    ax[1].plot(fpr,fpr,'--k',linewidth=1)
    ax[1].plot(fpr,tpr,'-')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_ylim([-0.05,1.05])
    ax[1].set_xlim([-0.05,1.05])

    ax[2].plot([0,1],[sum(df['ytrue']/len(df)),sum(df['ytrue']/len(df))],'--k',linewidth=1,label='_')
    ax[2].plot(re,pr)
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_ylim([-0.05,1.05])
    ax[2].set_xlim([-0.05,1.05])

    plt.tight_layout()

def plot_timeseries(df,tstart,tend,goes=False,cal='yprob'):
    """
    Plot a timeseries of model predictions and the GOES xray flux
    
    Parameters:
        df (dataframe):     dataframe assembled from create_ensemble_df routine
        tstart (datetime):  start time
        tend (datetime):    end time
        goes (bool):        whether to download the GOES timeseries data and plot it
        cal (str):          label for calibrated probabilities
    """
    fig,ax = plt.subplots(figsize=(9,2.5))
    if goes:
        result = Fido.search(a.Time(datetime.strftime(tstart,'%Y-%m-%d %H:%M'),datetime.strftime(tend,'%Y-%m-%d %H:%M')),a.Instrument('XRS'))
        file_goes = Fido.fetch(result)
        goes_ts = ts.TimeSeries(file_goes,concatenate=True).to_dataframe()
        ax2 = ax.twinx()
        ax.plot(goes_ts.index,(np.log10(goes_ts.xrsb)+9)/6*1.1-0.05,'k',alpha=0.4)
        ax2.set_yscale('log')
        ax2.set_ylim((1e-9,1e-3))
        ax2.grid(True)
        ax2.set_ylabel('W/m^2')
    ax.plot(df['sample_time'],df['ytrue'],'--k')
    ax.plot(df['sample_time'],df[cal+'_median'],'.r')
    ax.fill_between(df['sample_time'],df.filter(regex=cal+'[0-9]').min(axis=1),df.filter(regex=cal+'[0-9]').max(axis=1),alpha=.7,interpolate=True)

    plt.xlim((tstart,tend))
    ax.set_ylim((-0.05,1.05))
