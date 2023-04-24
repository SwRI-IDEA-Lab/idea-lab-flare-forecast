import sys,os
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.metrics import average_precision_score,roc_auc_score
from src.data import split_data
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

class LinearModel():
    """
        Logistic regression model for flare forecasting
    """
    def __init__(self,data_file:str,window:int,val_split:int=0,flare_thresh:float=1e-5):
        self.data_file = data_file
        self.window = window
        self.flare_thresh = flare_thresh
        self.val_split = val_split
        self.scaler = MaxAbsScaler()
        self.features = ['tot_us_flux']
        self.label = 'flare'
        self.model = LogisticRegression(class_weight='balanced',random_state=val_split)

    def prepare_data(self):
        # load and prep dataframe
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        self.df['flare'] = (self.df['flare_intensity_in_'+str(self.window)+'h']>=self.flare_thresh).astype(int)

    def setup(self):
        # split data
        self.df_test,self.df_pseudotest,self.df_train,self.df_val = split_data(self.df,self.val_split)
        self.scaler.fit(self.df_train[self.features])
        self.X_train = self.scaler.transform(self.df_train[self.features])
        self.X_val = self.scaler.transform(self.df_val[self.features])
        self.X_pseudotest = self.scaler.transform(self.df_pseudotest[self.features])
        self.X_test = self.scaler.transform(self.df_test[self.features])
        return

    def train(self):
        self.model.fit(self.X_train,self.df_train[self.label])

    def test(self,X,y):
        ypred = self.model.predict_proba(X)
        return ypred[:,1]
        

if __name__ == "__main__":
    data_file = 'Data/labels_all_smoothed2.csv'
    window = 24
    print('Window: ',window,'h')

    for val_split in range(5):
        model = LinearModel(data_file=data_file,window=window,val_split=val_split)
        model.prepare_data()
        model.setup()
        model.train()
        ypred = model.test(model.X_pseudotest,model.df_pseudotest['flare'])
        y = model.df_pseudotest['flare']
        print('MSE:',(sum((ypred-y)**2))/len(ypred),
            'BSS:',(sum((ypred-y)**2)-sum((sum(y)/len(y)-y)**2))/(-sum((sum(y)/len(y)-y)**2)),
            'APS:',average_precision_score(y,ypred),
            'Gini:',2*roc_auc_score(y,ypred)-1)