import sys,os
sys.path.append(os.getcwd())

import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.metrics import average_precision_score,roc_auc_score
from src.data import split_data
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

class MLPModel():
    """
        MLP model for flare forecasting regression
    """
    def __init__(self,data_file:str,features:list,window:int,val_split:int=0,
                 flare_thresh:float=1e-5,hidden_layer_sizes=(100,)):
        self.data_file = data_file
        self.window = window
        self.flare_thresh = flare_thresh
        self.val_split = val_split
        self.scaler = StandardScaler()
        self.features = features
        self.label = 'flare'
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,random_state=val_split)

    def prepare_data(self):
        # load and prep dataframe
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        self.df['flare'] = self.df['xrsb_max_in_'+str(self.window)+'h']
        self.df['flare'] = (np.log10(self.df['flare'])+8.5)/6
        self.p_thresh = (np.log10(self.flare_thresh)+8.5)/6
        self.df.loc[self.df['flare']<0,'flare'] = 0
        self.df = self.df.dropna(axis=0,subset='flare')

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
        ypred = self.model.predict(X)
        return ypred
        

if __name__ == "__main__":
    data_file = 'Data/labels_regression_cleanedandchecked.csv'
    window = 24
    print('Window: ',window,'h')

    for val_split in range(5):
        model = MLPModel(data_file=data_file,features=['tot_us_flux'],window=window,val_split=val_split)
        model.prepare_data()
        model.setup()
        model.train()
        ypred = model.test(model.X_pseudotest,model.df_pseudotest['flare'])
        y = model.df_pseudotest['flare']
        print('MSE:',(sum((ypred*6-y*6)**2))/len(ypred))