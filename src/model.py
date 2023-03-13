from torch import nn, optim
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class convnet_sc(nn.Module):
    """ 
    Single stream conv net to ingest full-disk magnetograms based on Subhamoy Chatterjee's architecture

    Parameters:
        dim (int):    square dimension of input image
        length (int): number of images in a sequence
        dropoutRatio (float):   percentage of disconnections for Dropout

    """
    def __init__(self, dim:int=256, length:int=1, dropoutRatio:float=0.0):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, (3,3),padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, (3,3),padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, (3,3),padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block4 = nn.Sequential(
            nn.ZeroPad2d((2,2)),
            nn.Conv2d(64, 128, (3,3),padding='valid'),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, (3,3),padding='valid'),
            nn.ReLU(inplace=True),
        )

        self.fcl = nn.Sequential(
            nn.LazyLinear(100),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropoutRatio),
            nn.Linear(100,1),
            nn.Sigmoid()
        )
        
        self.forward(torch.ones(1,1,dim,dim))
        self.apply(self._init_weights)
        

    def _init_weights(self,module):
        """
            Function to check for layer instances within the model and initialize
            weights and biases.   We are using glorot/xavier uniform for the 2D convolution
            weights and random normal for the linear layers.  All biases are initilized
            as zeros.
        """
        if isinstance(module,nn.Conv2d):
            # nn.init.zeros_(module.weight)
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.zero_()

        if isinstance(module,nn.Linear):
            # nn.init.zeros_(module.weight)
            nn.init.normal_(module.weight)
            module.bias.data.zero_()

        if isinstance(module,nn.LazyLinear):
            # nn.init.zeros_(module.weight)
            nn.init.normal_(module.weight)
            module.bias.data.zero_()

    def forward(self,x):
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.shape[0],-1)
        x = self.fcl(x)
        return x

class LitConvNet(pl.LightningModule):
    """
        PyTorch Lightning module to classify magnetograms as flaring or non-flaring

        Parameters:
            model (torch.nn.Module):    a PyTorch model that ingests magnetograms and outputs a binary classification
            lr (float):                 learning rate
            wd (float):                 L2 regularization parameter
            epochs (int):               Number of epochs for scheduler
    """
    def __init__(self,model,lr:float=1e-4,wd:float=1e-2,epochs:int=100):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = wd
        self.epochs = epochs
        # define loss function
        self.loss = nn.BCELoss()    
        # define metrics
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.val_aps = torchmetrics.AveragePrecision(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.val_bss = torchmetrics.MeanSquaredError()
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(task='binary',num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.test_aps = torchmetrics.AveragePrecision(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')
        self.test_bss = torchmetrics.MeanSquaredError()
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task='binary',num_classes=2)
        self.save_hyperparameters(ignore=['model'])

    def training_step(self,batch,batch_idx):
        """
            Expands a batch into image and label, runs the model forward and 
            calculates loss.

            Parameters:
                batch:                  batch from a DataLoader
                batch_idx:              index of batch                  

            Returns:
                loss (torch tensor):    loss evaluated on batch
        """
        fname, x, y = batch
        y = y.view(y.shape[0],-1)
        y_hat = self.model(x)
        loss = self.loss(y_hat,y.type(torch.FloatTensor))
        self.train_acc(y_hat,y)
        self.log('loss',loss)
        self.log('train_acc',self.train_acc,on_step=True,on_epoch=False)
        return loss
    
    def validation_step(self,batch,batch_idx):
        """
            Runs the model on the validation set and logs validation loss 
            and other metrics.

            Parameters:
                batch:                  batch from a DataLoader
                batch_idx:              index of batch                  
        """
        fname, x, y = batch
        y = y.view(y.shape[0],-1)
        # forward pass
        y_hat = self.model(x)
        val_loss = self.loss(y_hat,y.type(torch.FloatTensor))

        # calculate metrics
        self.val_acc(y_hat,y)
        self.val_aps(y_hat,y)
        self.val_f1(y_hat,y)
        self.val_bss(y_hat,y)
        self.val_confusion_matrix.update(y_hat,y)

        self.log('val_loss',val_loss)
        self.log('val_acc',self.val_acc)
        self.log('val_aps',self.val_aps)
        self.log('val_f1',self.val_f1)
        self.log('val_bss',self.val_bss)

    def validation_epoch_end(self,outputs):
        """
        Finish logging validation metrics at end of epoch
        """
        confusion_matrix = self.val_confusion_matrix.compute()
        tp = confusion_matrix[1,1].type(torch.FloatTensor)
        tn = confusion_matrix[0,0].type(torch.FloatTensor)
        fp = confusion_matrix[0,1].type(torch.FloatTensor)
        fn = confusion_matrix[1,0].type(torch.FloatTensor)
        tss = (tp) / (tp + fn) - (fp) / (fp + tn)
        hss = 2*(tp*tn-fp*fn)/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))
        self.log_dict({'val_TP':tp,'val_TN':tn,'val_FP':fp,'val_FN':fn,'val_tss':tss,'val_hss':hss})
        self.val_confusion_matrix.reset()

    def test_step(self,batch,batch_idx):
        """
            Runs the model on the test set and logs test metrics 

            Parameters:
                batch:                  batch from a DataLoader
                batch_idx:              index of batch                  
        """
        fname, x, y = batch
        y = y.view(y.shape[0],-1)
        # forward pass
        y_hat = self.model(x)

        # calculate metrics
        self.test_acc(y_hat,y)
        self.test_aps(y_hat,y)
        self.test_f1(y_hat,y)
        self.test_bss(y_hat,y)
        self.test_confusion_matrix.update(y_hat,y)

        self.log('test_acc',self.test_acc)
        self.log('test_aps',self.test_aps)
        self.log('test_f1',self.test_f1)
        self.log('test_bss',self.test_bss)

    def test_epoch_end(self,outputs):
        confusion_matrix = self.test_confusion_matrix.compute()
        tp = confusion_matrix[1,1].type(torch.FloatTensor)
        tn = confusion_matrix[0,0].type(torch.FloatTensor)
        fp = confusion_matrix[0,1].type(torch.FloatTensor)
        fn = confusion_matrix[1,0].type(torch.FloatTensor)
        tss = (tp) / (tp + fn) - (fp) / (fp + tn)
        hss = 2*(tp*tn-fp*fn)/((tp+fp)*(fp+tn)+(tp+fn)*(fn+tn))
        self.log_dict({'test_TP':tp,'test_TN':tn,'test_FP':fp,'test_FN':fn,'test_tss':tss,'test_hss':hss})
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        """
            Sets up the optimizer and learning rate scheduler.
            
            Returns:
                optimizer:              A torch optimizer
        """
        optimizer = optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=self.epochs)
        return [optimizer],[scheduler]

    def predict_step(self,batch,batch_idx,dataloader_idx=0):
        """
            Forward pass of model for prediction

            Parameters:
                batch:          batch from a DataLoader
                batch_idx:      batch index
                dataloader_idx

            Returns:
                y_pred (tensor): model outputs for the batch
                y_true (tensor): true labels for the batch
        """
        fname, x, y = batch
        y = y.view(y.shape[0],-1)
        return self.model(x), y