from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

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
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, (3,3),padding='valid'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, (3,3),padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block4 = nn.Sequential(
            nn.ZeroPad2d((2,2)),
            nn.Conv2d(64, 128, (3,3),padding='valid'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=(2,2))
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(128, 256, (3,3),padding='valid'),
            nn.ReLU(),
        )

        self.fcl = nn.Sequential(
            nn.Linear(43008,100),
            nn.ReLU(),
            nn.Dropout1d(dropoutRatio),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

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
            
    """
    def __init__(self,model):
        super().__init__()
        self.model = model
        self.loss = nn.BCELoss()    # define loss function

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
        loss = self.loss(y_hat,y)
        self.log('loss',loss)
        return loss

    def configure_optimizers(self,lr=1e-4,weight_decay=1e-2):
        """
            Sets up the optimizer. Here we use Adagrad.

            Parameters:
                lr (float):             learning rate
                weight_decay (float):   L2 regularization parameter
            
            Returns:
                optimizer:              A torch optimizer
        """
        optimizer = optim.Adam(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        return optimizer

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