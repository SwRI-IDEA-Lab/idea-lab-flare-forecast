from torch import nn
import torch.nn.functional as F


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

