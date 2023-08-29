
import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(torch.nn.Module):
    """
    Multi-label focal loss implementation

    Inputs:
        gamma (float):  exponent for focusing
        alpha (list):   class weights: either a list of weights corresponding to each class 
                        or a scalar for the positive class in binary classification
        reduction:      mean, sum or none
    """
    def __init__(self, gamma=0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.reduction = reduction

    def forward(self, input, target):
        target = target.type_as(input.data)
        logpt = - F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -( (1-pt)**self.gamma ) * logpt
        loss = self.alpha**target * (1-self.alpha)**(1-target) * focal_loss
        loss = loss.sum(dim=1)

        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss

class WeightedMSELoss(torch.nn.Module):
    """
    MSE loss weighted by the true value: (y-y')^2*y 

    Parameters:
        reduction (str):    mean, sum or none
    """
    __constants__ = ['reduction']

    def __init__(self,eps:float=0.1,reduction: str = 'mean'):
        super().__init__()
        self.eps = torch.Tensor([eps])
        self.reduction = reduction


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.type_as(input.data)

        mse = F.mse_loss(input,target,reduction='none')
        loss = mse*(target+self.eps.type_as(input.data))
        
        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss
