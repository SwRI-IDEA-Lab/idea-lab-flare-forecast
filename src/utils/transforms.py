from torch import nn
import torch

class RandomPolaritySwitch(torch.nn.Module):
    """Inverts the polarity of the given magnetogram randomly with a given probability.
    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.

    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be inverted.

        Returns:
            Tensor: Randomly polarity inverted image.
        """
        if torch.rand(1).item() < self.p:
            return -img
        
        return img


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"