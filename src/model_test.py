import unittest
from model import convnet_sc
import numpy as np
import torch
import h5py



class ModelTest(unittest.TestCase):

    def setUp(self):
        filename = 'Data\MDI_small\mdi.fd_m_96m_lev182.19971224_000000_TAI.h5'
        self.x = torch.tensor(np.array(h5py.File(filename, 'r')['magnetogram']).astype(np.float32))[None,None,:,:]
        self.x[torch.isnan(self.x)]=0
        self.model = convnet_sc()

    def test_modelexists(self):
        self.assertIsNotNone(self.model)
    
    def test_imgShape(self):
        self.assertTrue(self.x.shape==(1,1,256,256))

    def test_imgNaN(self):
        self.assertTrue(not torch.isnan(self.x).any())

    def test_modelforward(self):
        output = self.model.forward(self.x)
        print(output)
        self.assertTrue(output.shape[0] == 1)
        self.assertTrue(output >= 0 and output <= 1)

    def test_weightint(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                print(name, layer)        
                print(layer.weight)
                print(layer.bias)


if __name__ == "__main__":
    unittest.main()

