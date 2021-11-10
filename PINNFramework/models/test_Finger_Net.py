import unittest
import torch

from Finger_Net import FingerNet
from torch.nn import Module



class FingerNetTest(unittest.TestCase):

    def test_constructor(self):
        lb = [0, 0, 0]
        ub = [1, 1, 1]
        model = FingerNet(lb, ub, 3, 1)
        self.assertIsNotNone(model, "Model is not none")  # add assertion here
        self.assertIsInstance(model, Module, "Model is instance of torch.nn.module ")
        del model

    def test_architecture(self):
        # test case with 1 finger
        InputSize = 1
        OutputSize = 1
        lb = [0]
        ub = [1]
        model = FingerNet(lb, ub, InputSize, OutputSize)
        self.assertEqual(len(model.finger_nets), InputSize)
        del model

        InputSize = 2
        OutputSize = 1
        lb = [0, 0, 0]
        ub = [1, 1, 1]
        model = FingerNet(lb, ub, InputSize , OutputSize)
        self.assertEqual(len(model.finger_nets), InputSize)
        del model

        InputSize = 10
        OutputSize = 1
        lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ub = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        model = FingerNet(lb, ub, InputSize , OutputSize)
        self.assertEqual(len(model.finger_nets),InputSize)
        del model

        # there are more things you can test here

    def test_device_movement(self):
        InputSize = 2
        OutputSize = 1
        lb = [0, 0]
        ub = [1, 1]
        model = FingerNet(lb, ub, InputSize, OutputSize)

        # moving model to cuda
        model.cuda()
        self.assertEqual(str(model.lb.device), 'cuda:0')
        self.assertEqual(str(model.ub.device), 'cuda:0')
        self.assertEqual(str(model.finger_nets[0][0].weight.device), 'cuda:0')
        self.assertEqual(str(model.lin_layers[0].weight.device), 'cuda:0')

        # moving model to cpu
        model.cpu()
        self.assertEqual(str(model.lb.device), 'cpu')
        self.assertEqual(str(model.ub.device), 'cpu')
        self.assertEqual(str(model.finger_nets[0][0].weight.device), 'cpu')
        self.assertEqual(str(model.lin_layers[0].weight.device), 'cpu')

        # use `to`-function to move it back to gpu
        model.to('cuda:0')
        self.assertEqual(str(model.lb.device), 'cuda:0')
        self.assertEqual(str(model.ub.device), 'cuda:0')
        self.assertEqual(str(model.finger_nets[0][0].weight.device), 'cuda:0')
        self.assertEqual(str(model.lin_layers[0].weight.device), 'cuda:0')
        del model

    def test_forward(self):
        # test forward on cpu
        InputSize = 2
        OutputSize = 1
        lb = [0, 0]
        ub = [1, 1]
        model = FingerNet(lb, ub, InputSize, OutputSize)
        x = torch.rand(10, InputSize)
        y = model(x)
        self.assertEqual(y.shape, (10, OutputSize))

        # test on different input and output size
        InputSize = 3
        OutputSize = 3
        lb = [0, 0, 0]
        ub = [1, 1, 1]
        model = FingerNet(lb, ub, InputSize, OutputSize)
        x = torch.rand(10, InputSize)
        y = model(x)
        self.assertEqual(y.shape, (10, OutputSize))

        # test forward on gpu
        InputSize = 3
        OutputSize = 3
        lb = [0, 0, 0]
        ub = [1, 1, 1]
        model = FingerNet(lb, ub, InputSize, OutputSize)
        model.cuda()
        x = torch.rand(10, InputSize ,device='cuda:0')
        y = model(x)
        self.assertEqual(y.shape, (10, OutputSize))
        self.assertEqual(str(y.device), 'cuda:0')


if __name__ == '__main__':
    unittest.main()
