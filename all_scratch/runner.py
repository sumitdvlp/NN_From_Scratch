
from data_setup import minst_dataset as dsets
import unittest
from arch import cnn as cnn
from loss import common_loss as loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestAllScratch(unittest.TestCase):
    def test_CNN(self):
        """
        Test class for CNN Regression model.
        """

        batch_size = 1
        n_iters = 3000
        dataset = dsets.MNISTDataset()
        train_dataset, test_dataset = dataset.getTestTrainDatasets()
        
        print(f'Number of classes: {dataset.num_classes}')
        print(f'Number of samples: {dataset.num_samples}')
        
        print(f'Test output_shape: {dataset.output_shape}')
        print(f'Train input_shape: {dataset.input_shape}')

        input_dim = 28*28
        output_dim = 10
        num_epochs = n_iters / (len(train_dataset) / batch_size)
        num_epochs = int(num_epochs)


        print(f'Input shape {input_dim}')
        print(f'Output shape {output_dim}')
        print(f'Number of epochs {num_epochs}')

        train_loader, test_loader = dataset.getTestTrainLoaders(batch_size=batch_size)

        '''
        for i, (single_sample, y_train) in enumerate(train_loader):
            if i > 0:
                break
            # single_sample: [batch_size, 1, 28, 28]
            # y_train: [batch_size]

            # Reshape single_sample to 2D: [batch_size, 28*28]
            single_sample = single_sample.view(single_sample.shape[0], -1)

            # Print shape of single_sample
            print('='*50)
            print(f'Single sample shape: \t {list(single_sample.shape)}')
            print('='*50)


            # Print shape of single_sample
            print('='*50)
            print(f'Single sample shape: \t {list(y_train.shape)}')
            print('='*50)
        '''
        single_sample_batched, y_train = next(iter(train_loader))

        # single_sample = single_sample.squeeze(0)  # Remove batch dimension
        # y_train = y_train.squeeze(0)  # Remove batch dimension
        single_sample = single_sample_batched.view( 28, 28)  # Reshape to 2D
        y_train = y_train.view(1)  # Reshape to 1D
        print('='*50)
        print(f'Input shape: \t {list(single_sample.shape)}')
        print(f'Pytorch batched shape: \t {list(single_sample_batched.shape)}')
        print('='*50)

        print('*#'*50)
        num_kernels = 8
        kernel_shape = [5, 5]
        kernel_size=5
        stride=1
        padding=2
        print('*#'*50)

        # Forward: conv
        # Scratch Layer
        conv = cnn.ConvolutionalLayer(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        output = conv.forward(single_sample)
        print('=*'*50)
        print(f'Conv (f) shape: \t {list(output.shape)}')
        print('=*'*50)
        # Pytorch Layer
        pycnn1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0)
        out = pycnn1(single_sample_batched)
        print('=*'*50)
        print(f'Pytorch Conv (f) shape: \t {list(out.shape)}')
        print('=*'*50)
        # assert torch.allclose(output, out.squeeze(0), atol=1e-6), "Output mismatch between custom and PyTorch Conv implementation"

        # Forward: ReLU
        relu1 = loss.Activation_ReLU()
        output = relu1.forward(output)
        print('=*'*50)
        print(f'ReLu1 (f) shape: \t {list(output.shape)}')
        print('=*'*50)
        
        pyrelu1 = nn.ReLU()
        out = pyrelu1(out)
        print('=*'*50)
        print(f'Pytorch ReLu1 (f) shape: \t {list(out.shape)}')
        print('=*'*50)
        # assert torch.allclose(output, out.squeeze(0), atol=1e-6), "Output mismatch between custom and PyTorch Relu implementation"

        # Forward: Maxpool
        pool = cnn.MaxPoolLayer(kernel_size=2)
        output = pool.forward(output)
        print('=*'*50)
        print(f'Maxpool output shape: \t {list(output.shape)}')
        print('=*'*50)
        pytorchmaxpool1 = nn.MaxPool2d(kernel_size=2)
        out = pytorchmaxpool1(out)
        print('=*'*50)
        print(f'Pytorch MaxPool (f) shape: \t {list(out.shape)}')
        print('=*'*50)



        # Forward: Affine and Softmax
        affinesoftmax = cnn.AffineAndSoftmaxLayer(affine_weight_shape=(output.shape[0], output.shape[1], output.shape[2], len(torch.unique(y_train))))
        output = affinesoftmax.forward(output)

        print('='*50)
        print(f'Affine & Soft(arg)max (f) shape: \t {list(output.shape)}')
        print('='*50)

        # Affine and Softmax
        self.affine_softmax = nn.Linear(in_features=out.shape[1] * out.shape[2] * out.shape[3], out_features=len(torch.unique(y_train)))
        out = out.view(out.shape[0], -1)
        out = self.affine_softmax(out)
        print('='*50)
        print(f'Pytorch Affine & Soft(arg)max (f) shape: \t {list(out.shape)}')


seed_value=42
torch.manual_seed(seed_value)

if __name__ == '__main__':
    unittest.main()