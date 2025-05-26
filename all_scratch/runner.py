
from data_setup import minst_dataset as dsets
import unittest
from arch import cnn as cnn
import torch

class TestAllScratch(unittest.TestCase):
    def test_logistic_regression(self):
        """
        Test class for Logistic Regression model.
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

        # for i, (single_sample, y_train) in enumerate(train_loader):
        #     if i > 0:
        #         break
        #     # single_sample: [batch_size, 1, 28, 28]
        #     # y_train: [batch_size]

        #     # Reshape single_sample to 2D: [batch_size, 28*28]
        #     single_sample = single_sample.view(single_sample.shape[0], -1)

        #     # Print shape of single_sample
        #     print('='*50)
        #     print(f'Single sample shape: \t {list(single_sample.shape)}')
        #     print('='*50)


        #     # Print shape of single_sample
        #     print('='*50)
        #     print(f'Single sample shape: \t {list(y_train.shape)}')
        #     print('='*50)

        # a = next(iter(train_loader))
        single_sample, y_train = next(iter(train_loader))
        # single_sample = single_sample.view(single_sample.shape[0], -1)
        # y_train = y_train.view(y_train.shape[0], -1)
        single_sample = single_sample.squeeze(0)  # Remove batch dimension
        y_train = y_train.squeeze(0)  # Remove batch dimension
        single_sample = single_sample.view( 28, 28)  # Reshape to 2D
        y_train = y_train.view(1)  # Reshape to 1D
        print('='*50)
        print(f'Input shape: \t {list(single_sample.shape)}')
        print('='*50)

        # Forward: conv
        conv = cnn.ConvolutionalLayer(num_kernels=8, kernel_shape=[5, 5])
        output = conv.forward(single_sample)

        print('='*50)
        print(f'Conv (f) shape: \t {list(output.shape)}')
        print('='*50)
        

        # Forward: Maxpool
        pool = cnn.MaxPoolLayer(pooling_kernel_shape=2)
        output = pool.forward(output)

        # Affine and Softmax
        print('='*50)
        print(f'Input shape: \t {list(output.shape)}')
        print('='*50)

        # Forward: Affine and Softmax
        affinesoftmax = cnn.AffineAndSoftmaxLayer(affine_weight_shape=(output.shape[0], output.shape[1], output.shape[2], len(torch.unique(y_train))))
        output = affinesoftmax.forward(output)

        print('='*50)
        print(f'Affine & Soft(arg)max (f) shape: \t {list(output.shape)}')
        print('='*50)
        

if __name__ == '__main__':
    unittest.main()