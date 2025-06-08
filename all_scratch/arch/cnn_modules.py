import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=None):
        super(ConvolutionalLayer, self).__init__()
        # Number of kernels: 1D
        self.in_channels = in_channels
        ## Kernel is Square shape slider will slide across input with fixed kernel size shape
        self.out_channels = out_channels
        # Shape of kernels: 2D
        # Kernal is Square shape
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # Kernel weights: 3D
        self.kernels_theta = nn.Parameter(torch.randn(self.out_channels, self.kernel_size, self.kernel_size))

    def slider(self, indx,channel, inp):
        '''
        Sliding generator that yields square areas of shape
        (kernel_shape, kernel_shape) sliding across our input. 
        This assumes valid padding (no padding) and step size 1.
        '''
        _,_, h, w = inp.shape
        # Slide across height
        for h_idx in range(0,h - (self.kernel_size - 1)):
            # Slide across width
            for w_idx in range(0, w - (self.kernel_size - 1)):
                single_slide_area = inp[indx][channel][h_idx:(h_idx + self.kernel_size), w_idx:(w_idx + self.kernel_size)]
                yield single_slide_area, h_idx,w_idx

    def forward(self, inp):
        '''
        Slides kernel across image doing an element-wise MM then summing.
        Results in forward pass of convolutional layer of shape
        (output shape, output shape, number of kernels).
        '''
        # Input: 2D of (height, width)
        # assert single_sample.dim() == 2, f'Input not 2D, given {single_sample.dim()}D'

        # Output via Valid Padding (No Padding): 3D of (height, width, number of kernels)
        batch_num,in_channel,in_h, w  = inp.shape
        # P = 0
        p = 0
        # O = ((W - K + 2P) / S) + 1 = (28 - 3 + 0) + 1 = 25
        o = (w - self.kernel_size) + 1
        # Print shapes
        # print('Padding shape: \t', p)
        # print('Output shape: \t', o)
        # Initialize blank tensor
        output = torch.zeros((batch_num, self.out_channels,o, o))
        for i in range(batch_num): 
            # Iterate through region
            # Iterate through each channel
            for channel in range(in_channel):
                actual_tensor = torch.zeros((o, o, self.out_channels))
                for single_slide_area, h_idx, w_idx in self.slider(i, channel, inp):

                    # Sum values with each element-wise matrix multiplication across each kernel
                    # Instead of doing another loop of each kernel, you simply just do a element-wise MM
                    # of the single slide area with all the kernels yield, then summing the patch
                    actual_tensor[h_idx, w_idx] = torch.sum(single_slide_area * self.kernels_theta, dim=(1, 2))

                    # Pass through non-linearity (sigmoid): 1 / 1 + exp(-output)
                    # actual_tensor = 1. / (1. + torch.exp(-actual_tensor))
                # Assign to output tensor
                # Transpose to (out_channels, o, o)
                # where o = ((W - K + 2P) / S) + 1
                # where W = width, K = kernel size, P = padding, S = stride
                # where o = (28 - 5 + 0) / 1 + 1 = 24


            output[i] = actual_tensor.transpose(0,2)
        print(f'Output shape: \t {list(output.shape)}')
        # Return output of shape (batch_num, out_channels, o, o)
        # where o = ((W - K + 2P) / S) + 1
        # where W = width, K = kernel size, P = padding, S = stride
        return output

class MaxPoolLayer(nn.Module):
    # O = ((W - K) / S) + 1
    def __init__(self, kernel_size):
        super(MaxPoolLayer, self).__init__()
        '''
        Max Pooling Layer that performs max pooling operation.
        - kernel_size: Size of the square kernel to slide across the input.
        - stride: Step size for sliding the kernel across the input.
        - padding: Padding applied to the input before pooling.
        '''
        # Assume simplicity of K = S then O = W / S
        self.kernel_size = kernel_size

    def slider(self, inp):
        '''
        Sliding generator that yields areas for max pooling.
        '''
        h, w = inp.shape
        output_size = int(w / self.kernel_size)  # Assume S = K

        for h_idx in range(output_size):
            for w_idx in range(output_size):
                single_slide_area = inp[h_idx * self.kernel_size:h_idx * self.kernel_size + self.kernel_size,
                                        w_idx * self.kernel_size:w_idx * self.kernel_size + self.kernel_size]
                yield single_slide_area, h_idx, w_idx

    def forward(self, inp):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        self.last_input = inp

        batch, num_kernels,h, w  = inp.shape
        output_size = int(w / self.kernel_size)  # Assume S = K
        output = torch.zeros(batch, num_kernels, output_size, output_size)

        # Iterate through each batch
        for i in range(batch):
            temp = torch.zeros(( output_size, output_size, num_kernels))
            # Iterate through each kernel
            for channel in range(num_kernels):
                # Iterate through each region
                # For each region, find the max value and assign to output
                # Use the slider to yield areas of shape (kernel_size, kernel_size)
                for single_slide_area, h_idx, w_idx in self.slider(inp[i][channel]):
                    # single_slide_area: (kernel_size, kernel_size)
                    single_slide_area = torch.flatten(single_slide_area, start_dim=0, end_dim=1)
                    temp[h_idx, w_idx] = torch.max(single_slide_area, dim=0).values
            # Assign to output tensor
            output[i] = temp.transpose(0, 2)
        # Return output of shape (batch, h / 2, w / 2, num_kernels)
        return output

class AvgPoolLayer(nn.Module):
    # O = ((W - K) / S) + 1
    def __init__(self, kernel_size):
        super(AvgPoolLayer, self).__init__()
        '''
        Avg Pooling Layer that performs Avg pooling operation.
        - kernel_size: Size of the square kernel to slide across the input.
        - stride: Step size for sliding the kernel across the input.
        - padding: Padding applied to the input before pooling.
        '''
        # Assume simplicity of K = S then O = W / S
        self.kernel_size = kernel_size

    def slider(self, inp):
        '''
        Sliding generator that yields areas for Avg pooling.
        '''
        h, w = inp.shape
        output_size = int(w / self.kernel_size)  # Assume S = K

        for h_idx in range(output_size):
            for w_idx in range(output_size):
                single_slide_area = inp[h_idx * self.kernel_size:h_idx * self.kernel_size + self.kernel_size,
                                        w_idx * self.kernel_size:w_idx * self.kernel_size + self.kernel_size]
                yield single_slide_area, h_idx, w_idx

    def forward(self, inp):
        '''
        Performs a forward pass of the Avgpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        self.last_input = inp

        batch, num_kernels,h, w  = inp.shape
        output_size = int(w / self.kernel_size)  # Assume S = K
        output = torch.zeros(batch, num_kernels, output_size, output_size)

        # Iterate through each batch
        for i in range(batch):
            temp = torch.zeros(( output_size, output_size, num_kernels))
            # Iterate through each kernel
            for channel in range(num_kernels):
                # Iterate through each region
                # For each region, find the Avg value and assign to output
                # Use the slider to yield areas of shape (kernel_size, kernel_size)
                for single_slide_area, h_idx, w_idx in self.slider(inp[i][channel]):
                    # single_slide_area: (kernel_size, kernel_size)
                    single_slide_area = torch.flatten(single_slide_area, start_dim=0, end_dim=1)
                    temp[h_idx, w_idx] = torch.Avg(single_slide_area, dim=0).values
            # Assign to output tensor
            output[i] = temp.transpose(0, 2)
        # Return output of shape (batch, h / 2, w / 2, num_kernels)
        return output

class AffineAndSoftmaxLayer(nn.Module):
    '''
    Affine Layer that performs a linear transformation followed by a softmax activation.
    - affine_weight_shape: Shape of the weight matrix for the affine transformation.
    '''
    def __init__(self, affine_weight_shape):
        super(AffineAndSoftmaxLayer, self).__init__()
        '''
        Initializes the Affine Layer with the given weight shape.
        - affine_weight_shape: Shape of the weight matrix for the affine transformation.
        '''
        self.affine_weight_shape = affine_weight_shape
        # Weight shape: flattened input x output shape
        self.w = nn.Parameter(torch.zeros(self.affine_weight_shape[0] * self.affine_weight_shape[1] * self.affine_weight_shape[2], self.affine_weight_shape[3]))
        self.b = nn.Parameter(torch.zeros(self.affine_weight_shape[3]))

        # Initialize weight/bias via Lecun initialization of 1 / N standard deviation
        # Refer to DLW guide on weight initialization mathematical derivation:
        # https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/
        print(f'Lecun initialization SD: {1/self.affine_weight_shape[3]}')
        self.w = nn.Parameter(torch.nn.init.normal_(self.w, mean=0, std=1/self.affine_weight_shape[3]))
        self.b = nn.Parameter(torch.nn.init.normal_(self.b, mean=0, std=1/self.affine_weight_shape[3]))

    def forward(self, inp):
        '''
        Performs Linear (Affine) Function & Soft(arg)max Function
        that returns our vector (1D) of probabilities.
        '''
        output = torch.zeros((inp.shape[0], self.affine_weight_shape[3]), dtype=inp.dtype, device=inp.device)
        for i in range(inp.shape[0]):
            # Flatten input to 1D
            # print(f'input shape: \t {inp.shape}')
            # print(f'weight shape: \t {self.w.shape}')
            # print(f'bias shape: \t {self.b.shape}')
            tmp = inp[i].reshape(1,-1)
            logits = torch.add(torch.mm(tmp, self.w),self.b)
            probas = torch.exp(logits) / torch.sum(torch.exp(logits))
            output[i] = probas
        # Return output of shape (batch_size, num_classes)
        return output
        

        

