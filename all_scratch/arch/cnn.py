import torch

class ConvolutionalLayer:
    def __init__(self, in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=None):
        # Number of kernels: 1D
        self.out_channels = out_channels
        # Shape of kernels: 2D
        # Kernal is Square shape
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        # Kernel weights: 3D
        self.kernels_theta = torch.randn(self.out_channels, self.kernel_size, self.kernel_size)

    def slider(self, inp):
        '''
        Sliding generator that yields square areas of shape
        (kernel_shape, kernel_shape) sliding across our input. 
        This assumes valid padding (no padding) and step size 1.
        '''
        h, w = inp.shape
        # Slide across height
        for h_idx in range(0,h - (self.kernel_size - 1), self.stride):
            # Slide across width
            for w_idx in range(0, w - (self.kernel_size - 1), self.stride):
                single_slide_area = inp[h_idx:(h_idx + self.kernel_size), w_idx:(w_idx + self.kernel_size)]
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
        _, w  = inp.shape
        # P = 0
        p = 0
        # O = ((W - K + 2P) / S) + 1 = (28 - 3 + 0) + 1 = 25
        o = (w - self.kernel_size) + 1
        # Print shapes
        print('Padding shape: \t', p)
        print('Output shape: \t', o)
        # Initialize blank tensor
        output = torch.zeros((o, o, self.out_channels))

        # Iterate through region
        for single_slide_area, h_idx, w_idx in self.slider(inp):
            if h_idx == 0 and w_idx == 0:
                print('Region shape: \t', list(single_slide_area.shape))
                print('Kernel shape: \t', list(self.kernels_theta.shape))
                print('Single Slide: \t', list(output[h_idx, w_idx].shape))

            # Sum values with each element-wise matrix multiplication across each kernel
            # Instead of doing another loop of each kernel, you simply just do a element-wise MM
            # of the single slide area with all the kernels yield, then summing the patch
            output[h_idx, w_idx] = torch.sum(single_slide_area * self.kernels_theta, axis=(1, 2))

        # Pass through non-linearity (sigmoid): 1 / 1 + exp(-output)
        output = 1. / (1. + torch.exp(-output))

        return output

class MaxPoolLayer:
    # O = ((W - K) / S) + 1
    def __init__(self, kernel_size):
        # Assume simplicity of K = S then O = W / S
        self.kernel_size = kernel_size

    def slider(self, inp):
        '''
        Sliding generator that yields areas for max pooling.
        '''
        h, w, _ = inp.shape
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

        h, w, num_kernels = inp.shape
        output_size = int(w / self.kernel_size)  # Assume S = K
        output = torch.zeros(output_size, output_size, num_kernels)

        for single_slide_area, h_idx, w_idx in self.slider(inp):
            single_slide_area = torch.flatten(single_slide_area, start_dim=0, end_dim=1)
            output[h_idx, w_idx] = torch.max(single_slide_area, dim=0).values

        return output

class AffineAndSoftmaxLayer:
    def __init__(self, affine_weight_shape):
        self.affine_weight_shape = affine_weight_shape
        # Weight shape: flattened input x output shape
        self.w = torch.zeros(self.affine_weight_shape[0] * self.affine_weight_shape[1] * self.affine_weight_shape[2], self.affine_weight_shape[3])
        self.b = torch.zeros(self.affine_weight_shape[3])

        # Initialize weight/bias via Lecun initialization of 1 / N standard deviation
        # Refer to DLW guide on weight initialization mathematical derivation:
        # https://www.deeplearningwizard.com/deep_learning/boosting_models_pytorch/weight_initialization_activation_functions/
        print(f'Lecun initialization SD: {1/self.affine_weight_shape[3]}')
        self.w = torch.nn.init.normal_(self.w, mean=0, std=1/self.affine_weight_shape[3])
        self.b = torch.nn.init.normal_(self.b, mean=0, std=1/self.affine_weight_shape[3])

    def forward(self, inp):
        '''
        Performs Linear (Affine) Function & Soft(arg)max Function
        that returns our vector (1D) of probabilities.
        '''
        inp = inp.reshape(1, -1)
        print(f'input shape: \t {inp.shape}')
        print(f'weight shape: \t {self.w.shape}')
        print(f'bias shape: \t {self.b.shape}')
        logits = torch.mm(inp, self.w) + self.b
        probas = torch.exp(logits) / torch.sum(torch.exp(logits))
        return probas
