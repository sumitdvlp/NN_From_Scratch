import torchvision.transforms as transforms
import torchvision.datasets as dsets
# MNIST dataset setup
# Download the MNIST dataset and apply transformations
import torch

class MNISTDataset:
    def __init__(self,download=False, root='./data'):
        self.root = root

        # Define the transformation to convert images to tensors
        self.transform = transforms.ToTensor()
        # Load the MNIST dataset
        self.train_dataset = dsets.MNIST(root=self.root,
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
        self.test_dataset = dsets.MNIST(root=self.root,
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
        self.num_classes = self.train_dataset.classes.__len__()
        self.num_samples = len(self.train_dataset)
        self.dataset_size = len(self.train_dataset)
        # Print dataset information
        print(f'Train dataset size: {self.train_dataset.data.shape}')
        print(f'Test dataset size: {self.test_dataset.data.shape}')
        print(f'Number of classes: {self.num_classes}')
        # Set input and output shapes
        self.input_shape = self.train_dataset[0][0].shape
        self.output_shape = (self.num_classes,)
        print(f'Input shape: {self.input_shape}')
        print(f'Output shape: {self.output_shape}')

    def getTestTrainDatasets(self):
        return self.train_dataset, self.test_dataset
    
    def getTestTrainLoaders(self, batch_size=1):
        """
        Returns the train and test dataloaders.
        """
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        return train_loader, test_loader


