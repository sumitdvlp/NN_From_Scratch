import torchvision.transforms as transforms
import torchvision.datasets as dsets
# MNIST dataset setup
# Download the MNIST dataset and apply transformations
import torch

class MNISTDataset:
    def __init__(self,  download=True):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transforms.ToTensor()
        self.dataset = dsets.MNIST(root=self.root,
                                   train=self.train,
                                   transform=self.transform,
                                   download=self.download)
        self.train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
        self.test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
        
        self.dataset_size = len(self.dataset)
        self.num_classes = len(self.dataset.classes)
        self.input_shape = self.dataset[0][0].shape
        self.output_shape = (self.num_classes,)
    def __len__(self):
        return self.dataset_size
    def __getitem__(self, idx):
        return self.dataset[idx]
    def get_input_shape(self):
        return self.input_shape
    def get_output_shape(self):
        return self.output_shape
    def get_loader(self, batch_size=64, shuffle=True):
        return torch.utils.data.DataLoader(dataset=self.dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle)
    def getTestTrainDatasets(self):
        return self.train_dataset, self.test_dataset


