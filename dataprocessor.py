import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler, Dataset
from data.imbalanced_cifar import IMBALANCECIFAR100, IMBALANCECIFAR10
import matplotlib.pyplot as plt
import numpy as np

class IndexedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index

class IndexedCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index

class IndexedCIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        return data, target, index

class DataProcessor():
    def __init__(
        self,
        batch_size: int = 64,
        dataset="mnist",
        dataset_split: float = 0.8,
        device="cuda",
    ):
        self.batch_size = batch_size
        self.error_indices = []
        
        if dataset == "mnist":
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5],std=[0.5])])
            self.source_data = IndexedMNIST(root = "./data/",
                                        transform=transform,
                                        train = True,
                                        download = True)

            self.data_test = IndexedMNIST(root="./data/",
                                    transform = transform,
                                    train = False)
            split = int(dataset_split*len(self.source_data))
            self.data_train, self.data_val = torch.utils.data.random_split(self.source_data, [split, len(self.source_data)-split])
        if dataset == "cifar-10":
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])
            self.source_data = IndexedCIFAR10(root = "./data/",
                                        transform=transform,
                                        train = True,
                                        download = True)

            self.data_test = IndexedCIFAR10(root="./data/",
                                    transform = transform,
                                    train = False)
            split = int(dataset_split*len(self.source_data))
            self.data_train, self.data_val = torch.utils.data.random_split(self.source_data, [split, len(self.source_data)-split])
        if dataset == "cifar10-lt":
            transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.source_data = IMBALANCECIFAR10(root='./data', train=True,
                        download=True, transform=transform)
            self.data_test = IndexedCIFAR10(root="./data/",
                                    transform = transform,
                                    train = False)
            split = int(1.0*len(self.source_data))
            self.data_train, self.data_val = torch.utils.data.random_split(self.source_data, [split, len(self.source_data)-split])
            self.data_val = IMBALANCECIFAR10(root='./data', train=False,
                            transform=transform)
        self.train_and_error_indices = self.data_train.indices + self.error_indices

    def load_data(self):
        data_loader_train = torch.utils.data.DataLoader(dataset=self.data_train,
                                                        batch_size = self.batch_size,
                                                        shuffle = True,
                                                        drop_last = True)
        data_loader_val = torch.utils.data.DataLoader(dataset=self.data_val,
                                                        batch_size = self.batch_size,
                                                        shuffle = True,
                                                        drop_last = True)
        data_loader_test = torch.utils.data.DataLoader(dataset=self.data_test,
                                                    batch_size = self.batch_size,
                                                    shuffle = True,
                                                        drop_last = True)
        return data_loader_train, data_loader_val, data_loader_test

    def build_error_set(self):
        data_loader_error = torch.utils.data.DataLoader(dataset=self.source_data,
                                                    batch_size = self.batch_size,
                                                    sampler = SubsetRandomSampler(self.error_indices),
                                                    drop_last = True)
        return data_loader_error

    def build_train_and_error_set(self):
        data_loader_train_and_error = torch.utils.data.DataLoader(dataset=self.source_data,
                                                    batch_size = self.batch_size,
                                                    sampler = SubsetRandomSampler(self.train_and_error_indices),
                                                        drop_last = True)
        return data_loader_train_and_error

    def update_error_indices(self, error_list):
        [self.error_indices.pop() for i in range(len(self.error_indices))]
        self.error_indices.extend(error_list)
        [self.train_and_error_indices.pop() for i in range(len(self.train_and_error_indices))]
        self.train_and_error_indices.extend(self.data_train.indices)
        self.train_and_error_indices.extend(error_list)


if __name__ == "__main__":
    data_processor = DataProcessor(dataset="mnist")
    data_loader_train, data_loader_val, data_loader_test = data_processor.load_data()
    data_loader_error = data_processor.build_error_set()
    data_processor.update_error_indices([1, 2, 3, 49999])
    images, labels, indices = next(iter(data_loader_error))
    print(indices)
    img = torchvision.utils.make_grid(images)

    img = img.numpy().transpose(1,2,0)
    std = [0.5]
    mean = [0.5]
    img = img*std+mean
    print([labels[i] for i in range(64)])
    plt.imshow(img)  
    plt.show()                                              