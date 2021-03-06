import random
import numpy as np
import torchvision.datasets as datasets

import pickle

class NoisyCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None, noise_type='sym', noise_rate=0.1):
        super(NoisyCIFAR10, self).__init__(root, train=train, download=download, transform=transform)
        
        self.noise_rate = noise_rate
        self.noise_type = noise_type

        if noise_rate <= 0:
            return
        
        # num samples in dataset
        n_samples = self.__len__()
        
        # num classes in dataset
        n_classes = len(self.classes)
        
        # num noisy samples to generate
        n_noisy_per_class = int(noise_rate * n_samples / n_classes)
        
        # for each class add noise to noise_rate percentage of its samples 
        for c in range(n_classes):
            indeces = np.where(np.array(self.targets) == c)[0]
            noisy_samples_idx = np.random.choice(indeces, n_noisy_per_class, replace=False)            
            
            if noise_type == 'sym':
                # list of alternative class ids to choose from as a noisy target; excludes original class id
                class_ids = [i for i in range(n_classes) if i!=c]    

                for idx in noisy_samples_idx:
                    # pick a new class from the remaining 9 classes at random as noisy class for this sample
                    self.targets[idx] = random.choice(class_ids)
            elif noise_type == 'asym':
                for idx in noisy_samples_idx:
                    # use current_class+1 as the noisy class with prob noise_rate
                    current_class = self.targets[idx]
                    self.targets[idx] = np.random.choice([current_class, (current_class+1)%n_classes], p=[1-noise_rate, noise_rate])
            else:
                raise ValueError(f'Undefined noise_type: {noise_type}!') 

        return
    
    def dump_(self, path_='checkpoint/noisy_cifar10.pkl'):
        with open(path_,'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_(path_):
        with open(path_, 'rb') as f:
            return pickle.load(f)

## Usage
# from dataloader import NoisyCIFAR10
# noisy_cifar10_trainset = NoisyCIFAR10(root='./data', train=True, download=True, noise_type = 'sym', noise_rate=0.2)

# save dataset as a pickle for re-use
# dataset.dump_(path_='checkpoint/cifar10_noisy.pkl')

# re-use existing dataset pickle
# dataset = NoisyCIFAR10.load_(path_ = 'checkpoint/cifar10_noisy.pkl')


class NoisyCIFAR10_Subset(datasets.CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None, noise_type='sym', noise_rate=0.1, split_ratio=0.5):
        super(NoisyCIFAR10_Subset, self).__init__(root, train=train, download=download, transform=transform)
        
        self.split_ratio = split_ratio
        
        # num samples in dataset
        n_samples = self.__len__()
        
        # num classes in dataset
        n_classes = len(self.classes)
        
        # num samples per class
        n_samples_per_class = int(split_ratio * n_samples / n_classes)
        
        pre_split_idx = []
        for c in range(n_classes):
            indeces = np.where(np.array(self.targets) == c)[0]
            np.random.seed(42)
            samples_idx = np.random.choice(indeces, n_samples_per_class, replace=False)            
            pre_split_idx.extend(samples_idx)
        self.pre_split_idx = pre_split_idx
        
        # create the pretraining dataset using 50% of the data
        self.data_pre_split = self.data[self.pre_split_idx]
        self.targets_pre_split = [self.targets[i] for i in self.pre_split_idx]
        self.pretrn_cifar10=datasets.CIFAR10(root, train=train, download=download, transform=transform)
        self.pretrn_cifar10.data=self.data_pre_split
        self.pretrn_cifar10.targets=self.targets_pre_split
        
        # remove pretraining split from this dataset
        self.targets = np.delete(self.targets, self.pre_split_idx)
        self.data = np.delete(self.data, self.pre_split_idx, axis=0)
                  
        ## add noise to clf split
        self.noise_rate = noise_rate
        self.noise_type = noise_type
        
        random.seed(42)
        np.random.seed(42)

        if noise_rate <= 0:
            return
        
        # num samples in dataset
        n_samples = self.__len__()
        
        # num noisy samples to generate
        n_noisy_per_class = int(noise_rate * n_samples / n_classes)
        
        # for each class add noise to noise_rate percentage of its samples 
        for c in range(n_classes):
            indeces = np.where(np.array(self.targets) == c)[0]
            noisy_samples_idx = np.random.choice(indeces, n_noisy_per_class, replace=False)            
            
            if noise_type == 'sym':
                # list of alternative class ids to choose from as a noisy target; excludes original class id
                class_ids = [i for i in range(n_classes) if i!=c]    

                for idx in noisy_samples_idx:
                    # pick a new class from the remaining 9 classes at random as noisy class for this sample
                    self.targets[idx] = random.choice(class_ids)
            elif noise_type == 'asym':
                for idx in noisy_samples_idx:
                    # use current_class+1 as the noisy class with prob noise_rate
                    current_class = self.targets[idx]
                    self.targets[idx] = np.random.choice([current_class, (current_class+1)%n_classes], p=[1-noise_rate, noise_rate])
            else:
                raise ValueError(f'Undefined noise_type: {noise_type}!') 

        return

# Usage 
# noisy_dataset = NoisyCIFAR10_Subset(root='./data', train=True, download=True, noise_type = 'sym', noise_rate=0.1, split_ratio=0.5)
# pretrain_dataset = noisy_dataset.pretrn_cifar10
