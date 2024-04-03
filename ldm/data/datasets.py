# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from torch.utils.data import ConcatDataset


ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class DictDataset:
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x,y = self.dataset[index]
        out_dict = {}
        out_dict["y"] = torch.tensor(y, dtype=torch.long)
        return x,out_dict

    def __len__(self):
        return len(self.dataset)

class MultipleDomainDataset:
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, data_augmentation = False, image_size = 64, mean = (1,1,1), std = (1,1,1), **kwargs):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std)
        ]) 
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if data_augmentation and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, image_size, image_size,)
        self.num_classes = len(self.datasets[-1].classes)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, **hparams): 
        self.dir = os.path.join(root, "PACS/")
        # hparams["mean"] = (0.5085, 0.4832, 0.4396)
        # hparams["std"] = (0.2749, 0.2665, 0.2841)
        hparams["mean"] = (0.485, 0.456, 0.406)
        hparams["std"] = (0.229, 0.224, 0.225)
        super().__init__(self.dir, test_envs, **hparams)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, **hparams):
        self.dir = os.path.join(root, "VLCS/")
        hparams["mean"] = (0.485, 0.456, 0.406)
        hparams["std"] = (0.229, 0.224, 0.225)
        super().__init__(self.dir, test_envs, **hparams)



class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, **hparams):
        self.dir = os.path.join(root, "office_home/")
        hparams["mean"] = (0.485, 0.456, 0.406)
        hparams["std"] = (0.229, 0.224, 0.225)
        super().__init__(self.dir, test_envs, **hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, **hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        hparams["mean"] = (0.485, 0.456, 0.406)
        hparams["std"] = (0.229, 0.224, 0.225)
        super().__init__(self.dir, test_envs, **hparams)


class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, **hparams):
        self.dir = os.path.join(root, "domain_net/")
        hparams["mean"] = (0.485, 0.456, 0.406)
        hparams["std"] = (0.229, 0.224, 0.225)
        super().__init__(self.dir, test_envs, **hparams)








class TextOneDomain:
    def __init__(self, name, *args, **kwargs):
        if name=="PACS":
            data = PACS(*args, **kwargs)

        elif name=="VLCS":
            data = VLCS(*args, **kwargs)

        elif name=="OH":
            data = OfficeHome(*args, **kwargs)

        elif name=="TI":
            data = TerraIncognita(*args, **kwargs)

        elif name=="DN":
            data = DomainNet(*args, **kwargs)

        self.dataset = data[kwargs["test_envs"][0]]

    @staticmethod
    def extract_names(path):
        # Split the path into its components
        parts = path.split(os.sep)

        # Extract the DOMAIN_NAME and CLASS_NAME using the known structure of the path
        domain_name = parts[-3]  # Three directories up from the filename
        class_name = parts[-2]  # Two directories up from the filename

        return domain_name, class_name

    def __getitem__(self, index):
        x,y = self.dataset[index]
        domain_name, class_name = self.extract_names(self.dataset.samples[index][0])
        return {"image":x.permute(1,2,0) ,"txt":f"{domain_name}, {class_name}","y":y}

    def __len__(self):
        return len(self.dataset)


class CombinedTextDomains:
    def __init__(self, *args, **kwargs):
        test_envs = kwargs['test_envs']
        datasets = [TextOneDomain(*args, name=kwargs['dataset_name'], root=kwargs['root'], test_envs=[i], data_augmentation=kwargs['data_augmentation'], image_size=kwargs['image_size']) for i in test_envs]
        self.dataset = ConcatDataset(datasets)


    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)



