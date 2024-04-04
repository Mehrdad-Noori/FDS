# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import pandas as pd
from PIL import Image, ImageFile
from torchvision import transforms as T
from torch.utils.data import TensorDataset, Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import rotate

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Big images
    "VLCS",
    "VLCSGenCSV",
    "PACS",
    "PACSGenCSV",
    "OfficeHome",
    "OfficeHomeGenCSV",
    "GeneralDGEvalDataset"
]


def pil_loader(path: str):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def cc_extract_and_join(path, base_dir):
     # Split the path and take the last 3 components
    #  print("\n\n++++ fixing the data path")

    #  print(f"csv path: {path}")
    #  print(f"base_dir: {base_dir}")

     extracted_parts = path.split('/')[-3:]
     # Join the new base directory with the extracted parts
     new_path = os.path.join(base_dir, *extracted_parts)
    #  print(f"new_path: {new_path}")
     return new_path


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        """
        Return: sub-dataset for specific domain
        """
        return self.datasets[index]

    def __len__(self):
        """
        Return: # of sub-datasets
        """
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,)),
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ["0", "1", "2"]


class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ["0", "1", "2"]



class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)
        self.environments = environments

        self.datasets = []
        for environment in environments:
            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224)
        self.num_classes = len(self.datasets[-1].classes)


class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["C", "L", "S", "V"]

    def __init__(self, root):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir)


class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "S"]

    def __init__(self, root):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir)



class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "R"]

    def __init__(self, root):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir)





class ImageFolderPath(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

class MultipleEnvironmentImageFolderEval(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = T.Compose([
            # T.Resize((224,224)),
            T.RandomResizedCrop(224, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.3),
            T.RandomGrayscale(),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolderPath(path, transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)
    
class MultipleEnvironmentWithGeneratedImageFolderCSV(MultipleDomainDataset):
    def __init__(self, org_root, gen_root, gen_csv_root, gen_num_per_class, gen_max_entropy, gen_random_selection, gen_all_data, gen_only_correct, gen_envs):
        super().__init__()
        environments = [f.name for f in os.scandir(org_root) if f.is_dir()]
        environments = sorted(environments)

        environments_gen = [f.name for f in os.scandir(gen_root) if f.is_dir()]
        environments_gen = sorted(environments_gen)

        environments.extend(environments_gen)

        self.environments = environments

        self.datasets = []
        final_envs = []
        for environment in environments:
            
            
            if environment in environments_gen:
                gen_env_index = environments_gen.index(environment)
                path = os.path.join(gen_root, environment)

                if gen_envs:
                    if gen_env_index in gen_envs:
                        env_dataset = DomainCSV(gen_csv_root, gen_env_index, gen_max_entropy, gen_num_per_class, new_data_dir=gen_root, transform=None, random_selection=gen_random_selection, all_data=gen_all_data, only_correct=gen_only_correct)
                        self.datasets.append(env_dataset)
                        final_envs.append(environment)

                    else:
                        print(f"+++ the generated environment {environment} (index= {gen_env_index}) is ignored! only {gen_envs} are selected!")

                else:
                    env_dataset = DomainCSV(gen_csv_root, gen_env_index, gen_max_entropy, gen_num_per_class, new_data_dir=gen_root, transform=None, random_selection=gen_random_selection, all_data=gen_all_data, only_correct=gen_only_correct)
                    self.datasets.append(env_dataset)
                    final_envs.append(environment)



            else:
                path = os.path.join(org_root, environment)
                env_dataset = ImageFolder(path)
                self.datasets.append(env_dataset)
                final_envs.append(environment)

        #  the final selected envs
        self.environments = final_envs
        self.input_shape = (3, 224, 224)
        assert self.datasets[0].classes == self.datasets[-1].classes
        self.num_classes = len(self.datasets[-1].classes)


class GeneralDGEvalDataset(MultipleEnvironmentImageFolderEval):
    CHECKPOINT_FREQ = 300
    def __init__(self, root, hparams):
        self.dir = root
        super().__init__(self.dir, [-1], hparams['data_augmentation'], hparams)

class DomainCSV(Dataset):
    def __init__(self, csv_file, domain_index, max_entropy, num_per_class, new_data_dir=None, transform=None, random_selection=False, all_data=False, only_correct=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            domain_index (int): The index of the domain for which the dataset is created.
            max_entropy (float): Maximum allowed entropy for an image to be included.
            num_per_class (int): Number of images to include per class.
            only_correct (bool, optional): If True, only include images marked as correct.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        column_names = ['Path', 'Label', 'Prediction', 'Correct', 'Entropy', 'Logit']
        
        self.data_frame = pd.read_csv(csv_file, names=column_names)
        if only_correct:

            print("+++ only correct is set to True. So, only the correctly predicted classes are used!")
            print(f"+++ total images changed from {len(self.data_frame)}")
            # Filter the dataframe to include only the images marked as "Yes" in the "Correct" column
            self.data_frame = self.data_frame[self.data_frame['Correct'] == 'Yes']
            print(f"+++ to {len(self.data_frame)} (only correct classes)")


        if new_data_dir:
            # Replace the specified part of the path

            tmp_list = list(self.data_frame['Path'])
            print(f"\n\n+++++Changing the data dir from: \n{tmp_list[0]}")

            if 'projets' in tmp_list[0]:
                self.data_frame['Path'] = self.data_frame['Path'].str.replace(r'/projets/Mnoori/original_dm/generation/final/[^/]+', new_data_dir, regex=True)
            elif "localscratch" in tmp_list[0]:
                self.data_frame['Path'] = self.data_frame['Path'].apply(lambda x: cc_extract_and_join(x, new_data_dir))
                print("++++ cc detcted!a")

            print(f"to: \n{list(self.data_frame['Path'])[0]}")

        # Extract and sort domain names
        
        domain_names = sorted(self.data_frame['Path'].apply(lambda x: x.split('/')).apply(lambda x: x[-3] if len(x) > 2 else None).unique())
        if domain_index >= len(domain_names):
            raise ValueError("Invalid domain index")
        selected_domain = domain_names[domain_index]
        print(f"Selected domain: {selected_domain}")

        # Filter and process data for the selected domain
        domain_data = self.data_frame[self.data_frame['Path'].str.contains(f'/{selected_domain}/')]

        # Extract class names, sort them, and map them to indices
        class_names = sorted(self.data_frame['Path'].apply(lambda x: x.split('/')).apply(lambda x: x[-2] if len(x) > 1 else None).unique())
        self.classes = class_names
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        print(f"Class to index mapping: {self.class_to_idx}")

        # Filter images based on random or max_entropy (both choose num_per_class)
        filtered_paths = []
        for class_name in class_names:
            class_images = domain_data[domain_data['Path'].str.contains(f'/{class_name}/')]

            if all_data:
                selected_images = class_images

            else:
            
                if random_selection:
                    # If random_selection is True, randomly select images
                    print(f"+++ will select data rondomly ({num_per_class} images) are used (not varying these selected images during different epochs)")
                    selected_images = class_images.sample(n=num_per_class)
                else:
                    # If random_selection is False, proceed with the existing logic
                    print(f"+++ will select {num_per_class} images based on entropy")
                    class_images = class_images[class_images.iloc[:, -2] < max_entropy]  # Filter by max_entropy
                    class_images = class_images.sort_values(by=class_images.columns[-2], ascending=False)  # Sort by entropy
                    selected_images = class_images.head(num_per_class)  # Select top num_per_class images

            filtered_paths.extend(selected_images['Path'].tolist())

        self.image_paths = filtered_paths
        self.labels = [self.class_to_idx[path.split('/')[-2]] for path in self.image_paths]
        self.entropies = domain_data.iloc[:, -2].loc[domain_data['Path'].isin(self.image_paths)].tolist()

        # self.root_dir = root_dir
        self.transform = transform
        print(f"Total images loaded: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # img_path = os.path.join(self.root_dir, self.image_paths[idx])
        img_path = self.image_paths[idx]
        image = pil_loader(img_path)  # Use pil_loader to load the image

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        # entropy = self.entropies[idx]

        return image, label

class PACSGenCSV(MultipleEnvironmentWithGeneratedImageFolderCSV):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "S", "Z1", "Z2", "Z3"]

    def __init__(self, org_root, gen_root, csv_root, gen_num_per_class, gen_max_entropy, test_env, gen_random_selection=False, gen_all_data=False, gen_only_correct=False, gen_envs=None):
        self.dir = os.path.join(org_root, "PACS/")
        super().__init__(self.dir, gen_root, csv_root, gen_num_per_class, gen_max_entropy, gen_random_selection, gen_all_data, gen_only_correct=gen_only_correct, gen_envs=gen_envs)

class VLCSGenCSV(MultipleEnvironmentWithGeneratedImageFolderCSV):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["C", "L", "S", "V", "Z1", "Z2", "Z3"]

    def __init__(self, root, gen_root, csv_root, gen_num_per_class, gen_max_entropy, test_env, gen_random_selection=False, gen_all_data=False, gen_only_correct=False, gen_envs=None):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, gen_root, csv_root, gen_num_per_class, gen_max_entropy, gen_random_selection, gen_all_data, gen_only_correct=gen_only_correct, gen_envs=gen_envs)

class OfficeHomeGenCSV(MultipleEnvironmentWithGeneratedImageFolderCSV):
    CHECKPOINT_FREQ = 200
    ENVIRONMENTS = ["A", "C", "P", "R", "Z1", "Z2", "Z3"]

    def __init__(self, root, gen_root, csv_root, gen_num_per_class, gen_max_entropy, test_env, gen_random_selection=False, gen_all_data=False, gen_only_correct=False, gen_envs=None):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, gen_root, csv_root, gen_num_per_class, gen_max_entropy, gen_random_selection, gen_all_data, gen_only_correct=gen_only_correct, gen_envs=gen_envs)

