import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class Dataloader_UTKFace_Real(Dataset):
    def __init__(self, 
                 mode="train", 
                 dataset_path=None, 
                 img_size=None):
        super(Dataloader_UTKFace_Real, self).__init__()

        self.mode = mode
        self.dataset_path = dataset_path
        self.images = []
        self.img_size = img_size
        self.labels_target = []
        self.labels_sensitive = []

        all_images_path = glob.glob(os.path.join(dataset_path, "*.jpg"))

        for filename in all_images_path:
            self.images.append(filename)
            parts = filename.split("/")[-1].split("_")
            age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            self.labels_target.append(int(gender))
            if race == 0:
                self.labels_sensitive.append(int(1))
            elif race in [1, 2, 3, 4]:
                self.labels_sensitive.append(int(0))

        print("UTKFace Data Number: ", len(self.images))

        self.length = len(self.images)

        self.transforms_train = transforms.Compose([transforms.RandomResizedCrop((self.img_size, self.img_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
        
        self.transforms_test = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __getitem__(self, index):

        if self.mode == "train":
            image_tensor = self.transforms_train(Image.open(self.images[index]))
        elif self.mode == "test":
            image_tensor = self.transforms_test(Image.open(self.images[index]))
        label_target = torch.tensor(self.labels_target[index])
        label_sensitive = torch.tensor(self.labels_sensitive[index])
        image_name = self.images[index]

        return image_tensor, label_target, label_sensitive, image_name
    
    def __len__(self):
        return self.length


class Dataloader_UTKFace_Synthetic(Dataset):
    def __init__(self, 
                 img_size=None,
                 dataset_path_synthetic=None,
                 synthetic_number_train_0_0=None,
                 synthetic_number_train_0_1=None,
                 synthetic_number_train_1_0=None,
                 synthetic_number_train_1_1=None):
        super().__init__()

        self.images = []
        self.img_size = img_size
        self.labels_target = []
        self.labels_sensitive = []

        synthetic_image_path_0_0 = glob.glob(os.path.join(dataset_path_synthetic, "0_0", "*.png"))
        synthetic_image_path_0_1 = glob.glob(os.path.join(dataset_path_synthetic, "0_1", "*.png"))
        synthetic_image_path_1_0 = glob.glob(os.path.join(dataset_path_synthetic, "1_0", "*.png"))
        synthetic_image_path_1_1 = glob.glob(os.path.join(dataset_path_synthetic, "1_1", "*.png"))

        images_0_0_sampled = random.sample(synthetic_image_path_0_0, synthetic_number_train_0_0)
        images_0_1_sampled = random.sample(synthetic_image_path_0_1, synthetic_number_train_0_1)
        images_1_0_sampled = random.sample(synthetic_image_path_1_0, synthetic_number_train_1_0)
        images_1_1_sampled = random.sample(synthetic_image_path_1_1, synthetic_number_train_1_1)

        for synthetic_image_path_ in [images_0_0_sampled, images_0_1_sampled, images_1_0_sampled, images_1_1_sampled]:
            for _image_path in synthetic_image_path_:
                self.images.append(_image_path)
                _data_label = _image_path.split('/')[-2]
                y = int(_data_label.split('_')[0])
                self.labels_target.append(y)
                s = int(_data_label.split('_')[1])
                self.labels_sensitive.append(s)
        print("Training Data Number (Synthetic): ", len(images_0_0_sampled), len(images_0_1_sampled), len(images_1_0_sampled), len(images_1_1_sampled))

        self.length = len(self.images)

        self.transforms_train = transforms.Compose([transforms.RandomResizedCrop((self.img_size, self.img_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
        
        self.transforms_test = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __getitem__(self, index):

        if self.mode == "train":
            image_tensor = self.transforms_train(Image.open(self.images[index]))
        elif self.mode == "test":
            image_tensor = self.transforms_test(Image.open(self.images[index]))
        label_target = torch.tensor(self.labels_target[index])
        label_sensitive = torch.tensor(self.labels_sensitive[index])
        image_name = self.images[index]

        return image_tensor, label_target, label_sensitive, image_name
    
    def __len__(self):
        return self.length


class Dataloader_UTKFace_RealAndSynthetic(Dataset):
    def __init__(self, 
                 mode="train",
                 img_size=None,
                 dataset_path_real=None,
                 dataset_path_synthetic=None,
                 real_data_number_train=None,
                 synthetic_number_train_0_0=None,
                 synthetic_number_train_0_1=None,
                 synthetic_number_train_1_0=None,
                 synthetic_number_train_1_1=None):
        super().__init__()

        self.mode = mode
        self.dataset_path_real = dataset_path_real
        self.images = []
        self.img_size = img_size
        self.labels_target = []
        self.labels_sensitive = []

        # 1. Load the real data
        if real_data_number_train != 0:
            all_images_path = glob.glob(os.path.join(dataset_path_real, "*.jpg"))

            for filename in all_images_path:
                self.images.append(filename)
                parts = filename.split("/")[-1].split("_")
                age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
                self.labels_target.append(int(gender))
                if race == 0:
                    self.labels_sensitive.append(int(1))
                elif race in [1, 2, 3, 4]:
                    self.labels_sensitive.append(int(0))

            print("UTKFace Data Number: ", len(self.images))

        # 2. Load the synthetic data
        if synthetic_number_train_0_0 != 0 or synthetic_number_train_0_1 != 0:

            synthetic_image_path_0_0 = glob.glob(os.path.join(dataset_path_synthetic, "0_0", "*.png"))
            synthetic_image_path_0_1 = glob.glob(os.path.join(dataset_path_synthetic, "0_1", "*.png"))
            synthetic_image_path_1_0 = glob.glob(os.path.join(dataset_path_synthetic, "1_0", "*.png"))
            synthetic_image_path_1_1 = glob.glob(os.path.join(dataset_path_synthetic, "1_1", "*.png"))

            images_0_0_sampled = random.sample(synthetic_image_path_0_0, synthetic_number_train_0_0)
            images_0_1_sampled = random.sample(synthetic_image_path_0_1, synthetic_number_train_0_1)
            images_1_0_sampled = random.sample(synthetic_image_path_1_0, synthetic_number_train_1_0)
            images_1_1_sampled = random.sample(synthetic_image_path_1_1, synthetic_number_train_1_1)

            for synthetic_image_path_ in [images_0_0_sampled, images_0_1_sampled, images_1_0_sampled, images_1_1_sampled]:
                for _image_path in synthetic_image_path_:
                    self.images.append(_image_path)
                    _data_label = _image_path.split('/')[-2]
                    y = int(_data_label.split('_')[0])
                    self.labels_target.append(y)
                    s = int(_data_label.split('_')[1])
                    self.labels_sensitive.append(s)
            print("Training Data Number (Synthetic): ", len(images_0_0_sampled), len(images_0_1_sampled), len(images_1_0_sampled), len(images_1_1_sampled))

        self.length = len(self.images)

        self.transforms_train = transforms.Compose([transforms.RandomResizedCrop((self.img_size, self.img_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
        
        self.transforms_test = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __getitem__(self, index):

        if self.mode == "train":
            image_tensor = self.transforms_train(Image.open(self.images[index]))
        elif self.mode == "test":
            image_tensor = self.transforms_test(Image.open(self.images[index]))
        label_target = torch.tensor(self.labels_target[index])
        label_sensitive = torch.tensor(self.labels_sensitive[index])
        image_name = self.images[index]

        return image_tensor, label_target, label_sensitive, image_name
    
    def __len__(self):
        return self.length


if __name__ == "__main__":
    pass