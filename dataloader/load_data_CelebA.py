import os
import glob
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random


class Dataloader_CelebA_Real(Dataset):
    def __init__(self, mode="train", 
                 dataset_path=None, 
                 annotation_path=None, 
                 target=None, 
                 sensitive=None, 
                 img_size=None):
        
        super().__init__()

        self.mode = mode
        self.dataset_path = dataset_path
        self.annotation_csv = pd.read_csv(annotation_path)
        self.images = []
        self.img_size = img_size
        self.labels_target = []
        self.labels_sensitive = []

        for i in range(len(self.annotation_csv)):
            # self.images.append(self.annotation_csv.loc[i]['image_id'][:6]+".png")
            self.images.append(self.annotation_csv.loc[i]['image_id'])
            y = int((self.annotation_csv.loc[i][target]+1)/2)
            self.labels_target.append(y)
            s = int((self.annotation_csv.loc[i][sensitive]+1)/2)
            self.labels_sensitive.append(s)
               
        self.length = len(self.images)

        self.transforms_train = transforms.Compose([transforms.RandomResizedCrop((224, 224)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
        
        self.transforms_test = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __getitem__(self, index):

        if self.mode == "train":
            image_tensor = self.transforms_train(Image.open(os.path.join(self.dataset_path, self.images[index])))
        elif self.mode == "val" or self.mode == "test":
            image_tensor = self.transforms_test(Image.open(os.path.join(self.dataset_path, self.images[index])))
        label_target = torch.tensor(self.labels_target[index])
        label_sensitive = torch.tensor(self.labels_sensitive[index])
        image_name = self.images[index]

        return image_tensor, label_target, label_sensitive, image_name
    
    def __len__(self):
        return self.length

    
class Dataloader_CelebA_Synthetic(Dataset):
    def __init__(self,
                 dataset_path_synthetic=None,
                 img_size=None,
                 synthetic_number_train_0_0=None,
                 synthetic_number_train_0_1=None,
                 synthetic_number_train_1_0=None,
                 synthetic_number_train_1_1=None):

        super().__init__()
                
        self.img_size = img_size

        self.images_0_0_sampled = []
        self.images_0_1_sampled= []
        self.images_1_0_sampled= []
        self.images_1_1_sampled = []

        synthetic_image_path_0_0 = glob.glob(os.path.join(dataset_path_synthetic, "0_0", "*.png"))
        synthetic_image_path_0_1 = glob.glob(os.path.join(dataset_path_synthetic, "0_1", "*.png"))
        synthetic_image_path_1_0 = glob.glob(os.path.join(dataset_path_synthetic, "1_0", "*.png"))
        synthetic_image_path_1_1 = glob.glob(os.path.join(dataset_path_synthetic, "1_1", "*.png"))

        self.images_0_0_sampled = random.sample(synthetic_image_path_0_0, synthetic_number_train_0_0)
        self.images_0_1_sampled = random.sample(synthetic_image_path_0_1, synthetic_number_train_0_1)
        self.images_1_0_sampled = random.sample(synthetic_image_path_1_0, synthetic_number_train_1_0)
        self.images_1_1_sampled = random.sample(synthetic_image_path_1_1, synthetic_number_train_1_1)

        self.length = len(self.images_0_0_sampled)

        print("Training Data Number (Synthetic): ", len(self.images_0_0_sampled), len(self.images_0_1_sampled), len(self.images_1_0_sampled), len(self.images_1_1_sampled))

        self.transforms_train = transforms.Compose([transforms.RandomResizedCrop((self.img_size, self.img_size)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
        
    def __getitem__(self, index):

        image_tensor_0_0 = self.transforms_train(Image.open(self.images_0_0_sampled[index]))
        image_tensor_0_1 = self.transforms_train(Image.open(self.images_0_1_sampled[index]))
        image_tensor_1_0 = self.transforms_train(Image.open(self.images_1_0_sampled[index]))
        image_tensor_1_1 = self.transforms_train(Image.open(self.images_1_1_sampled[index]))

        return image_tensor_0_0, image_tensor_0_1, image_tensor_1_0, image_tensor_1_1
    
    def __len__(self):
        return self.length


class Dataloader_CelebA_RealAndSynthetic(Dataset):
    def __init__(self, dataset_path_real=None,
                 annotation_path_real=None,
                 dataset_path_synthetic=None,
                 target=None,
                 sensitive=None,
                 img_size=None,
                 real_data_number_train=None,
                 synthetic_number_train_0_0=None,
                 synthetic_number_train_0_1=None,
                 synthetic_number_train_1_0=None,
                 synthetic_number_train_1_1=None):
        
        super().__init__()
                
        self.images = []
        self.img_size = img_size
        self.labels_target = []
        self.labels_sensitive = []

        # 1. Load the real data
        if real_data_number_train != 0:
            real_data_annotation_csv = pd.read_csv(annotation_path_real)
            for i in range(len(real_data_annotation_csv)):
                self.images.append(os.path.join(dataset_path_real, real_data_annotation_csv.loc[i]['image_id']))
                y = int((real_data_annotation_csv.loc[i][target]+1)/2)
                self.labels_target.append(y)
                s = int((real_data_annotation_csv.loc[i][sensitive]+1)/2)
                self.labels_sensitive.append(s)
            print("Training Data Number (Real): ", real_data_number_train)
            
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

        image_tensor = self.transforms_train(Image.open(self.images[index]))
        label_target = torch.tensor(self.labels_target[index])
        label_sensitive = torch.tensor(self.labels_sensitive[index])
        image_path = self.images[index]

        return image_tensor, label_target, label_sensitive, image_path
    
    def __len__(self):
        return self.length

    

if __name__ == "__main__":
    
    train_data = Dataloader_CelebA_Real(mode="train",
                                        dataset_path="/data/EECS-IoannisLab/datasets/img_align_celeba/",
                                        annotation_path="/data/home/acw717/code/Fairness_Attributes/annotations/CelebA_Train_Smiling_Male_2000_0.05.csv",
                                        target="Smiling", 
                                        sensitive="Male", 
                                        img_size=224)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4,
                                  drop_last=True)
    
    print(len(train_dataloader))
    
    for batch in train_dataloader:
        image_tensor, label_target, label_sensitive_1, label_sensitive_2, image_name = batch
        print(image_tensor.shape, label_target, label_sensitive_1, label_sensitive_2, image_name)
        break