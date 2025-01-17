import os
import cv2
import glob
import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import skimage
from skimage import io
from skimage.transform import rescale
from sklearn.utils import shuffle

from image_augmentations import select_stack_augmentation

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
compose = transforms.Compose([normalize])


class ProbaVLoader(data.Dataset):
    def __init__(
        self,
        data_directory,
        to_tensor=False,
        mode="train",
        interpolate=True,
        augment=True,
    ):
        # directories = os.listdir(data_directory)
        # self.directories = [x[0] for x in os.walk(data_directory)]
        self.mode = mode
        self.interpolate = interpolate
        self.augment = augment
        directories = []
        for item in os.walk(data_directory):
            if len(item[0].split("/")) > 4:  # ignore high level directories
                directories.append(item[0])
        self.directories = directories
        self.to_tensor = to_tensor
        print("Directories: ", len(self.directories))
        print(data_directory, " Done!")

    def image_loader(self, path):
        return cv2.imread(path)

    def image_loader_super(self, path):
        image = cv2.imread(path)
        if self.augment and random.random() > 0.5:
            image = select_stack_augmentation(
                [image],
                saturation=True,
                brightness=True,
                contrast=True,
                sharpness=True,
                hue=True,
                gamma=True,
            )
        return cv2.resize(image, (384, 384), interpolation=cv2.INTER_CUBIC)

    def __getitem__(self, index):
        directory = self.directories[index]
        files = glob.glob(directory + "/*.png")

        lo_images = []
        lo_q_maps = []
        for item in files:
            if item[-6:] == "HR.png":
                target_image = self.image_loader(item)
            elif item[-6:] == "SM.png":
                target_q_map = self.image_loader(item)
            elif item.split("/")[-1][:2] == "LR":
                lo_images.append(item)
            else:
                lo_q_maps.append(item)

        # 1. randomly pick an image to load
        # other strategies might be better
        # depending on solution pipeline
        lo_id = random.choice(range(len(lo_q_maps)))
        lo_image = self.image_loader(lo_images[lo_id])
        lo_q_map = self.image_loader(lo_q_maps[lo_id])

        # 2. Apply transormations; Resize, etc, if needed
        # transformed_sample = tsfrm(sample)  # <- example

        lo_image = lo_image / 255.0
        if self.mode == "train":
            target_image = target_image / 255.0
        # lo_q_map = lo_q_map/255.0
        # target_q_map = target_q_map/255.0

        lo_q_map[lo_q_map > 0] = 1
        target_q_map[target_q_map > 0] = 1

        # 3. To tensor
        if self.to_tensor:
            if self.mode == "train":
                target_image = target_image.transpose(2, 0, 1)
                target_image = torch.from_numpy(target_image).float()

            target_q_map = target_q_map.transpose(2, 0, 1)
            target_q_map = torch.from_numpy(target_q_map).float()
            lo_image = lo_image.transpose(2, 0, 1)
            lo_q_map = lo_q_map.transpose(2, 0, 1)
            lo_image = torch.from_numpy(lo_image).float()
            lo_q_map = torch.from_numpy(lo_q_map).float()

        if self.mode == "train":
            data_dict = {
                "input_image": lo_image,
                "input_mask": lo_q_map,
                "target_image": target_image,
                "target_mask": target_q_map,
            }
        else:
            data_dict = {
                "input_image": lo_image,
                "input_mask": lo_q_map,
                "target_mask": target_q_map,
            }

        return data_dict

    def __len__(self):
        return len(self.directories)


def process_image_batch(image_files_list, process_function, interpolate=False):
    all_images = []
    for image_file in image_files_list:
        image = process_function(image_file)
        all_images.append(image)

    return all_images


def tensorize_batch(np_array_list):
    all_tensors = []
    for np_array in np_array_list:
        torch_array = np_array.transpose(2, 0, 1) / 255.0
        torch_array = torch.from_numpy(torch_array).float()
        all_tensors.append(torch_array)
    return all_tensors


class ProbaVLoaderRNN(ProbaVLoader):
    def __getitem__(self, index):
        directory = self.directories[index]
        files = glob.glob(directory + "/*.png")

        lo_images = []
        lo_q_maps = []
        for item in files:
            if item[-6:] == "HR.png":
                target_image = self.image_loader(item)
            elif item[-6:] == "SM.png":
                target_q_map = self.image_loader(item)
            elif item.split("/")[-1][:2] == "LR":
                lo_images.append(item)
            else:
                lo_q_maps.append(item)

        # 1. shuffle the lo_res and map
        # lo_images, _ = shuffle(lo_images, lo_q_maps, random_state=0)
        # load images
        if self.interpolate:
            lo_images_im = process_image_batch(lo_images, self.image_loader_super)
        else:
            lo_images_im = process_image_batch(lo_images, self.image_loader)

        # lo_image = lo_images_im / 255.0
        if self.mode == "train":
            target_image = target_image / 255.0

        target_q_map[target_q_map > 0] = 1

        if self.mode == "train":
            target_image = target_image.transpose(2, 0, 1)
            target_image = torch.from_numpy(target_image).float()

        target_q_map = target_q_map.transpose(2, 0, 1)
        target_q_map = torch.from_numpy(target_q_map).float()
        lo_image = tensorize_batch(lo_images_im)

        if self.mode == "train":
            data_dict = {
                "input_image": lo_image,
                "target_image": target_image,
                "target_mask": target_q_map,
                "directory": directory.split("/")[-1],
            }
        else:
            data_dict = {
                "input_image": lo_image,
                "target_mask": target_q_map,
                "directory": directory,
            }
        return data_dict

    def __len__(self):
        return len(self.directories)


if __name__ == "__main__":
    train_dataset = ProbaVLoaderRNN("./data/train", to_tensor=True)
    print(train_dataset[0]["directory"])
