import os
import cv2
import glob
import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageOps

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


# def image_loader(path):
#     return Image.open(path)
def image_loader(path):
    return cv2.imread(path)


class Dataloader(data.Dataset):
    def __init__(
        self, data_directory, to_tensor=False, training=True,
    ):
        # directories = os.listdir(data_directory)
        # self.directories = [x[0] for x in os.walk(data_directory)]
        directories = []
        for item in os.walk(data_directory):
            if len(item[0].split("/")) > 4:  # ignore high level directories
                directories.append(item[0])
        self.directories = directories
        self.to_tensor = to_tensor
        print("Directories: ", len(self.directories))
        print("Done!")

    def __getitem__(self, index):
        directory = self.directories[index]
        print("Current directory: ", directory)
        files = glob.glob(directory + "/*.png")

        lo_images = []
        lo_q_maps = []
        for item in files:
            if item[-6:] == "HR.png":
                target_image = image_loader(item)
            elif item[-6:] == "SM.png":
                target_q_map = image_loader(item)
            elif item.split("/")[-1][:2] == "LR":
                lo_images.append(item)
            else:
                lo_q_maps.append(item)

        # 1. randomly pick an image to load
        # other strategies might be better
        # depending on solution pipeline
        lo_id = random.choice(range(len(lo_q_maps)))
        lo_image = image_loader(lo_images[lo_id])
        lo_q_map = image_loader(lo_q_maps[lo_id])

        # 2. Resize, if needed
        # print(lo_q_map.shape)
        # exit(0)

        # 3. To tensor
        if self.to_tensor:
            target_image = target_image.transpose(2, 0, 1)
            target_q_map = target_q_map.transpose(2, 0, 1)
            lo_image = lo_image.transpose(2, 0, 1)
            lo_q_map = lo_q_map.transpose(2, 0, 1)
            target_image = torch.from_numpy(target_image).unsqueeze(0).float()
            target_q_map = torch.from_numpy(target_q_map).unsqueeze(0).float()
            lo_image = torch.from_numpy(lo_image).unsqueeze(0).float()
            lo_q_map = torch.from_numpy(lo_q_map).unsqueeze(0).float()

        data_dict = {
            "input_image": lo_image,
            "input_mask": lo_q_map,
            "target_image": target_image,
            "target_mask": target_q_map,
        }

        return data_dict

    def __len__(self):
        return len(self.directories)


if __name__ == "__main__":
    train_dataset = Dataloader("./data/train", to_tensor=True)
    print(train_dataset[0]["target_mask"].shape)
