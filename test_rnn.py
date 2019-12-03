"""
quick train
"""
import torch
import numpy as np
import torchvision.models as torch_models
import torchvision.transforms as transforms

from tqdm import tqdm
from torch import nn
from torchsummary import summary
from skimage import io
import cv2

# private libs
from data.loader import ProbaVLoaderRNN
from models.simple_autoencoder import autoencoder
from models.resnet_rnn import resnet50_AERNN
from losses import ProbaVLoss, ProbaVEval


# hyperparameters
BATCH_SIZE = 2
WORKERS = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 500  # since each data point has at least 19 input samples
SUMMARY = False
PRETRAINED = False
CHECKPOINT_PATH = "./checkpoints/checkpoint_rnn.ckpt"
USE_MASK = True


test_dataloader = ProbaVLoaderRNN("./data/valid", to_tensor=True, mode="train")
test_data = torch.utils.data.DataLoader(
    test_dataloader, batch_size=1, shuffle=True, num_workers=WORKERS, pin_memory=True,
)


# model = autoencoder().cuda()
model = resnet50_AERNN(pretrained=PRETRAINED).cuda()
if SUMMARY:
    summary(model, (3, 128, 128))

criterion = ProbaVEval()

# load existing model
try:
    # check if checkpoints file of weights file
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    epoch_chk = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("\n\nModel Loaded; ", CHECKPOINT_PATH)
except Exception as e:
    print("\n\nModel not loaded; ", CHECKPOINT_PATH)
    print("Exception: ", e)
    epoch_chk = 0

for epoch in range(NUM_EPOCHS):
    if epoch < epoch_chk:
        continue
    losses = []
    model.train()
    with tqdm(total=len(test_data)) as pbar:
        for data in test_data:
            img = data["input_image"]
            target_image = data["target_image"].cuda()
            target_mask = data["target_mask"].cuda()
            samples_num = len(img)
            output, hidden_ith = model(img[0].cuda())  # first sample
            for ith in range(1, samples_num):
                img_ith = img[ith].cuda()
                output, hidden_ith = model(img_ith, hidden_ith)
            image = torch.nn.functional.interpolate(img[0].cuda(), output.shape[2:])
            loss = criterion(image, target_image, target_mask)
            losses.append(loss.item())
            output_image = output[0].cpu().detach().numpy()
            output_image = np.transpose(output_image, [1, 2, 0])
            target_image = output[0].cpu().detach().numpy()
            target_image = np.transpose(target_image, [1, 2, 0])
            cv2.imwrite("output_image.jpg", output_image * 255)
            cv2.imwrite("target_image.jpg", target_image * 255)
            exit(0)
            # io.imsave(output_image, "output_image.jpg")
            # print("output.shape: ", output_image.shape)
            pbar.set_description("Evaluation")
            pbar.update()
    break
print("Final Error: ", np.mean(losses))
