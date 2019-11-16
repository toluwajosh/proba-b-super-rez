"""
quick train
"""
import tqdm
import torch
import numpy as np
import torchvision.models as torch_models
import torchvision.transforms as transforms

from torch import nn
from torchsummary import summary

from data.loader import ProbaVLoader
from models.simple_autoencoder import autoencoder
from models.resnet import resnet18_base, resnet50_base


# hyperparameters
BATCH_SIZE = 16
WORKERS = 8
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
SUMMARY = True


train_dataloader = ProbaVLoader("./data/train", to_tensor=True)
train_data = torch.utils.data.DataLoader(
        train_dataloader,
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=WORKERS, pin_memory=True)

# # 1. we can use resnet this way
# resnet18 = torch_models.resnet18()
# print(resnet18.layer1)
# exit(0)



# model = autoencoder().cuda()
model = resnet50_base(pretrained=True).cuda()
if SUMMARY:
    summary(model, (3,128,128))
# exit(0)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

for epoch in range(NUM_EPOCHS):
    losses = []
    for data in train_dataloader:
        img = data["input_image"].cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        losses.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, NUM_EPOCHS, np.mean(losses)))
torch.save(model.state_dict(), "./conv_autoencoder.pth")

