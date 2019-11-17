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

# private libs
from data.loader import ProbaVLoader
from models.simple_autoencoder import autoencoder
from models.resnet import resnet18_AE, resnet50_AE


# hyperparameters
BATCH_SIZE = 2
WORKERS = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 2000  # since each data point has at least 19 input samples
SUMMARY = False
PRETRAINED = False
CHECKPOINT_PATH = "./checkpoints/checkpoint.ckpt"


train_dataloader = ProbaVLoader("./data/train", to_tensor=True)
train_data = torch.utils.data.DataLoader(
    train_dataloader,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
)


# model = autoencoder().cuda()
model = resnet50_AE(pretrained=PRETRAINED).cuda()
if SUMMARY:
    summary(model, (3, 128, 128))
# exit(0)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # , weight_decay=1e-5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.8, patience=3, verbose=True, min_lr=1e-8
)

# load existing model
try:
    # check if checkpoints file of weights file
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_chk = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("\n\nModel Loaded; ", CHECKPOINT_PATH)
except Exception as e:
    print("\n\nModel not loaded; ", CHECKPOINT_PATH)
    print("Exception: ", e)
    epoch_chk = 0

model.train()
for epoch in range(NUM_EPOCHS):
    if epoch < epoch_chk:
        continue
    losses = []
    for data in train_dataloader:
        img = data["input_image"].cuda()
        target = data["target_image"].cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output.mul(255.0), target.mul(255.0))
        losses.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step(np.mean(losses))
    # ===================log========================
    print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, NUM_EPOCHS, np.mean(losses)))
    # save checkpoint
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": np.mean(losses),
        },
        CHECKPOINT_PATH,
    )

torch.save(model.state_dict(), "./proba_v.weights")

