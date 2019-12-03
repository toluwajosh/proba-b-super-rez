"""
quick train
"""
import os
import torch
import time
import logging
import numpy as np
import torchvision.models as torch_models
import torchvision.transforms as transforms

from tqdm import tqdm
from torch import nn
from torchsummary import summary

# private libs
from data.loader import ProbaVLoaderRNN
from models.simple_autoencoder import autoencoder
from models.resnet import resnet18_AE, resnet50_AE
from models.resnet_rnn import resnet50_AERNN
from losses import ProbaVLoss, ProbaVEval

torch.set_num_threads(10)

# hyperparameters
BATCH_SIZE = 1
WORKERS = 10
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100  # since each data point has at least 19 input samples
SUMMARY = False
PRETRAINED = True
CHECKPOINT_PATH = "./checkpoints/checkpoint_rnn.ckpt"
USE_MASK = True
ACCUMULATE = 1
CHECKPOINT_INTERVAL = 1

# log parameters
human_time = str(time.asctime()).replace(" ", "_").replace(":", "")
log_path = "./logs/{}_rnn.log".format(human_time)
# create file if it does not exist
logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Model Hyperparameters --------------- >")
logging.info("BATCH_SIZE: {}".format(BATCH_SIZE))
logging.info("LEARNING_RATE: {}".format(LEARNING_RATE))
logging.info("ACCUMULATE: {}".format(ACCUMULATE))


train_dataloader = ProbaVLoaderRNN("./data/train", to_tensor=True)
train_data = torch.utils.data.DataLoader(
    train_dataloader,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
)

valid_dataloader = ProbaVLoaderRNN("./data/valid", to_tensor=True)
valid_data = torch.utils.data.DataLoader(
    valid_dataloader,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
)


model = resnet50_AERNN(pretrained=PRETRAINED).cuda()
logging.info(str(model))
if SUMMARY:
    summary(model, (3, 128, 128))

# criterion = nn.MSELoss()
criterion = ProbaVLoss(mask_flag=USE_MASK)
eval_criterion = ProbaVEval(mask_flag=USE_MASK)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # , weight_decay=1e-5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.9, patience=60, verbose=True, min_lr=1e-8
)
epoch_chk = 0

# load existing model
try:
    # check if checkpoints file of weights file
    checkpoint = torch.load(CHECKPOINT_PATH)

    ## partial loading of model dict
    # pretrained_dict = checkpoint["model_state_dict"]
    # model_dict = model.state_dict()
    # pretrained_dict = {
    #     k: v
    #     for k, v in pretrained_dict.items()
    #     if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)
    # }
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict, strict=False)

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_chk = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print("\n\nModel Loaded; ", CHECKPOINT_PATH)
except Exception as e:
    print("\n\nModel not loaded; ", CHECKPOINT_PATH)
    print("Exception: ", e)

for epoch in range(NUM_EPOCHS):
    if epoch < epoch_chk:
        continue
    losses = []
    model.train()
    with tqdm(total=len(train_data)) as pbar:
        for i, data in enumerate(train_data):
            img = data["input_image"]
            target = data["target_image"].cuda()
            target_mask = data["target_mask"].cuda()

            samples_num = len(img)
            # ===================forward=====================
            output, hidden_ith = model(img[0].cuda())  # first sample
            # # calculate loss
            # loss = criterion(output, target, target_mask)
            # losses.append(loss.item())
            # ===================backward====================
            # optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            # optimizer.step()
            for ith in range(1, samples_num - 1):
                img_ith = img[ith]
                output, hidden_ith = model(img_ith.cuda(), hidden_ith)
                # # iterative refinement
                # loss = criterion(output, target, target_mask)
                # losses.append(loss.item())
                # optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                # optimizer.step()

            output, hidden_ith = model(img[-1].cuda(), hidden_ith)
            # iterative refinement, final
            loss = criterion(output, target, target_mask)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                "Epoch: {:5d}; Loss: {:8.5f}".format(epoch + 1, np.mean(losses))
            )
            pbar.update()
    scheduler.step(np.mean(losses))
    # ===================log========================
    logging.info(
        "Epoch [{}/{}], Training Loss:{:.4f}".format(
            epoch + 1, NUM_EPOCHS, np.mean(losses)
        )
    )

    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        # evaluate model
        model.eval()
        with tqdm(total=len(valid_data)) as pbar_eval:
            error_list = []
            for data in valid_data:
                img = data["input_image"]
                target = data["target_image"].cuda()
                target_mask = data["target_mask"].cuda()
                samples_num = len(img)
                output, hidden_ith = model(img[0].cuda())  # first sample
                for ith in range(1, samples_num):
                    img_ith = img[ith].cuda()
                    output, hidden_ith = model(img_ith, hidden_ith)
                error = eval_criterion(output, target, target_mask)
                error_list.append(error.item())

                pbar_eval.set_description(
                    "Evaluation score: {:8.5f}".format(np.mean(error_list))
                )
                pbar_eval.update()
        logging.info("Evaluation Score:{:.4f}".format(np.mean(error_list)))
        # save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.mean(losses),
            },
            CHECKPOINT_PATH,
        )

torch.save(model.state_dict(), "./proba_v.weights")

