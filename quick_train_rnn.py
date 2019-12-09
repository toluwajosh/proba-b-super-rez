"""
quick train
"""
import logging
import os
import time

import numpy as np
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
from torch import nn
from torchsummary import summary
from tqdm import tqdm

# private libs
from data.loader import ProbaVLoaderRNN
from losses import ProbaVEval, ProbaVLoss
from models.resnet import resnet18_AE, resnet50_AE
from models.resnet_rnn import resnet50_AERNN
from models.simple_autoencoder import autoencoder

torch.set_num_threads(10)

# hyperparameters
BATCH_SIZE = 1
WORKERS = 10
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000  # since each data point has at least 19 input samples
SUMMARY = True
PRETRAINED = True
CHECKPOINT_PATH = "./checkpoints/checkpoint_conv_lstm.ckpt"
USE_MASK = True
ACCUMULATE = 1
CHECKPOINT_INTERVAL = 1
load_partial = True

# log parameters
human_time = str(time.asctime()).replace(" ", "_").replace(":", "")
log_path = "./logs/{}_rnn.log".format(human_time)
# create file if it does not exist
# logging.basicConfig(
#     level=logging.INFO,
#     filename=log_path,
#     filemode="w",
#     format="%(asctime)s - %(levelname)s - %(message)s",
# )
# logging.info("Model Hyperparameters --------------- >")
# logging.info("BATCH_SIZE: {}".format(BATCH_SIZE))
# logging.info("LEARNING_RATE: {}".format(LEARNING_RATE))
# logging.info("ACCUMULATE: {}".format(ACCUMULATE))


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
# if SUMMARY:
#     logging.info(str(model))
#     # summary(model, (3, 128, 128))

# criterion = nn.MSELoss()
criterion = ProbaVLoss(mask_flag=USE_MASK, ssim_weight=0.1)
eval_criterion = ProbaVEval(mask_flag=USE_MASK)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)  # , weight_decay=1e-5
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.9, patience=3, verbose=True, min_lr=1e-8
)
epoch_chk = 0

# load existing model
try:
    # check if checkpoints file of weights file
    checkpoint = torch.load(CHECKPOINT_PATH)

    if load_partial:
        # partial loading of model dict
        pretrained_dict = checkpoint["model_state_dict"]
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_chk = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print("\n\nModel Loaded; ", CHECKPOINT_PATH)
    print("Learning Rate: ", optimizer.param_groups[0]['lr'])
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
            img_prev = img[0].cuda()
            for ith in range(1, samples_num):
                img_ith = img[ith]
                output, hidden_ith = model(img_ith.cuda(), img_prev, hidden_ith)
                img_prev = img_ith.cuda()
            # calculate loss
            loss = criterion(output, target, target_mask)
            losses.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                "Epoch: {:5d}; Loss: {:.10f}".format(epoch + 1, np.mean(losses))
            )
            pbar.update()
    # scheduler.step(np.mean(losses))
    # ===================log========================
    logging.info(
        "Epoch [{}/{}], Training Loss:{:.10f}".format(
            epoch + 1, NUM_EPOCHS, np.mean(losses)
        )
    )

    torch.cuda.empty_cache()
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        # evaluate model
        model.eval()
        with torch.no_grad():
            with tqdm(total=len(valid_data)) as pbar_eval:
                error_list = []
                for data in valid_data:
                    image = data["input_image"]
                    target = data["target_image"].cuda()
                    target_mask = data["target_mask"].cuda()
                    samples_num = len(image)
                    output, hidden_ith = model(image[0].cuda())  # first sample
                    img_prev = image[0].cuda()
                    for ith in range(1, samples_num):
                        img_ith = image[ith]
                        output, hidden_ith = model(img_ith.cuda(), img_prev, hidden_ith)
                        img_prev = img_ith.cuda()
                    error = eval_criterion(output, target, target_mask)
                    error_list.append(error.item())

                    pbar_eval.set_description(
                        "Evaluation score: {:.10f}".format(np.mean(error_list))
                    )
                    pbar_eval.update()
        logging.info("Evaluation Score:{:.4f}".format(np.mean(error_list)))
        # save checkpoint
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.mean(error_list),
            },
            CHECKPOINT_PATH,
        )
    scheduler.step(np.mean(error_list))

torch.save(model.state_dict(), "./proba_v.weights")
