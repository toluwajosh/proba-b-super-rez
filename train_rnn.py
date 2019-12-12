"""
quick train
"""
import argparse
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
from score_csv import BaseScore

parser = argparse.ArgumentParser(
    description="Train model for predicting Proba-v Super resoltion image"
)
parser.add_argument(
    "--checkpoint_path",
    default="./checkpoints/checkpoint_rnn_final.ckpt",
    metavar="'./path/to/checkpoint/file/'",
    help="Path to the checkpoint file",
)
parser.add_argument(
    "--learning_rate",
    required=False,
    default=0.0001,
    metavar="<0.0001>",
    help="supply the learning rate for the training",
)
parser.add_argument(
    "--num_epochs",
    required=False,
    default=1000,
    metavar="<1000>",
    help="The number of epochs for training",
)
parser.add_argument(
    "--workers",
    required=False,
    default=10,
    metavar="<10>",
    help="The number of process workers",
)
parser.add_argument(
    "--chekcpoints_interval",
    required=False,
    default=1,
    metavar="<10>",
    help="Intervals for saving checkpoints",
)
parser.add_argument(
    "--load_partial", action="store_true", help="Load partial weights",
)

args = parser.parse_args()

# hyperparameters
CHECKPOINT_PATH = args.checkpoint_path
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
WORKERS = args.workers
LOAD_PARTIAL = args.load_partial
CHECKPOINT_INTERVAL = args.chekcpoints_interval
BATCH_SIZE = 1  # solution pipeline only works with batch size 1
SUMMARY = False  # show model summary
PRETRAINED = True  # load pretrained model (for base model)
USE_MASK = True  # use dataset masks
ACCUMULATE = 1  # accumulate gradients

torch.set_num_threads(WORKERS)

# log parameters
human_time = str(time.asctime()).replace(" ", "_").replace(":", "")
log_path = "./logs/{}_rnn.log".format(human_time)
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

base_scores = BaseScore()


model = resnet50_AERNN(pretrained=PRETRAINED).cuda()
logging.info(str(model))

if SUMMARY:
    summary(model, (3, 128, 128))

criterion = ProbaVLoss(mask_flag=USE_MASK, ssim_weight=0.1)
eval_criterion = ProbaVEval(mask_flag=USE_MASK)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.9, patience=3, verbose=True, min_lr=1e-8
)
epoch_chk = 0

# load existing model
try:
    # check if checkpoints file of weights file
    checkpoint = torch.load(CHECKPOINT_PATH)

    if LOAD_PARTIAL:
        print("Loading Partial......")
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
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_chk = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print("\n\nModel Loaded; ", CHECKPOINT_PATH)
    print("Learning Rate: ", optimizer.param_groups[0]["lr"])
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
            # baseline = base_scores[data["directory"][0]]
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
                    baseline = base_scores[data["directory"][0]]
                    error = eval_criterion(output, target, target_mask, baseline=baseline)
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
