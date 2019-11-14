"""
quick train
"""
import torch
from torch import nn
from data.loader import Dataloader

from models.simple_autoencoder import autoencoder

train_dataloader = Dataloader("./data/train", to_tensor=True)


# hyper parameters
learning_rate = 0.0001
num_epochs = 50
batch_size = 2

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in train_dataloader:
        img = data["input_image"]
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print("epoch [{}/{}], loss:{:.4f}".format(epoch + 1, num_epochs, loss.data[0]))
torch.save(model.state_dict(), "./conv_autoencoder.pth")

