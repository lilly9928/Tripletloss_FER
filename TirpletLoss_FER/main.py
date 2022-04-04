import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import os
from imagedata import ImageData
from torchsummary import summary
from transformNetwork import Network, init_weights
from tripletloss import TripletLoss

torch.manual_seed(300)
np.random.seed(300)
random.seed(300)

batch_size = 8
epochs =100

train_df = pd.read_csv("C:/Users/1315/Desktop/data/ck_test.csv")
test_df = pd.read_csv("C:/Users/1315/Desktop/data/ck_val.csv")

train_ds = ImageData(train_df,
                 train=True,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                 ]))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)


test_ds = ImageData(test_df, train=False, transform=transforms.Compose([
                     transforms.ToTensor()
                 ]))
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


anchor,positive,negative,labels = next(iter(train_loader))
# print(anchor.shape)
# print(positive.shape)
# print(negative.shape)
# print(labels.shape)

anchor[0].shape
torch_image = torch.squeeze(anchor[0])
image = torch_image.numpy()
image.shape

## Create instance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)

summary(model, input_size=(1, 48, 48))

model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = TripletLoss()
# criterion = nn.TripletMarginLoss(margin=1.0, p=2)


model.train()
for epoch in range(epochs):
    running_loss = []
    for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(train_loader):
        anchor_img, positive_img, negative_img, anchor_label = anchor_img.to(device), positive_img.to(
            device), negative_img.to(device), anchor_label.to(device)
        optimizer.zero_grad()
        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.detach().cpu().numpy())
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, epochs, np.mean(running_loss)))

train_results = []
labels = []
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model.eval()
with torch.no_grad():
    for img, _, _, label in train_loader:
        img, label = img.to(device), label.to(device)
        train_results.append(model(img).cpu().numpy())
        labels.append(label.cpu())

train_results = np.concatenate(train_results)
labels = np.concatenate(labels)
train_results.shape

## visualization
plt.figure(figsize=(15, 10), facecolor="azure")
for label in np.unique(labels):
    tmp = train_results[labels == label]
    plt.scatter(tmp[:, 0], tmp[:, 1], label=classes[label])

plt.legend()
plt.show()

tree = XGBClassifier(seed=300)
tree.fit(train_results, labels)

test_results = []
test_labels = []

model.eval()
with torch.no_grad():
    for img in test_loader:
        img = img.to(device)
        test_results.append(model(img).cpu().numpy())
        test_labels.append(tree.predict(model(img).cpu().numpy()))

test_results = np.concatenate(test_results)
test_labels = np.concatenate(test_labels)

plt.figure(figsize=(15, 10), facecolor="azure")
for label in np.unique(test_labels):
    tmp = test_results[test_labels == label]
    plt.scatter(tmp[:, 0], tmp[:, 1], label=classes[label])

plt.legend()
plt.show()

# accuracy
true_ = (tree.predict(test_results) == test_labels).sum()
len_ = len(test_labels)
print(tree.predict(test_results))
print(test_labels)
print("Accuracy :{}%".format((true_ / len_) * 100))  ##100%
