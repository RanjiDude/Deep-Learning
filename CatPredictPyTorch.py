import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random

random_seed = random.seed(3)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataset = h5py.File('train_catvnoncat.h5', 'r')
train_set_x_orig = np.array(train_dataset['train_set_x'][:])
train_set_y_orig = np.array(train_dataset['train_set_y'][:])

test_dataset = h5py.File('test_catvnoncat.h5', 'r')
test_set_x_orig = np.array(test_dataset['test_set_x'][:])
test_set_y_orig = np.array(test_dataset['test_set_y'][:])

train_set_x = torch.from_numpy(train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)).float().to(device)
train_set_y = torch.from_numpy(train_set_y_orig.reshape(-1, 1)).float().to(device)
test_set_x = torch.from_numpy(test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)).float().to(device)
test_set_y = test_set_y_orig.reshape(-1, 1)

# print(train_set_x.shape)
# print(train_set_y.shape)


class Model(nn.Module):
    def __init__(self, input_size=12288, fc1_size=1024, output_size=1):
        super(Model, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, fc1_size)
        # self.fc2 = nn.Linear(fc1_size, fc2_size)
        # self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.output = nn.Linear(fc1_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(self.bn1(x)))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.output(x))
        return x

model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)
losses = np.zeros(2000)

predictions = model(train_set_x).cpu().detach().numpy()
train_set_y = train_set_y.cpu().detach().numpy()
for i in range(predictions.shape[0]):
    if predictions[i]<=0.5:
        predictions[i] = 0
    else:
        predictions[i] = 1
print("train_accuracy: {:.2f}%".format(100 - np.mean(np.abs(predictions - train_set_y))*100))

test_predictions = model(test_set_x).cpu().detach().numpy()
for i in range(test_predictions.shape[0]):
    if test_predictions[i]<=0.5:
        test_predictions[i]=0
    else:
        test_predictions[i]=1
print("test_accuracy: {:.2f}%".format(100 - np.mean(np.abs(test_predictions - test_set_y))*100))

train_set_y = torch.from_numpy(train_set_y_orig.reshape(-1, 1)).float().to(device)

for i in range(2000):
    prediction = model(train_set_x)
    loss = F.mse_loss(prediction, train_set_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses[i] = loss

torch.save(model.state_dict(), 'trainedweights.pth')

fig = plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.plot(np.arange(len(losses)), losses)
plt.show()

predictions = model(train_set_x).cpu().detach().numpy()
train_set_y = train_set_y.cpu().detach().numpy()
for i in range(predictions.shape[0]):
    if predictions[i]<=0.5:
        predictions[i] = 0
    else:
        predictions[i] = 1
print("train_accuracy: {:.2f}%".format(100 - np.mean(np.abs(predictions - train_set_y))*100))

test_predictions = model(test_set_x).cpu().detach().numpy()
for i in range(test_predictions.shape[0]):
    if test_predictions[i]<=0.5:
        test_predictions[i]=0
    else:
        test_predictions[i]=1

print("test_accuracy: {:.2f}%".format(100 - np.mean(np.abs(test_predictions - test_set_y))*100))
