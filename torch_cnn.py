import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

IMG_SIZE = 130
SLICE_COUNT = 26


class CustomDataset(Dataset):
	def __init__(self, data, transforms=None):
		self.transforms = transforms
		self.data = data
		self.len_data = len(data)  # Add one for the len() method.

	def __getitem__(self, index):
		data_pair = self.data[index]
		image = data_pair[0]
		# print(image.dtype)
		image = image.astype("uint8")
		# print(image.dtype)
		label = data_pair[1]

		# apply transforms to image
		if self.transforms is not None:
			image = self.transforms(image)

		return (image, label)

	def __len__(self):
		return self.len_data




transform = transforms.Compose(
    [transforms.ToTensor()])

# load data
much_data = np.load('available_data-130-130-26-class1.npy', encoding="latin1")

trainset = much_data[:-20]
testset = much_data[-20:]

trainset = CustomDataset(trainset, transforms=transform)
testset = CustomDataset(testset, transforms=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)
dataiter = iter(trainloader)
images, labels = dataiter.next()

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()


		self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3)
		self.pool=nn.MaxPool3d(2, 2)
		self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)
		self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
		self.fc1 = nn.Linear(25088, 1024)
		self.fc2 = nn.Linear(1024, 256)
		self.fc3 = nn.Linear(256, 2)
		self.Dropout = nn.Dropout3d(p=0.4)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = x.view(-1, 25088)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.Dropout(x)
		x = self.fc3(x)
		# x = nn.Softmax(x)
		return x

net = Net()

# debug
# a = [1, 0]
# b = [0, 1]
# print(metrics.roc_auc_score(a, b))

# loss functions
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

d_loss = {}

for epoch in range(1):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get inputs
		inputs, labels = data
		# print(inputs.shape)
		inputs = torch.unsqueeze(inputs, -4)
		# inputs = torch.unsqueeze(inputs,0)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		# print(outputs.shape)
		# print("outputs", outputs)
		# print("labels", labels)
		# outputs = nn.Softmax(outputs)
		loss = criterion(outputs, torch.max(labels, 1)[1])
		# loss = criterion(outputs, labels)
		d_loss[i] = loss

		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 25 == 0: 	# print every 25 mini-batches
			print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 25))
			running_loss = 0.0

	# plt.figure()
	# plt.title("loss for epoch {}".format(epoch))
	# plt.plot(d_loss.keys(), d_loss.values())
	# plt.xlabel("Training time")
	# plt.ylabel("loss")
	# plt.savefig("loss{}.png".format(epoch))
	# d_loss = {}

print("Finished training.")



# Testing
y_pred = np.zeros(20)
y_test = np.zeros(20)

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(testloader):
        images, labels = data
        images = torch.unsqueeze(images, -4)
        outputs = (net(images))
        # print(outputs, labels)
        # print(outputs)
        # print(torch.max(outputs.data, 1))
        # print(torch.max(labels, 1)[1])
        
        _, predicted = torch.max(outputs, 1)
        # print(type(predicted),predicted[0])
        total += labels.size(0)
        
        labels = torch.max(labels, 1)[1]

        # print(labels, predicted)
        
        y_pred[i] = (predicted.item())
        y_test[i] = (labels.item())
       

        correct += (predicted == labels).sum().item()
    print(y_pred)
    print(y_test)


# Calculate AUC-ROC score
auc = metrics.roc_auc_score(y_test, y_pred)
print('AUC_ROC score of the network on the 20 test images: %d ' % (
    auc))
# print(total)
print('Accuracy of the network on the 20 test images: %d %%' % (
    100 * correct / total))




