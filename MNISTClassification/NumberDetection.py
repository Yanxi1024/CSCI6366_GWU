from torchvision.datasets import MNIST
import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Config
learning_rate = 0.03
batch_size = 64

class MyNN(nn.Module):
    def __init__(self) -> None:
        super(MyNN, self).__init__()
        self.input_to_hidden1 = nn.Linear(28 * 28, 512)
        self.hidden1_to_hidden2 = nn.Linear(512, 128)
        self.hidden2_to_output = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.input_to_hidden1(x))
        x = F.relu(self.hidden1_to_hidden2(x))
        x = self.hidden2_to_output(x)
        return x

transform = transforms.Compose([
    # transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_set = MNIST(root = "./", train = True, download = True, transform = transform)
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

# print(train_loader)

test_dataset = MNIST('./', train = False, transform = transform)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

model = MyNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, laberls = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, laberls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Test
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


