import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import DataPreProcess
import NeruralNetwork

PATH_TRAIN_DUCKS = "./Data/train/ducks/"
PATH_TRAIN_NOT_DUCKS = "./Data/train/not_ducks/"
PATH_TEST_DUCKS = "./Data/test/ducks/"
PATH_TEST_NOT_DUCKS = "./Data/test/not_ducks/"

EPCHO = 3
LR = 0.01
MOMENTUM = 0.9


train_set = DataPreProcess.DuckDataSet(PATH_TRAIN_DUCKS, target = "duck", transform = DataPreProcess.transform) + DataPreProcess.DuckDataSet(PATH_TRAIN_NOT_DUCKS, transform = DataPreProcess.transform)
test_set = DataPreProcess.DuckDataSet(PATH_TEST_DUCKS, target = "duck", transform = DataPreProcess.transform) + DataPreProcess.DuckDataSet(PATH_TEST_NOT_DUCKS, transform = DataPreProcess.transform)

train_loader = DataLoader(train_set, batch_size = 5, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 5, shuffle = True)

model = NeruralNetwork.FullyConnectedNN()
# learning_loss = 0.0
criterion = nn.CrossEntropyLoss()
optimizor = torch.optim.SGD(model.parameters(), lr = LR, momentum = MOMENTUM)

for epoch in range(EPCHO):
    running_loss = 0.0
    for index, data in enumerate(train_loader):
        batch, targets = data
        optimizor.zero_grad()
        output = model(batch)
        loss = criterion(output, targets.long())
        loss.backward()
        optimizor.step()
        running_loss += loss.item()
        print("[%d, %5d] loss: %.3f" % (epoch + 1, index + 1, running_loss))
        running_loss = 0.0

print("Training finished.")

model.eval()

# with torch.no_grad():
#     for index, data in enumerate(test_loader):
#         inputs, _ = data
#         outputs = model(inputs)
#         _, predictions = torch.max(outputs, 1)
#
#         # Get the image names and predictions
#         # batch_start_index = index * test_loader.batch_size
#         # batch_end_index = batch_start_index + len(batch)
#         # image_names = []
#         # for i in range(batch_start_index, batch_end_index):
#         #     dataset_index, sample_index = test_set.cumulative_sizes.index(i), i
#         #     if dataset_index > 0:
#         #         sample_index -= test_set.cumulative_sizes[dataset_index - 1]
#         #     image_names.append(test_set.datasets[dataset_index].imagesNameList[sample_index])
#
#         # Print image names and predictions
#         for image_name, prediction in zip(image_names, predictions):
#             print(f"{image_name}: {'Duck' if prediction.item() == 1 else 'Not Duck'}")


# Test
with torch.no_grad():
    correct = 0
    total = 0
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the duck test set: {accuracy:.2f}%")