import torch
import torchvision
import torch.nn as nn
from model import simple_CNN

################################### Dataset #######################################

mnist_train = torchvision.datasets.MNIST('.',train=True, download=True)
x_train_dataset = mnist_train.train_data
y_train_dataset = mnist_train.train_labels

train_dataset = torch.utils.data.TensorDataset(x_train_dataset, y_train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

mnist_test = torchvision.datasets.MNIST('.',train=False)
x_test_dataset = mnist_test.test_data
y_test_dataset = mnist_test.test_labels

test_dataset = torch.utils.data.TensorDataset(x_test_dataset, y_test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

################################### Model #######################################
model = simple_CNN.simpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    loss_num = 0.0
    for i, (data, label) in enumerate(train_loader):
        data = data.unsqueeze(dim=1).float()
        pred = model(data)
        loss = criterion(pred, label)
        loss_num += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss_num / len(train_loader))