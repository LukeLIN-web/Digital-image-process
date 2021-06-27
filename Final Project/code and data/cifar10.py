import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 8


class DealDataset(Dataset):

    def __init__(self, folder, data_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes') #从file中读取，并将它重构为原来的python对象。
    return dict ## 是标成0-9分类，还有其他key有字符串格式的分类说明


def load_data(folder, data_name):
    data_batch1 = unpickle(folder + data_name)
    train_set_x = data_batch1[b'data']
    train_set_x = train_set_x.reshape(10000, 3, 32, 32)# 转成矩阵格式
    train_set_x = np.transpose(train_set_x, (0, 2, 3, 1))# 改成 10000 * 32 * 32* 3
    binary_set_x = train_set_x / 255 #归一化
    train_set_y = data_batch1[b'labels']
    train_set_y = np.array(train_set_y)#载入labels
    return binary_set_x, train_set_y


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = DealDataset(r'./cifar/', "data_batch_1", transform=transform)
test_dataset = DealDataset(r'./cifar/', "test_batch", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 图片可视化
images, labels = next(iter(train_loader))#使用iter(dataloader)返回的是一个迭代器，然后可以使用next访问；
img = torchvision.utils.make_grid(images)
img = img / 2 + 0.5
img = img.numpy().transpose(1, 2, 0)
print(' '.join('%5s' % classes[labels[j]] for j in range(8)))
plt.imshow(img)
plt.show()

#一个Net，也就是继承自nn.Module的类
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) #设置网络中的全连接层的，需要注意的是二维卷积全连接层的输入与输出都是二维张量

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net = net.double()#
criterion = nn.CrossEntropyLoss() #交叉熵函数
optimizer = optim.SGD(net.parameters(), lr=0.002, momentum=0.9)  # 优化器,用于更新权重；

loss_list = []
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward() # Compute new gradients
        optimizer.step() # Apply these gradients

        # print statistics
        running_loss += loss.item()

        if i % 500 == 499:  # print every 500 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            loss_list.append(running_loss / 500)
            running_loss = 0.0

print('Finished Training')
plt.plot(loss_list)
plt.title('traning loss')
plt.xlabel('epochs')
plt.ylabel('loss') # 绘制loss曲线
plt.show()

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH) # 保存

# 测试
images, labels = next(iter(test_loader))
img = torchvision.utils.make_grid(images)
img = img / 2 + 0.5
img = img.numpy().transpose(1, 2, 0)
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(8)))
plt.imshow(img)
plt.show()

net = Net().double()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(8))) # 输出网络分类的结果

correct = 0
total = 0
with torch.no_grad():#torch.no_grad() 会关闭自动求导引擎的， 因此能节省显存，和加速。
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1) #按维度dim==1 返回最大值，并且返回索引。
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))  # 输出整个测试集的准确率

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze() #将输入张量形状中的1 去除并返回。 如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))  # 输出每个class的准确率
