import torch.optim
from model import Cifar10
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10('../data/CIFAR10', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('./data/CIFAR10', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

# 使用 DataLoader 加载数据集
train_loader = DataLoader(train_data, batch_size=64)
test_loader = DataLoader(test_data)

# 创建网络模型
model = Cifar10()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 10  # 训练的轮数

# 添加 TensorBoard
writer = SummaryWriter('../logs')

# 训练
for i in range(epoch):
    print(f'------第 {i + 1} 轮训练开始------')

    model.train()  # 模型进入训练模式
    for imgs, targets in train_loader:
        outputs = model.forward(imgs)  # 前向传播
        loss = loss_fn(outputs, targets)  # 计算损失函数值

        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 执行优化

        total_train_step += 1
        writer.add_scalar('train_loss', loss.item(), total_train_step)
        if (total_train_step + 1) % 100 == 0:
            print(f'训练次数: {total_train_step}, Loss: {loss.item()}')

    # 每轮训练结束后，用测试集验证模型
    model.eval()  # 模型进入验证模式
    total_test_loss = 0
    total_correct_count = 0
    with torch.no_grad():  # 禁用梯度，降低计算成本
        for imgs, targets in test_loader:
            outputs = model.forward(imgs)  # 前向传播
            loss = loss_fn(outputs, targets)  # 计算损失函数值
            total_test_loss += loss.item()

            correct_count = (outputs.argmax(1) == targets).sum()
            total_correct_count += correct_count

    print(f'测试集上的Loss: {total_test_loss}')
    writer.add_scalar('test_loss', total_test_loss, total_test_step)

    total_accuracy = total_correct_count / len(test_loader)
    print(f'测试集上的准确率: {total_accuracy}')
    writer.add_scalar('test_accuracy', total_accuracy, total_test_step)

    total_test_step += 1

    # 保存每轮训练后的模型参数
    torch.save(model, f'../model/CIFAR10{i}.pt')

writer.close()
