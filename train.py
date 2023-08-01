import torch
import torch.nn as nn
from torch import optim

from train_valid_function import train
from dataloader import generate_data
from SRNet import SRNet
# from SRNet_Attention import SRNet_CBAM as SRNet

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True)
model = torch.nn.DataParallel(model).cuda()
print('model down successfully')

# 数据预处理
data_path = {
    'train_cover': '/data/zhangle/github_test/SR_github/dataset/train/cover',
    'train_stego': '/data/zhangle/github_test/SR_github/dataset/train/container',
    'valid_cover': '/data/zhangle/github_test/SR_github/dataset/test/cover',
    'valid_stego': '/data/zhangle/github_test/SR_github/dataset/test/container'
}

# batch_size = {'train': 8, 'valid': 8}
batch_size = {'train': 10, 'valid': 10}
train_loader, valid_loader = generate_data(data_path, batch_size)
print('data_loader down successfully')

# 训练参数设置
EPOCHS = 180
write_interval = 1
valid_interval = 1
save_interval = 1
learning_rate = 1e-3

# 损失函数 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

load_path = None

# 开始训练
print('start train')
train(model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      EPOCHS=EPOCHS,
      optimizer=optimizer,
      criterion=criterion,
      device=device,
      valid_interval=valid_interval,
      save_interval=save_interval,
      write_interval=write_interval,
      load_path=load_path)

