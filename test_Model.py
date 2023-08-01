import torch
from dataloader import generate_test_data
from train_valid_function import test
from SRNet import SRNet

#events.out.tfevents.1618881943.heu-ubuntu
weight_path = '/data/zhangle/3_SRNet/HUGO_01_Model/summary/events.out.tfevents.1659016368.amax-SYS-7049GP-TRT'

# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = SRNet(data_format='NCHW', init_weights=True)
print('model down successfully')

# 数据预处理
data_path = {
    'test_cover': '/data/zhangle/S-UNIWARD_linux_make_v10/images_cover/',
    'test_stego': '/data/zhangle/S-UNIWARD_linux_make_v10/images_stego/'
}
batch_size = 8
test_loader = generate_test_data(data_path, batch_size)
print('data_loader down successfully')

test(model=model,
     test_loader=test_loader,
     device=device,
     weight_path=weight_path)


