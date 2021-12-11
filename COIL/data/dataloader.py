import torch
from data.dataset import MNIST
data_loader_train = torch.utils.data.DataLoader(MNIST(), batch_size=1440, shuffle=False)


