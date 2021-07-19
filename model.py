import torch
from torch import nn
from skorch import NeuralNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ArseNet(nn.Module):
  def __init__(self):
    super(ArseNet, self).__init__()
    c, h, w = (8, 8, 12)
    self.net = nn.Sequential(
        nn.Conv2d(512,  256, 3, padding='same'),
        nn.ReLU(),
        nn.Conv2d(256, 128, 3, padding='same'),
        nn.ReLU(),
        nn.Conv2d(128, 81, 3, padding='same'),
        nn.ReLU(),
        nn.Sigmoid()
    )
  def forward(self, x):
    return self.net(x)
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model():
  optimizer = torch.optim.Adam
  criterion = nn.BCELoss
  batch_size = 16
  model = NeuralNet(ArseNet, criterion, optimizer, lr=0.0001, max_epochs=1000,
                    batch_size=batch_size, device=device)
  return model
