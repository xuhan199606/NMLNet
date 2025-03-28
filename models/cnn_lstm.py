import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50

from models.mobilenet_v2 import *
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.mobileNeAt_v2 = mobilenet_v2(pretrained=False, num_classes=512, channel=1)
        # self.mobileNeAt_v2 = resnet50(pretrained=False, num_classes=5)


    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.mobileNeAt_v2(x_3d[:, t, :, :, :]).to(device)  # ResNet

            cnn_embed_seq.append(x)
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class colorClassification(nn.Module):
    def __init__(self, num_classes=5):
        super(colorClassification, self).__init__()

        self.mobileNeAt_v2 = mobilenet_v2(pretrained=False, num_classes=5, channel=3)
        # self.mobileNeAt_v2 = resnet50(pretrained=False, num_classes=5)

    def forward(self, x):
        x = self.mobileNeAt_v2(x)
        return x

if __name__ == "__main__":
    t = torch.Tensor(224,224,1,4)

    # a = ResCNNEncoder()
    # print(a)



