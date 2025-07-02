import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class GSPCNNAEModel(BaseModel):
    def __init__(self, z_dim, is_positioning):
        super().__init__()
        self.z_dim = z_dim
        self.is_positioning = is_positioning

        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = nn.Conv1d(256, self.z_dim, 3, padding=1)

        self.conv6 = nn.Conv1d(self.z_dim, 256, 3, padding=1)
        self.conv7 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv8 = nn.Conv1d(256, 128, 5, padding=2)
        self.conv9 = nn.Conv1d(128, 64, 5, padding=2)
        self.conv10 = nn.Conv1d(64, 1, 5, padding=2)

        self.fc1 = nn.Linear(self.z_dim, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

        self.pool = nn.MaxPool1d(2, 2)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout(0.2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.kaiming_normal_(self.conv10.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        encoded = self.conv5(x)
        
        if self.is_positioning:
            x = self.global_pool(encoded)
            x = x.view(-1, 20)
            x = self.fc1(x)
            x = F.leaky_relu(x)
            x = self.fc2(x)
            x = F.leaky_relu(x)
            x = self.fc3(x)
            return x
        else:
            x = self.upsampling(encoded)
            x = self.conv6(x)
            x = self.batchnorm3(x)
            x = F.leaky_relu(x)
            x = self.upsampling(x)
            x = self.conv7(x)
            x = self.batchnorm3(x)
            x = F.leaky_relu(x)
            x = self.upsampling(x)
            x = self.conv8(x)
            x = self.batchnorm2(x)
            x = F.leaky_relu(x)
            x = self.upsampling(x)
            x = self.conv9(x)
            x = self.batchnorm1(x)
            x = F.leaky_relu(x)
            x = self.conv10(x)
            return x
            
class GSPCNNAEPeakModel(BaseModel):
    def __init__(self, z_dim, is_positioning, model_weighted_path, frozen):
        super().__init__()
        self.z_dim = z_dim
        self.is_positioning = is_positioning

        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = nn.Conv1d(256, 32, 3, padding=1)
        self.conv11 = nn.Conv1d(32, self.z_dim, 1)

        self.conv12 = nn.Conv1d(self.z_dim, 32, 1)
        self.conv6 = nn.Conv1d(32, 256, 3, padding=1)
        self.conv7 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv8 = nn.Conv1d(256, 128, 5, padding=2)
        self.conv9 = nn.Conv1d(128, 64, 5, padding=2)
        self.conv10 = nn.Conv1d(64, 1, 5, padding=2)

        # self.transformer_layer = nn.TransformerEncoderLayer(
        #     d_model=self.z_dim,
        #     nhead=8,
        #     dim_feedforward=1024,
        #     dropout=0.1,
        #     activation="relu"
        #     )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     encoder_layer=self.transformer_layer,
        #     num_layers=3
        #     )
        self.fc11 = nn.Linear(self.z_dim, 16)
        self.fc12 = nn.Linear(16, 2)
        self.fc13 = nn.Linear(60, 2)

        self.fc21 = nn.Linear(self.z_dim * 30, 32)
        self.fc22 = nn.Linear(32, 10)
        self.fc23 = nn.Linear(10, 2)

        self.pool = nn.MaxPool1d(2, 2)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout(0.2)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(32)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)
        nn.init.kaiming_normal_(self.conv6.weight)
        nn.init.kaiming_normal_(self.conv7.weight)
        nn.init.kaiming_normal_(self.conv8.weight)
        nn.init.kaiming_normal_(self.conv9.weight)
        nn.init.kaiming_normal_(self.conv10.weight)
        nn.init.kaiming_normal_(self.conv11.weight)
        nn.init.kaiming_normal_(self.conv12.weight)
        nn.init.kaiming_normal_(self.fc11.weight)
        nn.init.kaiming_normal_(self.fc12.weight)
        nn.init.kaiming_normal_(self.fc13.weight)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.leaky_relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # encoded = self.conv5(x)
        x = self.conv5(x)
        x = self.batchnorm4(x)
        x = F.leaky_relu(x)
        encoded = self.conv11(x)
        
        if self.is_positioning:
            
            # x = torch.transpose(encoded, 1, 2)
            # x = self.transformer_encoder(x)
            # x = self.fc11(x)
            # x = F.leaky_relu(x)
            # x = self.fc12(x)
            # x = F.leaky_relu(x)
            # x = x.view(-1, 60)
            # x = self.fc13(x)

            x = encoded.view(-1, self.z_dim * 30)
            x = self.fc21(x)
            x = F.leaky_relu(x)
            x = self.fc22(x)
            x = F.leaky_relu(x)
            x = self.fc23(x)
            
            return x
        else:
            x = self.conv12(encoded)
            x = self.batchnorm4(x)
            x = F.leaky_relu(x)
            x = self.upsampling(x)
            x = self.conv6(x)
            x = self.batchnorm3(x)
            x = F.leaky_relu(x)
            x = self.upsampling(x)
            x = self.conv7(x)
            x = self.batchnorm3(x)
            x = F.leaky_relu(x)
            x = self.upsampling(x)
            x = self.conv8(x)
            x = self.batchnorm2(x)
            x = F.leaky_relu(x)
            x = self.upsampling(x)
            x = self.conv9(x)
            x = self.batchnorm1(x)
            x = F.leaky_relu(x)
            x = self.conv10(x)
            # x = F.sigmoid(x)
            return x
