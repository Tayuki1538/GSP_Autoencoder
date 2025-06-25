import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class GSPCNNAEModel(BaseModel):
    def __init__(self, is_positioning):
        super().__init__()
        self.is_positioning = is_positioning

        self.conv1 = nn.Conv1d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = nn.Conv1d(256, 20, 3, padding=1)

        self.conv6 = nn.Conv1d(20, 256, 3, padding=1)
        self.conv7 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv8 = nn.Conv1d(256, 128, 5, padding=2)
        self.conv9 = nn.Conv1d(128, 64, 5, padding=2)
        self.conv10 = nn.Conv1d(64, 1, 5, padding=2)

        self.fc1 = nn.Linear(20,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16,2)

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
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        encoded = self.conv5(x)
        
        if self.is_positioning:
            x = self.global_pool(encoded)
            x = x.view(-1, 20)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            return x
        else:
            x = self.upsampling(encoded)
            x = self.conv6(x)
            x = self.batchnorm3(x)
            x = F.relu(x)
            x = self.upsampling(x)
            x = self.conv7(x)
            x = self.batchnorm3(x)
            x = F.relu(x)
            x = self.upsampling(x)
            x = self.conv8(x)
            x = self.batchnorm2(x)
            x = F.relu(x)
            x = self.upsampling(x)
            x = self.conv9(x)
            x = self.batchnorm1(x)
            x = F.relu(x)
            x = self.conv10(x)
            return x
            
