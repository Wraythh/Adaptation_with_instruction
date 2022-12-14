import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.nn.functional import cross_entropy
from model.resnet50 import BCLModel

class Instructor(nn.Module):
    def __init__(
        self, 
        class_num: int = 10,
        feat_dim: int = 1024,
        hidden_sizes: tuple = [64, 64],
        device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        self.instructor = nn.Sequential(*[
            nn.Linear(class_num + 3 + feat_dim, 128), nn.ReLU(),
            nn.Linear(128, class_num),
        ]).to(self.device)
    
    def forward(self, x):
        x = self.instructor(x)
        return x

class Learner(nn.Module):
    def __init__(
        self,
        class_num,
        device,
        feat_dim: int = 1024
    ):
        super().__init__()
        self.device = device
        # self.model = LeNet(class_num=class_num).to(device)
        self.model = BCLModel(name='resnet50', num_classes=class_num, feat_dim=feat_dim,
                                 use_norm=True).to(device)
    
    def forward(self, x, retrain=False):
        logits, feature = self.model(x, retrain=retrain)
        return logits, feature
    
    def get_params(self):
        params_vector = []
        for k, v in self.model.state_dict():
            params_vector.extend(v)
        return params_vector
    
    def feature_output(self, images, labels, indices):
        with torch.no_grad():
            images = images.to(self.device)
            labels = labels.to(self.device)
            pred_y, feature = self(images)
            feature_map = self.model.feature_map(images)
            loss = cross_entropy(pred_y, labels, reduction='none').unsqueeze(1)
            p_y = torch.softmax(pred_y, 1).gather(1, labels.unsqueeze(1)).squeeze()
            top2 = torch.topk(torch.softmax(pred_y, 1), 2).values
            cond = p_y == top2[:, 0]
            p_y_ = torch.where(cond, top2[:, 1], top2[:, 0])
            margin_p = (p_y - p_y_).unsqueeze(1)
            indices = (indices / 50000.).unsqueeze(1).to(self.device)
            # model_feature = (torch.tensor([iteration, max(losses), max(val_accuracy)])*torch.ones((len(pred_y), 3))).to(self.device)
            feature = torch.concat([pred_y, feature_map, loss, margin_p, indices], dim=1)
        return feature


class Model(nn.Module):
    def __init__(
        self, 
        class_num: int = 10,
    ):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, 3, 1, 1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2, 2)) 
        self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p = 0.5),
                                         torch.nn.Linear(1024, class_num))
        
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14*14*128)
        x = self.dense(x)
        return x
    
    
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.dense = nn.Sequential(nn.Linear(28*28, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 10))
        
    def forward(self, x) :
        x = x.view(-1, 28*28)
        x = self.dense(x)
        return x

class LeNet(nn.Module):
    def __init__(
        self,
        class_num: int = 10,
    ):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, class_num)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    
    def feature_map(self, img):
        x = self.conv(img)
        return x.view(img.shape[0], -1)