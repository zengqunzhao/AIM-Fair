from torch import nn
import torchvision.models as models
from timm import create_model

 
class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 1)

    def forward(self, image):
        output = self.resnet18(image)
        output = output.squeeze()

        return output
    
class GenerateModel_ViT_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = create_model("vit_small_patch32_224", pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, 1)

    def forward(self, image):
        output = self.vit(image)
        output = output.squeeze()

        return output