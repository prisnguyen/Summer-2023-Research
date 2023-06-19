import torchvision.models as models
import torch


# Use the VGG16 model
vgg16 = models.vgg16(pretrained=True)
input_data = torch.randn(1, 3, 224, 224)
output = vgg16(input_data)
print("Output shape:", output.shape)
print("Output values:", output)