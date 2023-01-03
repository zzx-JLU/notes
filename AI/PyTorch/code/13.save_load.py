import torchvision
import torch

vgg16 = torchvision.models.vgg16()

torch.save(vgg16, './model/vgg16.pt')
saved_model = torch.load('./model/vgg16.pt')
print(saved_model)

torch.save(vgg16.state_dict(), './model/vgg16_param.pt')
params = torch.load('./model/vgg16_param.pt')
vgg16_load = torchvision.models.vgg16()
vgg16_load.load_state_dict(params)
