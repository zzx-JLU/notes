from torchvision import transforms
from PIL import Image

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')
print(type(img))

tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)

print(type(img_tensor))
