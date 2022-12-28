from torchvision import transforms
from PIL import Image

img = Image.open('./data/hymenoptera_data/train/ants/0013035.jpg')
print(type(img))
print(img)
print(img.size)

# ToTensor: 转换为 Tensor 类型
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
print(type(img_tensor))
print(img_tensor)

# Normalize: 对 Tensor 类型的图像数据进行归一化
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm)

# Resize: 修改 PIL 图像大小
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)

# Compose: 组合多个 Transforms 类
trans_compose = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])
img_compose = trans_compose(img)
print(type(img_compose))
print(img_compose.size())
