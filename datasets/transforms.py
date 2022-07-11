
from torchvision import transforms


def get_transforms(cfg):
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg.DATA_LOADER.RESIZE),
        transforms.CenterCrop(cfg.DATA_LOADER.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.43216, 0.394666, 0.37645],
                            [0.22803, 0.22145, 0.216989]),
    ])
    return data_transforms


inv_normalize = transforms.Normalize(
   mean=[-0.43216/0.22803, -0.394666/0.22145, -0.37645/0.216989],
   std=[1/0.22803, 1/0.22145, 1/0.216989]
)
