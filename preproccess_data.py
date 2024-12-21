import torch
from torchvision import transforms
from dataset import UNet_Dataset

normalization = transforms.Normalize(
    [151.1672, 141.4257, 139.8164], [55.9709, 52.2902, 55.2580]
)

torch.save(
    UNet_Dataset(
        "./swimseg/train/data/",
        "./swimseg/train/labels/",
        normalization,
    ),
    "./train_data.pt",
)
torch.save(
    UNet_Dataset(
        "./swimseg/test/input/",
        "./swimseg/test/labels/",
        normalization,
    ),
    "./test_data.pt",
)
