import os
import torch
from torchvision.transforms import Resize, functional
from PIL import Image
from torch.utils.data import Dataset

image_resize = 256


def img_to_tensor(path: str) -> torch.Tensor:
    return functional.pil_to_tensor(Image.open(path).convert("RGB")).float()


def to_batch(data_path: str, list_dir: list[str]) -> torch.Tensor:
    resize = Resize([image_resize, image_resize])
    return torch.stack(
        [
            resize(img_to_tensor(os.path.join(data_path, file))).float()
            for file in list_dir
        ]
    )


def to_batch_labels(label_path: str, list_dir: list[str]) -> torch.Tensor:
    resize = Resize([image_resize, image_resize])
    return torch.stack(
        [resize(img_to_tensor(os.path.join(label_path, file)))[0]
         for file in list_dir]
    )


def get_mean_and_std(data_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    tensor: torch.Tensor = to_batch(data_path, os.listdir(data_path))
    tensor = tensor.view(3, -1)
    return (tensor.mean(dim=1), tensor.std(dim=1))


class UNet_Dataset(Dataset):
    images: torch.Tensor
    labels: torch.Tensor

    def __init__(self, data_path: str, labels_path: str, data_transform=None):
        super().__init__()
        files = os.listdir(data_path)
        self.images = to_batch(data_path, files)
        self.labels = to_batch_labels(labels_path, files).div(255).round()
        self.data_transform = data_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.images[index]
        label = self.labels[index]

        if self.data_transform:
            data = self.data_transform(data)

        return (
            data.cuda(),
            label.unsqueeze(0).cuda(),
        )
