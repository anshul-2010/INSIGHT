import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random


# Simple dataset wrapper expecting folder structure:
# root/real/*.png
# root/fake/*.png
class InsightDataset(Dataset):
    def __init__(
        self, root_dir, split="train", img_size=32, transform=None, augment_twice=False
    ):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform or self.default_transforms(split)
        self.augment_twice = augment_twice
        self.samples = []
        for label_dir, label in [("real", 0), ("fake", 1)]:
            path = os.path.join(root_dir, label_dir)
            if not os.path.exists(path):
                continue
            for fname in os.listdir(path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(path, fname), label))
        random.shuffle(self.samples)

    def default_transforms(self, split):
        if split == "train":
            return transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)], p=0.5
                    ),
                    transforms.ToTensor(),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                ]
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.augment_twice:
            # produce two random augmented views using the same transform
            img1 = self.transform(img)
            img2 = self.transform(img)
            return (img1, img2), label
        else:
            img = self.transform(img)
            return img, label
