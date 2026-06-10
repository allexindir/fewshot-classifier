import logging
import os
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FewShotImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root = root
        self.transform = transform

        if class_names is None:
            class_names = [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]

        self.class_names = sorted(class_names, key=int)

        self.class_to_idx = {name: int(name) for name in self.class_names}

        self.samples: List[Tuple[str, int]] = []
        for class_name in self.class_names:
            class_dir = os.path.join(root, class_name)
            label = self.class_to_idx[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), label))

        logger.info(f"Loaded {len(self.samples)} images from {len(self.class_names)} classes")
        logger.info(f"Label mapping sample: '0'->{self.class_to_idx.get('0')}, "
                     f"'10'->{self.class_to_idx.get('10')}, '99'->{self.class_to_idx.get('99')}")

        # Sanity check
        for name in self.class_names:
            assert self.class_to_idx[name] == int(name), \
                f"Label mapping error: directory '{name}' mapped to {self.class_to_idx[name]}, expected {int(name)}"

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class TestImageDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform

        self.image_ids: List[int] = []
        for fname in os.listdir(root):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_id = int(os.path.splitext(fname)[0])
                self.image_ids.append(image_id)

        self.image_ids.sort()
        logger.info(f"Loaded {len(self.image_ids)} test images")

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        path = os.path.join(self.root, f"{image_id}.jpg")
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_id
