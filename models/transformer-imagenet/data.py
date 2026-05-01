import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from config import Config


class ImageNetLocalizationDataset(Dataset):
    """ImageNet LOC/CLS-LOC dataset returning image, class id, and normalized xyxy box."""

    def __init__(self, root: str | Path, split: str, image_size: int, class_to_idx=None):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.samples = self._find_samples()
        self.class_to_idx = class_to_idx or self._build_class_index()

        if split == "train":
            self.color_transform = T.ColorJitter(0.2, 0.2, 0.2, 0.05)
        else:
            self.color_transform = None

        self.tensor_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, annotation_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        boxes, class_names = self._read_annotation(annotation_path)

        class_name = class_names[0]
        box = boxes[0]

        if self.split == "train":
            image, box = self._resize_square(image, box)
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
                box = torch.tensor(
                    [self.image_size - box[2], box[1], self.image_size - box[0], box[3]],
                    dtype=torch.float32,
                )
            if self.color_transform is not None:
                image = self.color_transform(image)
        else:
            image, box = self._resize_square(image, box)

        target = {
            "label": torch.tensor(self.class_to_idx[class_name], dtype=torch.long),
            "box": box / self.image_size,
        }
        return self.tensor_transform(image), target

    def _find_samples(self):
        image_root = self.root / "Data" / "CLS-LOC" / self.split
        annotation_root = self.root / "Annotations" / "CLS-LOC" / self.split

        if not image_root.exists() or not annotation_root.exists():
            raise FileNotFoundError(
                "Expected ImageNet LOC layout under "
                f"{self.root}: Data/CLS-LOC/{self.split} and "
                f"Annotations/CLS-LOC/{self.split}"
            )

        samples = []
        for image_path in sorted(image_root.rglob("*.JPEG")):
            rel = image_path.relative_to(image_root).with_suffix(".xml")
            annotation_path = annotation_root / rel
            if annotation_path.exists():
                samples.append((image_path, annotation_path))

        if not samples:
            raise FileNotFoundError(f"No annotated ImageNet samples found for split {self.split}")
        return samples

    def _build_class_index(self):
        image_root = self.root / "Data" / "CLS-LOC" / "train"
        class_names = {path.name for path in image_root.iterdir() if path.is_dir()}
        if not class_names:
            for _, annotation_path in self.samples:
                _, names = self._read_annotation(annotation_path)
                class_names.update(names)
        return {name: idx for idx, name in enumerate(sorted(class_names))}

    def _read_annotation(self, annotation_path):
        root = ET.parse(annotation_path).getroot()
        boxes = []
        class_names = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            bndbox = obj.find("bndbox")
            if name is None or bndbox is None:
                continue
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
            boxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
            class_names.append(name)

        if not boxes:
            raise ValueError(f"No objects found in {annotation_path}")
        return boxes, class_names

    def _resize_square(self, image, box):
        width, height = image.size
        image = TF.resize(image, [self.image_size, self.image_size])
        scale = torch.tensor(
            [self.image_size / width, self.image_size / height,
             self.image_size / width, self.image_size / height],
            dtype=torch.float32,
        )
        return image, box * scale

    def _random_resized_crop(self, image, box):
        width, height = image.size
        top, left, crop_h, crop_w = T.RandomResizedCrop.get_params(
            image, scale=(0.6, 1.0), ratio=(0.75, 1.33)
        )
        image = TF.resized_crop(image, top, left, crop_h, crop_w, [self.image_size, self.image_size])

        shifted = box - torch.tensor([left, top, left, top], dtype=torch.float32)
        shifted = shifted.clamp(
            min=torch.tensor([0, 0, 0, 0], dtype=torch.float32),
            max=torch.tensor([crop_w, crop_h, crop_w, crop_h], dtype=torch.float32),
        )
        scale = torch.tensor(
            [self.image_size / crop_w, self.image_size / crop_h,
             self.image_size / crop_w, self.image_size / crop_h],
            dtype=torch.float32,
        )
        return image, shifted * scale


def _collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1]["label"] for item in batch])
    boxes = torch.stack([item[1]["box"] for item in batch])
    return images, {"label": labels, "box": boxes}


def build_dataloaders(cfg: Config):
    train = ImageNetLocalizationDataset(cfg.data_root, "train", cfg.image_size)
    val = ImageNetLocalizationDataset(
        cfg.data_root,
        "val",
        cfg.image_size,
        class_to_idx=train.class_to_idx,
    )

    train_loader = DataLoader(
        train,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate,
    )
    return train_loader, val_loader
