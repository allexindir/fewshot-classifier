from typing import Dict, List, Sequence, Tuple

from torchvision import transforms

from transformers import AutoProcessor
import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_SIGLIP_MODEL = "google/siglip2-so400m-patch14-384"

_SIGLIP_NORM_CACHE: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...], int]] = {}


def siglip_norm(model_name: str = DEFAULT_SIGLIP_MODEL) -> Tuple[Tuple[float, ...], Tuple[float, ...], int]:
    if model_name not in _SIGLIP_NORM_CACHE:
        proc = AutoProcessor.from_pretrained(model_name).image_processor
        _SIGLIP_NORM_CACHE[model_name] = (
            tuple(proc.image_mean),
            tuple(proc.image_std),
            int(proc.size["height"]),
        )
    return _SIGLIP_NORM_CACHE[model_name]


def build_train_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])


def build_eval_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def build_siglip_transform(model_name: str = DEFAULT_SIGLIP_MODEL, train: bool = False) -> T.Compose:
    mean, std, size = siglip_norm(model_name)
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def build_train_transform_for(backbone_name: str, model_name: str, image_size: int) -> transforms.Compose:
    if backbone_name == "siglip2":
        return build_siglip_transform(model_name or DEFAULT_SIGLIP_MODEL, train=True)
    return build_train_transform(image_size)


def build_eval_transform_for(backbone_name: str, model_name: str, image_size: int) -> transforms.Compose:
    if backbone_name == "siglip2":
        return build_siglip_transform(model_name or DEFAULT_SIGLIP_MODEL, train=False)
    return build_eval_transform(image_size)


def build_tta_transforms_for(backbone_name: str, model_name: str, image_size: int) -> List[transforms.Compose]:
    if backbone_name == "siglip2":
        mean, std, size = siglip_norm(model_name or DEFAULT_SIGLIP_MODEL)
        return build_tta_transforms(size, mean=mean, std=std)
    return build_tta_transforms(image_size)


def build_tta_transforms(
    image_size: int = 256,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> List[transforms.Compose]:
    normalize = transforms.Normalize(mean=mean, std=std)

    # resize
    base = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # resize + horizontal flip
    hflip = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize,
    ])

    # 3 slightly different scales resized back to image_size
    scale_sizes = [image_size - 16, image_size - 8, image_size + 16]
    scale_crops = []
    for size in scale_sizes:
        t = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        scale_crops.append(t)

    return [base, hflip] + scale_crops
