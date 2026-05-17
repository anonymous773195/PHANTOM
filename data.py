
import os
import ssl
import glob
import math

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

ssl._create_default_https_context = ssl._create_unverified_context

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASET_INFO = {
    "cifar10": {
        "cls": torchvision.datasets.CIFAR10,
        "num_classes": 10,
        "classes": (
            "plane", "car", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ),
        "grayscale": False,
    },
    "cifar100": {
        "cls": torchvision.datasets.CIFAR100,
        "num_classes": 100,
        "classes": None,
        "grayscale": False,
    },
    "mnist": {
        "cls": torchvision.datasets.MNIST,
        "num_classes": 10,
        "classes": tuple(str(i) for i in range(10)),
        "grayscale": True,
    },
    "imagenet": {
        "cls": None,
        "num_classes": 1000,
        "classes": None,
        "grayscale": False,
    },
    "imagenet100": {
        "cls": None,
        "num_classes": 100,
        "classes": None,
        "grayscale": False,
    },
}

DATASET_ALIASES = {
    "imagenet-100": "imagenet100",
}


def _build_transforms(image_size, is_train, grayscale):
    transforms_list = []

    if grayscale:
        transforms_list.append(transforms.Grayscale(num_output_channels=3))

    if is_train:
        transforms_list.extend([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5 if not grayscale else 0.0),
        ])
    else:
        transforms_list.append(transforms.Resize((image_size, image_size)))

    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return transforms.Compose(transforms_list)


def _get_imagenet_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _get_imagenet_val_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def prepare_data(
    dataset_name,
    batch_size=64,
    num_workers=4,
    data_root="./data",
    distributed=False,
):
    if dataset_name not in DATASET_INFO or dataset_name in {"imagenet", "imagenet100"}:
        raise ValueError(
            f"Use prepare_imagenet_data() / prepare_imagenet100_data() for ImageNet datasets. "
            f"For torchvision datasets choose from: {[k for k in DATASET_INFO if k not in {'imagenet', 'imagenet100'}]}"
        )

    info = DATASET_INFO[dataset_name]
    image_size = 224
    grayscale = info["grayscale"]

    train_transform = _build_transforms(image_size, is_train=True, grayscale=grayscale)
    test_transform = _build_transforms(image_size, is_train=False, grayscale=grayscale)

    trainset = info["cls"](root=data_root, train=True, download=True, transform=train_transform)
    testset = info["cls"](root=data_root, train=False, download=True, transform=test_transform)

    train_sampler = DistributedSampler(trainset, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(testset, shuffle=False) if distributed else None

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    classes = info["classes"]
    if classes is None and hasattr(trainset, "classes"):
        classes = tuple(trainset.classes)

    return trainloader, testloader, classes, info["num_classes"]


def _list_shards(data_root, split):
    shard_dir = os.path.join(data_root, f"{split}_shards")
    shards = sorted(glob.glob(os.path.join(shard_dir, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {shard_dir}")
    return shards


def _make_webdataset_loader(
    shard_pattern,
    batch_size,
    num_workers,
    train=True,
    batches_per_epoch=None,
):
    import webdataset as wds

    transform = _get_imagenet_train_transforms() if train else _get_imagenet_val_transforms()

    dataset = wds.WebDataset(
        shard_pattern,
        shardshuffle=100 if train else False,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
    ).decode("pil").to_tuple("jpg", "cls").map_tuple(
        transform,
        lambda x: int(x),
    )

    if train:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batched(batch_size, partial=False)

    loader_kwargs = dict(
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = wds.WebLoader(dataset, **loader_kwargs)

    if batches_per_epoch is not None:
        loader = loader.with_epoch(batches_per_epoch)

    return loader


def prepare_imagenet_data(
    data_root,
    batch_size=256,
    num_workers=8,
    world_size=1,
):
    train_shards = _list_shards(data_root, "train")
    val_shards = _list_shards(data_root, "val")

    imagenet_train_size = int(os.environ.get("IMAGENET_TRAIN_SIZE", "1281167"))
    imagenet_val_size = int(os.environ.get("IMAGENET_VAL_SIZE", "50000"))
    train_batches = max(1, math.floor(imagenet_train_size / (batch_size * world_size)))
    val_batches = max(1, math.floor(imagenet_val_size / (batch_size * world_size)))

    train_loader = _make_webdataset_loader(
        train_shards,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        batches_per_epoch=train_batches,
    )

    val_loader = _make_webdataset_loader(
        val_shards,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        batches_per_epoch=val_batches,
    )

    return train_loader, val_loader, None, 1000


def prepare_imagenet100_data(
    data_root,
    batch_size=256,
    num_workers=8,
    distributed=False,
    val_split=0.1,
    seed=42,
):
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"ImageNet-100 train directory not found: {train_dir}")
    if os.path.isdir(val_dir):
        trainset = torchvision.datasets.ImageFolder(
            root=train_dir,
            transform=_get_imagenet_train_transforms(),
        )
        valset = torchvision.datasets.ImageFolder(
            root=val_dir,
            transform=_get_imagenet_val_transforms(),
        )
        classes = tuple(trainset.classes) if hasattr(trainset, "classes") else None
    else:
        if not (0.0 < val_split < 1.0):
            raise ValueError(f"val_split must be in (0, 1), got {val_split}")

        base = torchvision.datasets.ImageFolder(root=train_dir, transform=None)
        num_samples = len(base)
        if num_samples < 2:
            raise ValueError(
                f"ImageNet-100 train directory needs >=2 samples for split, got {num_samples}"
            )

        val_count = max(1, int(num_samples * val_split))
        train_count = num_samples - val_count
        if train_count <= 0:
            raise ValueError(
                f"val_split={val_split} leaves no training samples ({num_samples} total)"
            )

        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(num_samples, generator=g).tolist()
        train_idx = perm[:train_count]
        val_idx = perm[train_count:]

        train_full = torchvision.datasets.ImageFolder(
            root=train_dir,
            transform=_get_imagenet_train_transforms(),
        )
        val_full = torchvision.datasets.ImageFolder(
            root=train_dir,
            transform=_get_imagenet_val_transforms(),
        )
        trainset = Subset(train_full, train_idx)
        valset = Subset(val_full, val_idx)
        classes = tuple(base.classes) if hasattr(base, "classes") else None

    train_sampler = DistributedSampler(trainset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(valset, shuffle=False) if distributed else None

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, valloader, classes, len(classes) if classes is not None else 100


def normalize_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def get_num_classes(dataset_name):
    dataset_name = normalize_dataset_name(dataset_name)
    return DATASET_INFO[dataset_name]["num_classes"]


if __name__ == "__main__":
    for name in ["cifar10", "cifar100"]:
        trainloader, testloader, classes, num_classes = prepare_data(name, batch_size=4)
        images, labels = next(iter(trainloader))
        print(f"{name}: images={images.shape}, labels={labels.shape}, "
              f"num_classes={num_classes}, classes[:5]={classes[:5] if classes else 'N/A'}")
