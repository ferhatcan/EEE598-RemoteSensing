import os, shutil, random
from pathlib import Path
# from kaggle import api
import zipfile
import torch
import numpy as np
import torchvision
import torchvision.transforms as T
from PIL import Image
# from fastdownload import FastDownload
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from dataloader import AerialDataset

import matplotlib
matplotlib.use('Qt5Agg')
plt.ion()

cifar_labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")
alphabet_labels = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try:
        torch.manual_seed(s)
    except NameError:
        pass
    try:
        torch.cuda.manual_seed_all(s)
    except NameError:
        pass
    try:
        np.random.seed(s % (2 ** 32 - 1))
    except NameError:
        pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def untar_data(url, force_download=False, base='./datasets'):
#     d = FastDownload(base=base)
#     return d.get(url, force=force_download, extract_key='data')


# def get_alphabet(args):
#     get_kaggle_dataset("alphabet", "thomasqazwsxedc/alphabet-characters-fonts-dataset")
#     train_transforms = T.Compose([
#         T.Grayscale(),
#         T.ToTensor(), ])
#     train_dataset = torchvision.datasets.ImageFolder(root="./alphabet/Images/Images/", transform=train_transforms)
#     if args.slice_size > 1:
#         train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
#     train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
#     return train_dataloader, None


# def get_cifar(cifar100=False, img_size=64):
#     "Download and extract CIFAR"
#     cifar10_url = 'https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz'
#     cifar100_url = 'https://s3.amazonaws.com/fast-ai-imageclas/cifar100.tgz'
#     if img_size == 32:
#         return untar_data(cifar100_url if cifar100 else cifar10_url)
#     else:
#         get_kaggle_dataset("datasets/cifar10_64", "joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution")
#         return Path("datasets/cifar10_64/cifar10-64")


# def get_kaggle_dataset(dataset_path,  # Local path to download dataset to
#                        dataset_slug,  # Dataset slug (ie "zillow/zecon")
#                        unzip=True,  # Should it unzip after downloading?
#                        force=False  # Should it overwrite or error if dataset_path exists?
#                        ):
#     '''Downloads an existing dataset and metadata from kaggle'''
#     if not force and Path(dataset_path).exists():
#         return Path(dataset_path)
#     api.dataset_metadata(dataset_slug, str(dataset_path))
#     api.dataset_download_files(dataset_slug, str(dataset_path))
#     if unzip:
#         zipped_file = Path(dataset_path) / f"{dataset_slug.split('/')[-1]}.zip"
#         import zipfile
#         with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
#             zip_ref.extractall(Path(dataset_path))
#         zipped_file.unlink()


def one_batch(dl):
    return next(iter(dl))


def plot_images(images):
    # plt.figure(figsize=(12, 12))
    plt.figure()
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    train_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size + int(.25 * args.img_size)),  # args.img_size + 1/4 *args.img_size
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = AerialDataset(args.dataset_path, transform=train_transforms)
    train_set_size = int(len(dataset) * 0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, len(dataset) - train_set_size])

    if args.slice_size > 1:
        train_set = torch.utils.data.Subset(train_set, indices=range(0, len(train_set), args.slice_size))
        val_set = torch.utils.data.Subset(val_set, indices=range(0, len(val_set), args.slice_size))

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_set, batch_size=2 * args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_dataloader, val_dataloader


def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
