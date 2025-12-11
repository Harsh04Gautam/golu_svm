import os
import torch
import torchvision
import torchvision.transforms as T

from config import Config

cfg = Config()

TRAIN_DATA = os.path.join(os.path.dirname(__file__), "train_data")
TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data")


test_transform = T.Compose(
    [T.ToTensor(), T.Normalize((0.5071, 0.4865, 0.4409),
                               (0.2673, 0.2564, 0.2761))])
train_transform = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode='reflect'),
    T.RandomHorizontalFlip(),
    T.ColorJitter(),
    # T.AutoAugment(),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4865, 0.4409),
                (0.2673, 0.2564, 0.2761)),
    T.RandomErasing(p=0.25),
])
trainset = torchvision.datasets.CIFAR100(
    root=TRAIN_DATA, train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR100(
    root=TEST_DATA, train=False, download=True, transform=test_transform)


def load_data():

    train = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.batch,
        shuffle=False,
        generator=torch.Generator(device=torch.get_default_device()))
    test = torch.utils.data.DataLoader(
        testset,
        batch_size=cfg.batch,
        shuffle=False,
        generator=torch.Generator(device=torch.get_default_device()))
    return train, test, trainset.classes
