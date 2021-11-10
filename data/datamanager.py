import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataclasses import dataclass
from collections import namedtuple
from data.datasets.image_dataset import ImageDataset
from data.datasets.market1501 import Market1501
import argparse


QueryGallery = namedtuple('QueryGallery', ['query', 'gallery'])

class DataManager(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.root
        self.prepare_data()

        self.num_train_classes = self.trainset.get_num_pids()

    def prepare_data(self) -> None:
        datasets = Market1501(root=self.root)
        train, query, gallery = datasets.split_data()
        self.trainset = ImageDataset(data=train, transforms=ReidTransforms.train)
        self.queryset = ImageDataset(data=query, transforms=ReidTransforms.val)
        self.galleryset = ImageDataset(data=gallery, transforms=ReidTransforms.val)
       
    def train_loader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size= self.args.batch_size_train,
            shuffle=True,
            num_workers= self.args.workers,
            drop_last=True,
        )

    def test_loader(self):
        return  QueryGallery(
            query=DataLoader(
                dataset=self.queryset,
                batch_size=self.args.batch_size_test,
                shuffle=False,
                num_workers=self.args.workers
            ),
            gallery=DataLoader(
                dataset=self.galleryset,
                batch_size=self.args.batch_size_test,
                shuffle=False,
                num_workers=self.args.workers
            )
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("DataManager")
        parser.add_argument("--root", type=str, help="Path to store data")
        parser.add_argument("--batch_size_train", type=int, default=32)
        parser.add_argument("--batch_size_test", type=int, default=32)
        parser.add_argument("--workers", type=int, default=4)
        return parent_parser


@dataclass
class ReidTransforms:
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train = T.Compose([
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(100),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    val = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    test = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])
