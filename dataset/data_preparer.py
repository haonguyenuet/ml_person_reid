from torch.utils.data import DataLoader

import torchvision.transforms as T

from collections import namedtuple
from dataset.image import init_image_dataset

QueryGallery = namedtuple('QueryGallery', ['query', 'gallery'])
TrainTest = namedtuple('TrainTest', ['train', 'test'])


class DataPreparer(object):
    def __init__(
        self,
        root="",
        dataset_name=None,
        batch_size_train=32,
        batch_size_test=32,
        workers=4
    ):
        self.root = root
        self.dataset_name = dataset_name

        self._prepare_data()
        self._prepare_loader(batch_size_train, batch_size_test, workers)

        self._num_train_pids = self.trainset.num_train_pids
        self._num_train_cams = self.trainset.num_train_cams

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams   

    def _prepare_data(self):
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        transform_train = T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.RandomCrop(100),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ])

        transform_test = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ])
 
        self.trainset = init_image_dataset(
            name=self.dataset_name,
            root=self.root,
            mode='train',
            transform=transform_train
        )

        self.queryset = init_image_dataset(
            name=self.dataset_name,
            root=self.root,
            mode='query',
            transform=transform_test
        )

        self.galleryset = init_image_dataset(
            name=self.dataset_name,
            root=self.root,
            mode='gallery',
            transform=transform_test
        )

    def _prepare_loader(self, batch_size_train, batch_size_test, workers):
        self.trainloader = DataLoader(
            dataset=self.trainset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=workers,
            drop_last=True,
        )

        self.testloader = QueryGallery(
            query=DataLoader(
                dataset=self.queryset,
                batch_size=batch_size_test,
                shuffle=True,
                num_workers=workers
            ),
            gallery=DataLoader(
                dataset=self.galleryset,
                batch_size=batch_size_test,
                shuffle=True,
                num_workers=workers
            )
        )

    def fetch_loader(self):
        return TrainTest(train=self.trainloader, test=self.testloader)
