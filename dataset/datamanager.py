from torch.utils.data import DataLoader

from dataset import init_image_dataset
from dataset.transform import build_transforms

from collections import namedtuple


QueryGallery = namedtuple('QueryGallery', ['query', 'gallery'])
TrainTest = namedtuple('TrainTest', ['train', 'test'])


class DataManager(object):
    def __init__(
        self,
        root="",
        dataset_name=None,
        height=256,
        width=128,
        transforms='random_flip',
        batch_size_train=32,
        batch_size_test=32,
        workers=4
    ):
        self.root = root
        self.dataset_name = dataset_name
        self.height = height
        self.width = width

        self.transform_train, self.transform_test = build_transforms(
            self.height,
            self.width,
            transforms=transforms,
        )

        self._prepare_data(dataset_name)
        self._prepare_loader(batch_size_train, batch_size_test, workers)

        self._num_train_pids = self.trainset.num_train_pids
        self._num_train_cams = self.trainset.num_train_cams

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def _prepare_data(self, dataset_name):
        print('=> Loading train (source) dataset')
        self.trainset = init_image_dataset(
            name=dataset_name,
            root=self.root,
            mode='train',
            transform=self.transform_train
        )

        self.queryset = init_image_dataset(
            name=dataset_name,
            root=self.root,
            mode='query',
            transform=self.transform_test
        )

        self.galleryset = init_image_dataset(
            name=dataset_name,
            root=self.root,
            mode='gallery',
            transform=self.transform_test
        )

    def _prepare_loader(self, batch_size_train, batch_size_test, workers):
        self.trainloader = DataLoader(
            dataset=self.trainset,
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=workers
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
