from .market1501 import  Market1501
from .dukemtmcreid import  DukeMTMCreID

__image_datasets = {
    'dukemtmcreid': DukeMTMCreID,
    'market1501': Market1501,
}

def init_image_dataset(name, **kwargs):
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name](**kwargs)