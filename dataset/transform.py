from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
)


def build_transforms(
    height=256,
    width=128,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
):
    """Builds train and test transform functions.
    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building train transforms ...')
    transform_train = []
    transform_train += [Resize((height, width))]
    if 'random_flip' in transforms:
        transform_train += [RandomHorizontalFlip()]
    if 'color_jitter' in transforms:
        transform_train += [
            ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)
        ]
    transform_train += [ToTensor()]
    transform_train += [normalize]
    transform_train = Compose(transform_train)

    print('Building test transforms ...')
    transform_test = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_train, transform_test
