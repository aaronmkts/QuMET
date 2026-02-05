import torch
from torchvision import transforms as tv_transforms

# MNIST
# -----------------------------------------


def lutnet_transform(img):
    """
    MNIST provided by PyTorch is a list of images with pixel values in the range [0, 1]

    MNIST provided by TensorFlow is a list of images with pixel values in the range [0, 255]

    This function converts the PyTorch representation to the TensorFlow representation, in order to recreate the results from the LUTNet paper
    """
    img = img * 255
    return img


def normalise_transform(img):

    img = img / img.sum()
    return img


def _get_mnist_default_transform():
    transform_list = [
        tv_transforms.ToTensor()
    ]  # , tv_transforms.Normalize((0.5,), (0.5,))]
    transform = tv_transforms.Compose(transform_list)
    return transform


def _get_mnist_patchgan_transform():
    transform_list = [tv_transforms.ToTensor(), tv_transforms.Resize(8)]
    transform = tv_transforms.Compose(transform_list)

    return transform


def _get_mnist_probsqgan_transform():
    transform_list = [
        tv_transforms.Resize((32, 32)),
        tv_transforms.ToTensor(),
        tv_transforms.Lambda(torch.flatten),
        tv_transforms.Lambda(normalise_transform),
    ]
    transform = tv_transforms.Compose(transform_list)

    return transform


def _get_mnist_mosaiq_transform():
    transform_list = [tv_transforms.ToTensor(), tv_transforms.Lambda(torch.flatten)]
    transform = tv_transforms.Compose(transform_list)

    return transform


def get_mnist_default_transform(train: bool) -> tv_transforms.Compose:
    return _get_mnist_default_transform()


def get_mnist_patchgan_transform(train: bool) -> tv_transforms.Compose:
    return _get_mnist_patchgan_transform()


def get_mnist_probsqgan_transform(train: bool):
    return _get_mnist_probsqgan_transform()


def get_mnist_mosaiq_transform(train: bool):
    return _get_mnist_mosaiq_transform()


def get_mnist_transform(train: bool, model: str = None):

    match model:
        case "patchgan":
            return get_mnist_patchgan_transform(train)
        case "mosaiq":
            return get_mnist_mosaiq_transform(train)
        case "qgan_probs":
            return get_mnist_probsqgan_transform(train)
        case _:
            return get_mnist_default_transform(train)
