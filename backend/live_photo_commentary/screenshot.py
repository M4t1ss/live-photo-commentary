import numpy as np
from PIL import Image

_imagehash_measures = {'average_hash', 'phash', 'phash_simple', 'dhash', 'dhash_vertical', 'whash', 'colorhash'}


def difference(image1, image2, measure='mse', *args, **kwargs):
    if measure == 'mse':
        return mse_difference(image1, image2)
    if measure == 'ssim':
        return ssim_difference(image1, image2, *args, **kwargs)
    if measure in _imagehash_measures:
        return imagehash_difference(image1, image2, measure, *args, **kwargs)
    if callable(measure):
        return measure(image1, image2, *args, **kwargs)
    raise ValueError(f'Unsupported similarity measure: {measure}')


def mse_difference(image1, image2):
    if image1.size != image2.size:
        image1 = image1.resize(image2.size)
    if image1.mode != 'RGB':
        image1 = image1.convert('RGB')
    if image2.mode != 'RGB':
        image2 = image2.convert('RGB')

    arr1 = np.array(image1, dtype=np.float32)
    arr2 = np.array(image2, dtype=np.float32)
    return np.mean((arr1 - arr2) ** 2) / (256 ** 2)


def _get_channel_axis(mode):
    if mode in {'L', 'I', 'F', '1'}:
        return None
    elif mode in {'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV'}:
        return -1
    else:
        raise ValueError(f'Unsupported image mode: {mode}')


# needs `pip install scikit-image`
def ssim_difference(image1, image2, mode='RGB', *args, **kwargs):
    from skimage.metrics import structural_similarity as ssim

    if image1.size != image2.size:
        image1 = image1.resize(image2.size)
    if image1.mode != mode:
        image1 = image1.convert(mode)
    if image2.mode != mode:
        image2 = image2.convert(mode)

    if 'channel_axis' not in kwargs:
        kwargs['channel_axis'] = _get_channel_axis(mode)

    arr1 = np.array(image1)
    arr2 = np.array(image2)
    sim_score: float = ssim(arr1, arr2, *args, **kwargs)  # pyright: ignore[reportAssignmentType]
    return (1 - sim_score) / 2


# needs `pip install imagehash`
def imagehash_difference(image1, image2, measure, *args, **kwargs):
    import imagehash

    hash_func = getattr(imagehash, measure)
    hash1 = hash_func(image1, *args, **kwargs)
    hash2 = hash_func(image2, *args, **kwargs)
    return (hash1 - hash2) / len(hash1.hash) ** 2
