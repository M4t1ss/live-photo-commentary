import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _is_wsl():
    try:
        with Path('/proc/version').open('r') as f:
            return 'microsoft' in f.read().lower()
    except FileNotFoundError:
        return False


def _screenshot_with_mss(path=None):
    with _mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])
        image = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
    if path is not None:
        image.save(path)
    return image


def _get_screenshot_exe_path():
    path = Path(__file__).parent.parent.parent / 'screenshot.exe'
    if not path.is_file():
        raise FileNotFoundError(f'screenshot.exe not found at {path}')
    return path


def _screenshot_with_exe(path=None):
    exe_path = _get_screenshot_exe_path()

    use_temp = path is None
    if use_temp:
        fd, temp_file = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        out_path = Path(temp_file)
    else:
        out_path = Path(path)

    try:
        win_path = subprocess.run(
            ['wslpath', '-w', str(out_path)],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        subprocess.run(
            [str(exe_path), win_path],
            capture_output=True, check=True, timeout=30
        )

        image = Image.open(out_path)
        image.load()
        return image

    except subprocess.TimeoutExpired:
        raise RuntimeError('screenshot.exe timed out after 30 seconds')
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f'screenshot.exe failed: {e.stderr}')
    finally:
        if use_temp:
            try:
                out_path.unlink(missing_ok=True)
            except OSError:
                pass


if _is_wsl():
    from PIL import Image
    screenshot = _screenshot_with_exe
else:
    import mss as _mss
    from PIL import Image
    screenshot = _screenshot_with_mss


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

    arr1 = np.array(image1)
    arr2 = np.array(image2)
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
