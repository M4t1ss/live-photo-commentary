import os
import platform
import urllib.request
import zipfile
from pathlib import Path
import tempfile
import numpy as np



def _is_wsl():
    try:
        with Path('/proc/version').open('r') as f:
            version_info = f.read().lower()
            return 'microsoft' in version_info or 'wsl' in version_info
    except FileNotFoundError:
        return False


def _screenshot_with_pil(bbox=None, include_layered_windows=False, all_screens=False, xdisplay=None, window=None):
    screenshot = ImageGrab.grab(
        bbox=bbox, include_layered_windows=include_layered_windows, all_screens=all_screens, xdisplay=xdisplay, window=window
    )
    return screenshot


def _get_nircmd_path():
    # Determine architecture
    is_64bit = platform.machine().endswith('64') or platform.architecture()[0] == '64bit'

    if is_64bit:
        name = 'nircmd-x64'
    else:
        name = 'nircmd-x32'

    download_url = f'https://nircmd.com/wp-content/uploads/2025/01/{name}.zip'

    # Check if nircmd.exe already exists
    current_dir = Path(__file__).parent
    nircmd_dir = current_dir / name
    nircmd_path = nircmd_dir / "nircmd.exe"

    if nircmd_path.is_file():
        return nircmd_path

    # Download and extract the ZIP file
    zip_path = current_dir / f"{name}.zip"
    nircmd_dir.mkdir(exist_ok=True)
    urllib.request.urlretrieve(download_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(nircmd_dir)
    zip_path.unlink()

    if not nircmd_path.is_file():
        raise Exception(f'nircmd.exe not found after extraction in {nircmd_dir}')
    return nircmd_path


def _screenshot_with_nircmd(bbox=None, include_layered_windows=False, all_screens=False, xdisplay=None, window=None):
    unsupported = []
    if include_layered_windows:
        unsupported.append('include_layered_windows')
    if xdisplay:
        unsupported.append('xdisplay')
    if window:
        unsupported.append('window')
    if unsupported:
        s = 's' if len(unsupported) > 1 else ''
        unsupported_str = ', '.join(f'`{arg}`' for arg in unsupported)
        raise TypeError(f'Unsupported argument{s}: {unsupported_str}')

    if bbox:
        if all_screens:
            raise ArgumentError('Cannot use both `bbox` and `all_screens`')
        left, upper, right, lower = bbox
        bbox = [
            str(coord)
            for coord in [left, upper, right - left, lower - upper]
        ]
    else:
        bbox = []

    subcommand = 'savescreenshotfull' if all_screens else 'savescreenshot'

    nircmd_path = _get_nircmd_path()

    try:
        cwd = os.getcwd()
        fd, temp_file = tempfile.mkstemp(suffix='.png', dir=cwd)
        os.close(fd)
    except Exception as x:
        raise Exception(f'Cannot make a tempfile: {x}')

    temp_path = Path(temp_file)

    try:
        cmd = [str(nircmd_path), subcommand, str(temp_path.relative_to(cwd)), *bbox]
        subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=30
        )

        if not temp_path.exists():
            raise Exception('Screenshot file was not created by nircmd')

        image = Image.open(temp_path)
        image.load()

        return image

    except subprocess.TimeoutExpired:
        raise Exception('nircmd.exe timed out after 30 seconds')
    except subprocess.CalledProcessError as x:
        raise Exception(f'nircmd.exe failed with return code {x.returncode}: {x.stderr}')
    except Exception as x:
        raise Exception(f'Failed to process screenshot: {x}')
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass


# (left, upper, right, lower)
if _is_wsl():
    from PIL import Image
    import subprocess
    screenshot = _screenshot_with_nircmd
else:
    from PIL import ImageGrab, Image
    screenshot = _screenshot_with_pil


_imagehash_measures = { 'average_hash', 'phash', 'phash_simple', 'dhash', 'dhash_vertical', 'whash', 'colorhash' }
def difference(image1, image2, measure='mse', *args, **kwargs):
    if measure == 'mse':
        return mse_difference(image1, image2)
    if measure == 'ssim':
        return ssim_difference(image1, image2, *args, **kwargs)
    if measure in _imagehash_measures:
        return imagehash_difference(image1, image2, measure, *args, **kwargs)
    if callable(measure):
        return measure(image1, image2, *args, **kwargs)
    raise ValueError('Unsupported similarity measure: {measure}')
    

def mse_difference(image1, image2):
    import numpy as np

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
    if mode in { 'L', 'I', 'F', '1' }:
        return None
    elif mode in { 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV' }:
        return -1
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


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
    sim_score: float = ssim(arr1, arr2, *args, **kwargs) # pyright: ignore[reportAssignmentType]
    return (1 - sim_score) / 2


# needs `pip install imagehash`
def imagehash_difference(image1, image2, measure, *args, **kwargs):
    import imagehash

    hash_func = getattr(imagehash, measure)

    hash1 = hash_func(image1, *args, **kwargs)
    hash2 = hash_func(image2, *args, **kwargs)
    return (hash1 - hash2) / len(hash1.hash) ** 2
