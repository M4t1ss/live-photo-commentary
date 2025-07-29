import platform
import urllib.request
import zipfile
from pathlib import Path
import tempfile
import subprocess

from PIL import ImageGrab, Image


def _is_wsl():
    try:
        with Path('/proc/version').open('r') as f:
            version_info = f.read().lower()
            return 'microsoft' in version_info or 'wsl' in version_info
    except FileNotFoundError:
        return False

def _screenshot_with_pil():
    screenshot = ImageGrab.grab()
    return screenshot


def _get_nircmd_path():
    # Determine architecture
    is_64bit = platform.machine().endswith('64') or platform.architecture()[0] == '64bit'

    if is_64bit:
        subdir = "nircmd-x64"
        download_url = "https://nircmd.com/wp-content/uploads/2025/01/nircmd-x64.zip"
    else:
        subdir = "nircmd-x32"
        download_url = "https://nircmd.com/wp-content/uploads/2025/01/nircmd-x32.zip"

    # Check if nircmd.exe already exists
    current_dir = Path(__file__).parent
    nircmd_dir = current_dir / subdir
    nircmd_path = nircmd_dir / "nircmd.exe"

    if nircmd_path.is_file():
        return nircmd_path

    # Download and extract the ZIP file
    zip_path = current_dir / f"{subdir}.zip"
    urllib.request.urlretrieve(download_url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(current_dir)
    zip_path.unlink()

    if not nircmd_path.is_file():
        raise Exception(f"nircmd.exe not found after extraction in {nircmd_dir}")
    return nircmd_path


def _screenshot_with_nircmd():
    nircmd_path = _get_nircmd_path()

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        cmd = [str(nircmd_path), 'savescreenshot', str(temp_path)]
        subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True,
            timeout=30
        )

        if not temp_path.exists():
            raise Exception("Screenshot file was not created by nircmd")

        image = Image.open(temp_path)
        image.load()

        return image

    except subprocess.TimeoutExpired:
        raise Exception("nircmd.exe timed out after 30 seconds")
    except subprocess.CalledProcessError as e:
        raise Exception(f"nircmd.exe failed with return code {e.returncode}: {e.stderr}")
    except Exception as e:
        raise Exception(f"Failed to process screenshot: {str(e)}")
    finally:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass

if _is_wsl():
    screenshot = _screenshot_with_nircmd
else:
    screenshot = _screenshot_with_pil
