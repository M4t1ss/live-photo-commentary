from PIL import Image
import sys
import time
import soundfile as sf
import torch
import platform
import argparse
import random
from transformers import set_seed
import numpy as np

from live_photo_commentary.local_describer import LocalDescriber
from live_photo_commentary.synthesizer import Synthesizer

def set_all_seed(seed=347155):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

rand_num = random.randint(0, 999999)
set_all_seed(rand_num)

model_id = sys.argv[1]
image_file = sys.argv[2]
if len(sys.argv) > 3:
    image_file_2 = sys.argv[3]
else:
    image_file_2 = False

kwargs = {
    "max_history_size": 0,
}

if torch.cuda.is_available():
    log_device = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").replace(" GPU", "")
else:
    log_device = platform.processor()
 
describer = LocalDescriber(model_id=model_id, **kwargs)
synthesizer = Synthesizer()

image = Image.open(image_file)
image.load()

start = time.perf_counter()
if not image_file_2:
    describer_output = describer(image)
else:
    image_2 = Image.open(image_file_2)
    image_2.load()
    describer_output = describer(image, image_2)
    image_file += " " + image_file_2
text_time = time.perf_counter() - start

sample_rate = synthesizer.sample_rate()
start = time.perf_counter()
synth_output = synthesizer(describer_output)

cnt = 0
for gs, _, segment in synth_output:
    sf.write('tests/audio-'+str(cnt)+'.wav', segment, sample_rate)
    cnt += 1

speech_time = time.perf_counter() - start

model = size = ""
if "FastVLM" in model_id:
    model = "FastVLM"
    if "7B" in model_id:
        size = "7B"
    elif "1.5B" in model_id:
        size = "1.5B"
    elif "0.5B" in model_id:
        size = "0.5B"
elif "Phi-3.5" in model_id:
    model = "Phi-3.5"
    size = "4.2B"
elif "Phi-4" in model_id:
    model = "Phi-4"
    size = "5.6B"
elif "gemma" in model_id:
    model = "Gemma"
    if "4b" in model_id:
        size = "4B"
    elif "12b" in model_id:
        size = "12B"
    elif "27b" in model_id:
        size = "27B"
elif "Qwen2.5-VL" in model_id:
    model = "Qwen2.5-VL"
    if "3B" in model_id:
        size = "3B"
    elif "7B" in model_id:
        size = "7B"
    elif "32B" in model_id:
        size = "32B"
    elif "72B" in model_id:
        size = "72B"

print(log_device + "\t" + model + "\t" + size + "\t" + str(text_time) + "\t" + str(speech_time) + "\t" + image_file.replace("tests/","") + "\t" + describer_output.replace("\n", " "))