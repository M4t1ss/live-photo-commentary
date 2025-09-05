from PIL import Image
import time
import soundfile as sf
import os
import torch
import platform
from transformers import set_seed
import argparse
import numpy as np
import random

from live_photo_commentary.remote_describer import RemoteDescriber
from live_photo_commentary.synthesizer import Synthesizer

gemini_api_key = os.environ["GEMINI_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

def set_all_seed(seed=347155):
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser=argparse.ArgumentParser()
parser.add_argument("--seed", help="Random seed")
parser.add_argument("--system", help="Model to evaluate")
parser.add_argument("--history", help="Maximum history size")

args=parser.parse_args()

if torch.cuda.is_available():
    log_device = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").replace(" GPU", "")
else:
    log_device = platform.processor()

system = args.system
if args.history != None:
    max_history = int(args.history)
else:
    max_history = 5
if args.seed != None:
    set_all_seed(args.seed)
else:
    set_all_seed(347155)

images = []
tests_dir = "tests"
for filename in os.listdir(tests_dir):
    if filename.lower().endswith(".jpg"):
        images.append(os.path.join(tests_dir, filename))

kwargs = {
    "max_history_size": max_history,
}
 
if system == "Gemini":
    describer = RemoteDescriber(provider="gemini", model_id="gemini-2.5-flash-lite", api_key=gemini_api_key)
elif system == "GPT":
    describer = RemoteDescriber(provider="openai", model_id="gpt-4o-mini", api_key=openai_api_key)
synthesizer = Synthesizer()
sample_rate = synthesizer.sample_rate()

prev_image = None
prev_file = ""
log_images = ""
history = -1
for jpg_file in images:
    rand_num = random.randint(0, 999999)
    set_all_seed(rand_num)

    image = Image.open(jpg_file)
    image.load()
    start = time.perf_counter()
    if prev_image:
        describer_output = describer(image, prev_image)
    else:
        describer_output = describer(image)
    log_images = jpg_file + " " + prev_file
    prev_image = image
    prev_file = jpg_file
    text_time = time.perf_counter() - start
    history += 1

    start = time.perf_counter()
    synth_output = synthesizer(describer_output)

    cnt = 0
    for gs, _, segment in synth_output:
        sf.write('tests/audio-'+str(cnt)+'.wav', segment, sample_rate)
        cnt += 1

    speech_time = time.perf_counter() - start

    model = size = ""
    if "Gemini" in system:
        model = "Gemini"
        size = "2.5-flash-lite"
    elif "GPT" in system:
        model = "GPT"
        size = "4o-mini"

    print(log_device + "\t" + model + "\t" + size + "\t" + str(text_time) + "\t" + str(speech_time) + "\t" + log_images.replace("tests/","") 
    + "\t" + str(history) + "\t" + str(max_history) + "\t" + str(rand_num) + "\t" + describer_output.replace("\n", " "))