from PIL import Image
import sys
import time
import soundfile as sf
import os
import torch
import platform

if torch.cuda.is_available():
    log_device = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "").replace(" GPU", "")
else:
    log_device = platform.processor()


from live_photo_commentary.local_describer import LocalDescriber
from live_photo_commentary.synthesizer import Synthesizer

model_id = sys.argv[1]
if len(sys.argv) > 2:
    max_history = sys.argv[2]
else:
    max_history = 5

images = []
tests_dir = "tests"
for filename in os.listdir(tests_dir):
    if filename.lower().endswith(".jpg"):
        images.append(os.path.join(tests_dir, filename))

kwargs = {
    "max_history_size": max_history,
}
# if model_id == "microsoft/Phi-4-multimodal-instruct":
#     kwargs["load_in_4bit"] = False
# else:
#     kwargs["load_in_4bit"] = True
 
describer = LocalDescriber(model_id=model_id, **kwargs)
synthesizer = Synthesizer()
sample_rate = synthesizer.sample_rate()

prev_image = None
prev_file = ""
log_images = ""
history = -1
for jpg_file in images:
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

    print(log_device + "\t" + model + "\t" + size + "\t" + str(text_time) + "\t" + str(speech_time) + "\t" + log_images.replace("tests/","") 
    + "\t" + str(history) + "\t" + str(max_history) + "\t" + describer_output.replace("\n", " "))