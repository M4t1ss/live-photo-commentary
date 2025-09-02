from PIL import Image
import sys
import time
import soundfile as sf

from live_photo_commentary.local_describer import LocalDescriber
from live_photo_commentary.synthesizer import Synthesizer

model_id = sys.argv[1]
image_file = sys.argv[2]
if len(sys.argv) > 3:
    image_file_2 = sys.argv[3]
else:
    image_file_2 = False

kwargs = {
    "max_history_size": 5,
}
 
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

print(model + "\t" + size + "\t" + str(text_time) + "\t" + str(speech_time) + "\t" + image_file.replace("tests/","") + "\t" + describer_output.replace("\n", " "))