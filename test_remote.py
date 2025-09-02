from PIL import Image
import sys
import time
import soundfile as sf
import os

from live_photo_commentary.remote_describer import RemoteDescriber
from live_photo_commentary.synthesizer import Synthesizer

gemini_api_key = os.environ["GEMINI_API_KEY"]
openai_api_key = os.environ["OPENAI_API_KEY"]

system = sys.argv[1]
image_file = sys.argv[2]
if len(sys.argv) > 3:
    image_file_2 = sys.argv[3]
else:
    image_file_2 = False

kwargs = {
    "max_history_size": 5,
}
 
if system == "Gemini":
    describer = RemoteDescriber(provider="gemini", model_id="gemini-2.5-flash-lite", api_key=gemini_api_key)
elif system == "GPT":
    describer = RemoteDescriber(provider="openai", model_id="gpt-4o-mini", api_key=openai_api_key)
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

if "Gemini" in system:
    model = "Gemini"
    size = "2.5-flash-lite"
elif "GPT" in system:
    model = "GPT"
    size = "4o-mini"

print(model + "\t" + size + "\t" + str(text_time) + "\t" + str(speech_time) + "\t" + image_file.replace("tests/","") + "\t" + describer_output.replace("\n", " "))