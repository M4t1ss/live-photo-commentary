import json
import uuid
import base64
import random
from datetime import datetime, timedelta
import time
from pathlib import Path
from contextlib import nullcontext
import threading
import io
import torchaudio

import ipywidgets as widgets
from IPython.display import display, HTML, Javascript

from live_photo_commentary.screenshot import screenshot


thinking_label = '[Thinking]'
timer_format = '[{:.1f}]'


def decide_gif(text):
    text = text.lower()
    outputs = []
    talking = ["gtalking_bg.gif","talking_bg.gif","talking2_bg.gif","talking3_bg.gif","ctalking_bg.gif"]
    
    if any(i in text for i in ["hello", "greet", "waving", "waves"]):
        outputs.append("gifs/"+"waving"+"_bg.gif")
    if any(i in text for i in ["scar", "creep", "fright", "spook"]):
        outputs.append("gifs/"+"scary"+"_bg.gif")
    if any(i in text for i in ["love", "cute", "nice", "like"]):
        outputs.append("gifs/"+"lovely"+"_bg.gif")
    if any(i in text for i in ["interest", "think", "wonder", "thought"]):
        outputs.append("gifs/"+"lovely"+"_bg.gif")
    if any(i in text for i in ["happy", "cheer", "inspir", "shin"]):
        outputs.append("gifs/"+"happy"+"_bg.gif")

    outputs = list(set(outputs))
    outputs.append("gifs/"+random.choice(talking))
    random.shuffle(outputs)
    return outputs


def img_file_to_data_uri(path):
    with open(path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        ext = path.split(".")[-1]
        return f"data:image/{ext};base64,{b64}"


def segment_to_data_url(segment, sample_rate):
    audio_tensor = segment.cpu().unsqueeze(0)
    wav_buffer = io.BytesIO()
    torchaudio.save(wav_buffer, audio_tensor, sample_rate, format="wav")
    wav_data = wav_buffer.getvalue()
    audio_url = "data:audio/wav;base64," + base64.b64encode(wav_data).decode()
    return audio_url


class UI:
    def __init__(
            self,
            describer, synthesizer,
            extra_delay=0, logdir=None,
            screenshot_kwargs=None, crop=None,
        ):
        self.instance_id = str(uuid.uuid4()).replace('-', '')
        self.describer = describer
        self.synthesizer = synthesizer
        self.extra_delay = extra_delay
        self.log_path = logdir and Path(logdir)
        self.screenshot_kwargs = screenshot_kwargs or {}
        self.crop = crop

        self.audio_id = f'audio_{self.instance_id}'
        self.img_id = f'img_{self.instance_id}'
        self.looping = False

        button_style = dict(button_color='white', font_weight='bold', font_size='16px')
        button_layout = widgets.Layout(width='80px', height='35px')
        self.btn_start = widgets.Button(description="Loop", layout=button_layout, style=button_style)
        self.btn_stop = widgets.Button(description="Stop", layout=button_layout, style=button_style)
        self.btn_wave = widgets.Button(description="Wave", layout=button_layout, style=button_style)
        self.btn_look = widgets.Button(description="Look", layout=button_layout, style=button_style)
        self.btn_dance = widgets.Button(description="Dance", layout=button_layout, style=button_style)
        self.btn_wait = widgets.Button(description="Wait", layout=button_layout, style=button_style)
        self.timer = widgets.Label(
            value='', 
            layout=widgets.Layout(align_self='center', display='flex', justify_content='center', width='100px'),
            style=dict(font_size='16px', font_weight='bold'),
        )
        self.output_image = widgets.Output(layout={'height': '550px'})
        self.textbox = widgets.Output(layout={'height': '100px'})
        self.javscr = widgets.Output()
        hbox = widgets.HBox(
            (self.btn_start, self.btn_stop, self.btn_wave, self.btn_look, self.btn_dance, self.btn_wait, self.timer),
            layout=widgets.Layout(align_items='center')
        )

        display(hbox, self.output_image, self.textbox, self.javscr)

        initial_uri = img_file_to_data_uri('gifs/waiting_bg.gif')

        # Show the initial image
        html = f"""
            <div>
              <img id="{self.img_id}" src="{initial_uri}" style="transition: opacity 1s ease-in-out; opacity: 1; max-width: 100%;">
            </div>
        """
        self.output_image.append_display_data(HTML(html))

        js = """
            window.fadeToImage = function fadeToImage(img_id, uri) {
                const img = document.getElementById(img_id);
                if (img) {
                    img.style.opacity = 0;
                    setTimeout(function() {
                        img.src = uri;
                        img.style.opacity = 1;
                    }, 50);
                }
            }
        """ + f"""
            window.{self.audio_id} = new Audio();
        """
        self.run_js(js)

        self.btn_stop.disabled = True

        self.btn_start.on_click(self.start_loop)
        self.btn_stop.on_click(self.stop_loop)
        self.btn_dance.on_click(self.dance)
        self.btn_look.on_click(self.look)
        self.btn_wave.on_click(self.wave)
        self.btn_wait.on_click(self.wait)


    def run_js(self, js):
        self.javscr.clear_output()
        self.javscr.outputs = []
        self.javscr.append_display_data(Javascript(js))


    def start_loop(self, _):
        if self.looping:
            return
        self.btn_start.disabled = True
        self.looping = True
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
        self.btn_stop.disabled = False

    def stop_loop(self, _):
        if not self.looping:
            return
        self.btn_stop.disabled = True
        self.looping = False
        if self.thread:
            self.thread.join()
        self.btn_start.disabled = False

    def dance(self, _):
        self.fade_to_image("gifs/xdancing_bg.gif")

    def look(self, _):
        self.fade_to_image("gifs/looking_bg.gif")

    def wave(self, _):
        self.fade_to_image("gifs/hello_bg.gif")

    def wait(self, _):
        self.fade_to_image("gifs/waiting_bg.gif")


    def fade_to_image(self, img_path):
        url = img_file_to_data_uri(img_path)
        js = f"""
            fadeToImage({json.dumps(self.img_id)}, {json.dumps(url)})
        """
        self.run_js(js)


    def run(self):
        with self.textbox:
            sample_rate = self.synthesizer.sample_rate()
            self.fade_to_image('gifs/waving_bg.gif')
            time_string = datetime.now().strftime("%Y-%m-%d_%H-%M")
            if self.log_path:
                logfile_path = self.log_path / (time_string + ".log.txt")
                log_context = logfile_path.open('wt')
            else:
                log_context = nullcontext()
            self.looping = True
            curr_screenshot = None
            prev_screenshot = None
            animation_switch_time = timedelta(seconds=10)
            animation_end = None
            with log_context as logfile:
                while self.looping:
                    self.textbox.clear_output()
                    prev_screenshot = curr_screenshot
                    before = datetime.now()
                    curr_screenshot = screenshot(**self.screenshot_kwargs)
                    if self.crop:
                        curr_screenshot = curr_screenshot.crop(self.crop)

                    if logfile:
                        elapsed_seconds = (datetime.now() - before).total_seconds()
                        logfile.write(f"[screenshot {elapsed_seconds}]\n")
                        logfile.flush()

                    self.timer.value = thinking_label
                    before = datetime.now()
                    text = self.describer(curr_screenshot, prev_screenshot)
                    if logfile:
                        elapsed_seconds = (datetime.now() - before).total_seconds()
                        logfile.write(f"[describer {elapsed_seconds}]\n")
                        logfile.flush()

                    generator = self.synthesizer(text)
                    while True:
                        try:
                            before = datetime.now()
                            gs, _, segment = next(generator)
                        except StopIteration:
                            break
                        if logfile:
                            elapsed = datetime.now() - before
                            elapsed_seconds = (datetime.now() - before).total_seconds()
                            logfile.write(f"[synthesizer {elapsed_seconds}]\n")
                            logfile.write(text.replace("\n", "") + "\n")
                            logfile.flush()

                        print(gs)
                        images = decide_gif(gs)
                        audio_url = segment_to_data_url(segment, sample_rate)
                        js = f"""
                            window.{self.audio_id}.src = {json.dumps(audio_url)}
                            window.{self.audio_id}.play()
                        """
                        self.run_js(js)

                        duration = len(segment) / sample_rate
                        now = datetime.now()
                        countdown_end = now + timedelta(seconds=duration + self.extra_delay)
                        if animation_end is None:
                            animation_end = now
                        animation_end += animation_switch_time
                        while True:
                            if not self.looping:
                                js = f"""
                                    window.{self.audio_id}.pause()
                                """
                                self.run_js(js)
                                self.timer.value = ''
                                break

                            now = datetime.now()
                            if animation_end and now > animation_end:
                                animation_end = now + animation_switch_time
                                animation_end = None
                            if animation_end is None and images:
                                next_image = images.pop(0)
                                self.fade_to_image(next_image)
                                animation_end = now

                            remaining_seconds = (countdown_end - now).total_seconds()
                            if remaining_seconds <= 0:
                                break

                            time.sleep(0.1)
                            self.timer.value = timer_format.format(remaining_seconds)

                        self.timer.value = ''
                        self.textbox.clear_output()
                        self.textbox.output = []
