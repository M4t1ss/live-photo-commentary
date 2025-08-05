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
from IPython import get_ipython  # type: ignore[import]
from IPython.display import display, HTML, Javascript, Image

from live_photo_commentary.screenshot import screenshot, difference


timer_format = '{:.1f}'
screenshot_height = 265
screenshot_margin = 10
waiting_image = 'gifs/waiting_bg.gif'
animation_length = 9.0



def shift(l, default=None):
    try:
        return l.pop(0)
    except IndexError:
        return default


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


def pil_to_image(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return Image(data=buf.getvalue())


def pil_resize_to_height(img, height):
    original_width, original_height = img.size
    width = int((height / original_height) * original_width)
    resized_img = img.resize((width, height))
    return resized_img


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
            difference_threshold=None, difference_measure='mse', difference_kwargs=None
        ):
        self.instance_id = str(uuid.uuid4()).replace('-', '')
        self.describer = describer
        self.synthesizer = synthesizer
        self.extra_delay = extra_delay
        self.log_path = logdir and Path(logdir)
        self.screenshot_kwargs = screenshot_kwargs or {}
        self.crop = crop
        self.difference_threshold = difference_threshold
        self.difference_measure = difference_measure
        self.difference_kwargs = difference_kwargs or {}

        self.audio_id = f'audio_{self.instance_id}'
        self.img_id = f'img_{self.instance_id}'
        self.looping = False
        self.images = None
        self.thread = None
        self.images_thread = None

        button_style = dict(button_color='white', font_weight='bold', font_size='16px')
        button_layout = widgets.Layout(width='80px', height='35px')
        self.btn_start = widgets.Button(description="Loop", layout=button_layout, style=button_style)
        self.btn_stop = widgets.Button(description="Stop", layout=button_layout, style=button_style)
        self.btn_wave = widgets.Button(description="Wave", layout=button_layout, style=button_style)
        self.btn_look = widgets.Button(description="Look", layout=button_layout, style=button_style)
        self.btn_dance = widgets.Button(description="Dance", layout=button_layout, style=button_style)
        self.btn_wait = widgets.Button(description="Wait", layout=button_layout, style=button_style)
        self.timer = widgets.Label(
            value='[📎]', 
            layout=widgets.Layout(align_self='center', display='flex', justify_content='flex-start', font_size='26px !important', width='200px'),
            style=dict(font_size='26px', font_weight='bold'),
        )
        self.output_image = widgets.Output(layout={'height': '550px'})
        self.textbox = widgets.HTML()
        self.javscr = widgets.Output()
        button_hbox = widgets.HBox(
            (self.btn_start, self.btn_stop, self.btn_wave, self.btn_look, self.btn_dance, self.btn_wait, self.timer),
            layout=widgets.Layout(align_items='center')
        )
        screenshot_layout = dict(height=f'{screenshot_height}px')
        self.curr_screenshot = widgets.Output(layout=screenshot_layout)
        self.prev_screenshot = widgets.Output(layout=dict(margin=f'{screenshot_margin}px 0 0 0', **screenshot_layout))
        screenshot_vbox = widgets.VBox(
            (self.curr_screenshot, self.prev_screenshot),
        )
        image_hbox = widgets.HBox(
            (self.output_image, screenshot_vbox)
        )

        display(button_hbox, image_hbox, self.textbox, self.javscr)

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


    def set_text(self, gs_list):
        self.textbox.value = ''.join(
            f'<div style="line-height: inherit; font-size: 16px">{gs}</div>'
            for gs in gs_list
        )

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
        self.images = None
        self.image_thread = threading.Thread(target=self.run_images)
        self.image_thread.start()

        self.btn_stop.disabled = False

    def stop_loop(self, _):
        if not self.looping:
            return
        self.btn_stop.disabled = True

        self.looping = False
        if self.images_thread:
            self.images_thread.join()
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
        sample_rate = self.synthesizer.sample_rate()
        time_string = datetime.now().strftime("%Y-%m-%d_%H-%M")
        if self.log_path:
            logfile_path = self.log_path / (time_string + ".log.txt")
            log_context = logfile_path.open('wt')
        else:
            log_context = nullcontext()
        curr_screenshot = None
        prev_screenshot = None
        with log_context as logfile:
            while self.looping:
                # count-in
                countdown_end = time.perf_counter() + self.extra_delay
                while True:
                    if not self.looping:
                        self.timer.value = ''
                        return

                    remaining_seconds = countdown_end - time.perf_counter()
                    if remaining_seconds <= 0:
                        break

                    time.sleep(0.1)
                    self.timer.value = f'[⏳: {timer_format}]'.format(remaining_seconds)

                self.timer.value = '[🖼️]'

                # screenshot
                before = time.perf_counter()
                new_screenshot = screenshot()
                if curr_screenshot and self.difference_threshold:
                    diff = difference(curr_screenshot, new_screenshot, measure=self.difference_measure, **self.difference_kwargs)
                    if logfile:
                        logfile.write(f'[screenshot diff ({self.difference_measure} threshold {self.difference_threshold}): {diff}]\n')
                        logfile.flush()
                    if diff < self.difference_threshold:
                        continue

                prev_screenshot = curr_screenshot
                curr_screenshot = screenshot(**self.screenshot_kwargs)
                if self.crop:
                    curr_screenshot = curr_screenshot.crop(self.crop)

                if logfile:
                    elapsed_seconds = time.perf_counter() - before
                    logfile.write(f'[screenshot {elapsed_seconds}]\n')
                    logfile.flush()

                # image processing and display
                self.curr_screenshot.outputs = ()
                curr_image = pil_to_image(pil_resize_to_height(curr_screenshot, screenshot_height))
                self.curr_screenshot.append_display_data(curr_image)
                if prev_screenshot:
                    self.prev_screenshot.outputs = ()
                    prev_image = pil_to_image(pil_resize_to_height(prev_screenshot, screenshot_height))
                    self.prev_screenshot.append_display_data(prev_image)

                # description
                self.timer.value = '[📝]'
                before = time.perf_counter()
                text = self.describer(curr_screenshot, prev_screenshot)
                if logfile:
                    elapsed_seconds = time.perf_counter() - before
                    logfile.write(f"[describer {elapsed_seconds}]\n")
                    logfile.flush()

                # synthesis
                generator = self.synthesizer(text)
                gs_list = []
                while True:
                    # get segment from TTS
                    try:
                        before = time.perf_counter()
                        self.timer.value = '[🎙️]'
                        gs, _, segment = next(generator)
                    except StopIteration:
                        break
                    gs_list.append(gs)
                    self.set_text(gs_list)

                    if logfile:
                        elapsed_seconds = time.perf_counter() - before
                        logfile.write(f"[synthesizer {elapsed_seconds}]\n")
                        logfile.write(text.replace("\n", "") + "\n")
                        logfile.flush()

                    self.images = decide_gif(gs)
                    audio_url = segment_to_data_url(segment, sample_rate)
                    js = f"""
                        window.{self.audio_id}.src = {json.dumps(audio_url)}
                        window.{self.audio_id}.play()
                    """
                    self.run_js(js)

                    # wait till synthesized clip is finished playing
                    duration = len(segment) / sample_rate
                    playback_end = time.perf_counter() + duration
                    while True:
                        if not self.looping:
                            js = f"""
                                window.{self.audio_id}.pause()
                            """
                            self.run_js(js)
                            break

                        remaining_seconds = playback_end - time.perf_counter()
                        if remaining_seconds <= 0:
                            break

                        time.sleep(0.1)
                        self.timer.value = f'[🔊: {timer_format}]'.format(remaining_seconds)

                    self.timer.value = ''
                self.timer.value = ''


    def run_images(self):
        self.fade_to_image(waiting_image)
        images = []
        while self.looping:
            if self.images:
                images, self.images = self.images, None

            new_image = shift(images, waiting_image)
            self.fade_to_image(new_image)
            time.sleep(animation_length)
