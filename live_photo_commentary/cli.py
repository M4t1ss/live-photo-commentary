import sys
from argparse import ArgumentParser
import time
from datetime import datetime, timedelta

import sounddevice as sd

from .screenshot import screenshot



def loop(describer, synthesizer):
    curr_screenshot = None
    prev_screenshot = None
    sample_rate = synthesizer.sample_rate()
    while True:
        prev_screenshot = curr_screenshot
        curr_screenshot = screenshot()
        text = describer(curr_screenshot, prev_screenshot)

        duration = 0
        for gs, _, segment in synthesizer(text):
            print(gs)
            sd.play(segment, sample_rate)
            duration = len(segment) / sample_rate
            playback_end = time.perf_counter() + duration
            while True:
                remaining_seconds = playback_end - time.perf_counter()
                if remaining_seconds <= 0:
                    print("\r   \r", end="")
                    break
                print(f"\r{remaining_seconds:.1f} \r{remaining_seconds:.1f}", end="")
                time.sleep(0.1)


def parse_args(args):
    parser = ArgumentParser()

    parser.add_argument('-g', '--gemini_api_key', required=False)
    parser.add_argument('-m', '--max_history_size', type=int, required=False)
    parser.add_argument('-l', '--local', action='store_true')

    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)

    params = {
        "max_history_size": args.max_history_size,
    }

    if args.local:
        from live_photo_commentary.local_describer import LocalDescriber
        describer = LocalDescriber(**params)
    else:
        from live_photo_commentary.gemini_describer import GeminiDescriber
        gemini_api_key = args.gemini_api_key
        describer = GeminiDescriber(gemini_api_key=gemini_api_key, **params)

    from live_photo_commentary.synthesizer import Synthesizer
    synthesizer = Synthesizer()

    loop(describer, synthesizer)


if __name__ == '__main__':
    main(sys.argv[1:])
