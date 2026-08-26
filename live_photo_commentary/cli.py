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

    parser.add_argument('-a', '--api_key', required=False, help='API key for remote providers')
    parser.add_argument('-p', '--provider', required=False, default='gemini', choices=['gemini', 'openai'], help='Remote API provider')
    parser.add_argument('-m', '--max_history_size', type=int, required=False)
    parser.add_argument('-l', '--local', action='store_true')
    parser.add_argument('-M', '--model', required=False, help='Model to use (local models or remote model names like gpt-4o, gemini-1.5-pro)')

    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)

    params = {
        "max_history_size": args.max_history_size,
    }

    if args.local:
        from live_photo_commentary.local_describer import LocalDescriber
        if args.model:
            params["model_id"] = args.model
        describer = LocalDescriber(**params)
    else:
        from live_photo_commentary.remote_describer import RemoteDescriber
        params["provider"] = args.provider
        if args.api_key:
            params["api_key"] = args.api_key
        if args.model:
            params["model_id"] = args.model
        describer = RemoteDescriber(**params)

    from live_photo_commentary.synthesizer import Synthesizer
    synthesizer = Synthesizer()

    loop(describer, synthesizer)


if __name__ == '__main__':
    main(sys.argv[1:])
