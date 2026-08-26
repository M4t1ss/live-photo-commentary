# Live Photo Commentary

A script workflow for automated live spoken commenting on photos shown on screen.

At the moment it is configured to run image understaning via the Gemini API (available for free for now...) or use the Phi 3.5 vision-instruct model locally. Speech is generated using the Kokoro-82M model.

## Supported Models

* microsoft/Phi-3.5-vision-instruct
	* Requires `pip install transformers<4.41 accelerate==0.30.0`
* microsoft/Phi-4-multimodal-instruct
	* Requires `transformers==4.51.3 peft backoff`
	* Does not work with quanitzation, so neeeds to have **load_in_4bit** set to False  `LocalDescriber(model_id="microsoft/Phi-4-multimodal-instruct", model_kwargs={ "load_in_4bit": False })`
* google/gemma-3-4b-it
	* The larger versions should also run with more capable GPUs
* Qwen/Qwen2.5-VL-3B-Instruct
	* The larger versions should also run with more capable GPUs
* apple/FastVLM-0.5B
* apple/FastVLM-1.5B
* apple/FastVLM-7B
* Google Gemini API
* OpenAI GPT API

## Usage Examples

The main example is shown in the [live_photo_commentary.ipynb](https://github.com/M4t1ss/live-photo-commentary/blob/main/live_photo_commentary.ipynb) IPython notebook, which uses Gemini, uses the full screenshot does not ignore identical screenshots, keeps up to 5 previous historical responses and compresses them after more are generated, and saves output logs in the `logs` directory.

The following is an example with running a local model, does not maintain response history, adds a 20 second delay between responses, crops the screenshot to the [bounding box](https://chayanvinayak.blogspot.com/2013/03/bounding-box-in-pilpython-image-library.html) (185,125,1440,965), and sets a difference threshold for screenshot similarity to not generate a response when the screen has not changed.
```python
%load_ext autoreload
%autoreload 3

from live_photo_commentary.synthesizer import Synthesizer
synthesizer = Synthesizer()

from live_photo_commentary.local_describer import LocalDescriber
describer = LocalDescriber(model_id="microsoft/Phi-3.5-vision-instruct", max_history_size=0)

from live_photo_commentary.notebook import UI
ui = UI(describer, synthesizer, logdir='logs', extra_delay=20, 
        screenshot_kwargs=dict(bbox=(185,125,1440,965)), difference_threshold=0.0002)
```

The default system and user prompts are specifically composed for commenting *photos* takend by a *photographer*, but these can aslo be adjusted or changed as in the following example
```python
from live_photo_commentary.describer import DEFAULT_PROMPT, DEFAULT_ENDING, DEFAULT_FIRST_PROMPT, DEFAULT_HISTORY_PROMPT, DEFAULT_COMPACT_PROMPT
from live_photo_commentary.local_describer import LocalDescriber

prompt = DEFAULT_PROMPT.replace('photographer', 'artist').replace('photo', 'artwork')
describer = LocalDescriber(model_id="microsoft/Phi-3.5-vision-instruct", prompt=prompt, system_prompt=DEFAULT_SYSTEM_PROMPT, ending=DEFAULT_ENDING, first_prompt=DEFAULT_FIRST_PROMPT,history_prompt=DEFAULT_HISTORY_PROMPT, compact_prompt=DEFAULT_COMPACT_PROMPT)
```


Publications
---------

If you use this tool in your work, please cite the following paper:

Matīss Rikters, Goran Topić (2025). "[Real-time Commentator Assistant for Photo Editing Live Streaming.](https://aclanthology.org/2025.ijcnlp-demo.3/)" In Proceedings of The 14th International Joint Conference on Natural Language Processing and The 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations, pages 17–24, Mumbai, India. Association for Computational Linguistics (2025).

```bibtex
@inproceedings{rikters-topic-2025-real,
    title = "Real-time Commentator Assistant for Photo Editing Live Streaming",
    author = "Rikters, Mat{\={i}}ss and Topi{\'c}, Goran",
    editor = "Liu, Xuebo and Purwarianti, Ayu",
    booktitle = "Proceedings of The 14th International Joint Conference on Natural Language Processing and The 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-demo.3/",
    pages = "17--24",
    ISBN = "979-8-89176-301-2",
}
```

## Example Videos

Recent example with an animated avatar.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ol3w4FaAuZo/0.jpg)](https://www.youtube.com/watch?v=ol3w4FaAuZo)

Initial example without an animated avatar.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/07sU403hMWE/0.jpg)](https://www.youtube.com/watch?v=07sU403hMWE)

## Speed And Compatibility Tests

The following table shows results in total seconds from testing the description generation on several selected consumer devices. The cases with a `-` denote unsuccesful runs for the specific device and model configuration.

| Model           	| Phi-3.5 	|  Phi-4  	|  Gemma 	|        	| Qwen2.5-VL 	|       	|       	| FastVLM 	|       	|
|-----------------	|--------:	|--------:	|-------:	|-------:	|-----------:	|------:	|------:	|--------:	|------:	|
| Size            	|    4.2B 	|    5.6B 	|     4B 	|    12B 	|         3B 	|    7B 	|  0.5B 	|    1.5B 	|    7B 	|
| RTX 3090        	|    9.77 	|   14.19 	|  25.93 	|  31.37 	|       9.91 	| 10.24 	|  9.06 	|   11.35 	| 12.11 	|
| GTX 1650 Laptop 	|  110.97 	|       - 	| 548.28 	|      - 	|     247.21 	|     - 	| 13.68 	|       - 	|     - 	|
| RTX 3070 Laptop 	|   10.95 	| 1360.62 	|  25.53 	| 421.02 	|      12.41 	| 12.38 	|  6.74 	|    8.37 	| 12.93 	|
| RTX 4070 Laptop 	|   13.58 	| 1270.20 	|  34.56 	| 510.24 	|      19.33 	| 32.87 	| 10.98 	|   16.00 	| 25.78 	|
| RTX 4090 Laptop 	|   10.00 	|   11.03 	|  25.15 	|  32.22 	|      11.11 	| 10.56 	|  8.69 	|   11.64 	| 11.89 	|
| M3 Pro 18GB     	|  103.43 	|       - 	|  38.34 	|      - 	|      20.67 	|     - 	|  7.32 	|   12.87 	|     - 	|
| Average         	|   43.12 	|  664.01 	| 116.30 	| 248.72 	|      53.44 	| 16.51 	|  9.41 	|   12.04 	| 15.68 	|

The different models also tend to generate outputs in different lengths

|            	|         	| 1 Image 	| 2 Images 	|
|------------	| -------:	|:-------:	|:--------:	|
| Phi-3.5    	| 4.2B    	|   363.1 	|    630.9 	|
| Phi-4      	| 5.6B    	|   482.2 	|    659.1 	|
| Gemma      	| 4B      	|   537.3 	|    727.1 	|
|            	| 12B     	|   431.9 	|    474.6 	|
| Qwen2.5-VL 	| 3B      	|   457.6 	|    598.5 	|
|            	| 7B      	|   407.4 	|    639.9 	|
|            	| 0.5B    	|   685.9 	|    779.0 	|
| FastVLM    	| 1.5B    	|   706.1 	|    891.5 	|
|            	| 7B      	|   668.0 	|    788.7 	|
|            	| Average 	|   526.6 	|    687.7 	|

There is also considerable overlap sometimes between the last outputs, especially for smaller models when using history.

|            	|                	| History 0 	|        	| History 1 	|        	| History 5 	|        	|
|------------	|---------------:	|----------:	|-------:	|----------:	|-------:	|----------:	|-------:	|
| Model      	|           Size 	|    tok    	|   chr  	|    tok    	|   chr  	|    tok    	|   chr  	|
| Phi-3.5    	|           4.2B 	|   58.22%  	|  8.77% 	|   90.93%  	| 63.01% 	|   89.47%  	| 43.23% 	|
| Phi-4      	|           5.6B 	|   57.29%  	|  7.08% 	|   96.10%  	| 85.32% 	|   96.10%  	| 54.88% 	|
| Gemma      	|             4B 	|   54.65%  	| 11.28% 	|   76.63%  	| 23.33% 	|   69.97%  	| 23.06% 	|
|            	|            12B 	|   50.15%  	| 10.68% 	|   54.17%  	| 11.75% 	|   45.66%  	| 10.69% 	|
| Qwen2.5-VL 	|             3B 	|   51.54%  	|  7.03% 	|   89.52%  	| 50.77% 	|   79.77%  	| 54.38% 	|
|            	|             7B 	|   48.98%  	|  7.65% 	|   50.65%  	|  7.47% 	|   51.33%  	|  7.70% 	|
| FastVLM    	|           0.5B 	|   56.76%  	| 10.20% 	|   88.54%  	| 42.77% 	|   87.45%  	| 51.12% 	|
|            	|           1.5B 	|   58.87%  	|  9.52% 	|   96.46%  	| 62.42% 	|   98.24%  	| 64.70% 	|
|            	|             7B 	|   57.31%  	|  7.86% 	|   94.90%  	| 26.79% 	|   90.52%  	| 27.40% 	|
| Gemini     	| 2.5-flash-lite 	|   36.50%  	|  7.32% 	|   40.76%  	|  8.39% 	|   41.67%  	|  6.21% 	|
| GPT        	|        4o-mini 	|   44.72%  	| 11.21% 	|   49.67%  	| 10.34% 	|   49.89%  	| 10.74% 	|


