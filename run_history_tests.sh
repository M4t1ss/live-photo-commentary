#!/bin/bash

pyenv activate phi4

models=("google/gemma-3-4b-it" "google/gemma-3-12b-it" "Qwen/Qwen2.5-VL-3B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" "microsoft/Phi-4-multimodal-instruct")

for model in "${models[@]}"; do
    echo "python test_history_local.py --model"$model" >> output.tsv"
    # python test_history_local.py --model $model --history 5 >> history.tsv
    python test_history_local.py --model $model --history 1 >> history.tsv
    python test_history_local.py --model $model --history 0 >> history.tsv
done

pyenv activate phi3

models=("microsoft/Phi-3.5-vision-instruct")

for model in "${models[@]}"; do
    echo "python test_history_local.py --model"$model" >> output.tsv"
    # python test_history_local.py --model $model --history 5 >> history.tsv
    python test_history_local.py --model $model --history 1 >> history.tsv
    python test_history_local.py --model $model --history 0 >> history.tsv
done

pyenv activate appl

models=("apple/FastVLM-0.5B" "apple/FastVLM-1.5B" "apple/FastVLM-7B")

for model in "${models[@]}"; do
    echo "python test_history_local.py --model"$model" >> output.tsv"
    # python test_history_local.py --model $model --history 5 >> history.tsv
    python test_history_local.py --model $model --history 1 >> history.tsv
    python test_history_local.py --model $model --history 0 >> history.tsv
done