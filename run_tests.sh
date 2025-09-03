#!/bin/bash

pick_and_concat() {
  local sep="${1:-}"; shift
  local arr=("$@")
  (( ${#arr[@]} == 0 )) && return 0

  local k=1
  (( ${#arr[@]} >= 2 )) && k=$(( (RANDOM % 2) + 1 ))

  local chosen=()
  if command -v shuf >/dev/null 2>&1; then
    mapfile -t chosen < <(printf '%s\n' "${arr[@]}" | shuf -n "$k")
  else
    if (( k == 1 )); then
      chosen=("${arr[RANDOM % ${#arr[@]}]}")
    else
      local i=$((RANDOM % ${#arr[@]}))
      local j
      while :; do
        j=$((RANDOM % ${#arr[@]}))
        (( j != i )) && break
      done
      chosen=("${arr[i]}" "${arr[j]}")
    fi
  fi

  local IFS="$sep"
  printf '%s\n' "${chosen[*]}"
}

conda activate phi4

models=("google/gemma-3-4b-it" "Qwen/Qwen2.5-VL-3B-Instruct" "microsoft/Phi-4-multimodal-instruct")

shopt -s nullglob nocaseglob
images=( "tests"/*.{jpg,jpeg} )
shopt -u nocaseglob

for model in "${models[@]}"; do
    concatenated="$(pick_and_concat ' ' "${images[@]}")"
    echo "python test_local.py "$model" "$concatenated" 2>/dev/null >> output.tsv"
    python test_local.py $model $concatenated >> output.tsv
done

conda activate phi3

models=("microsoft/Phi-3.5-vision-instruct" "microsoft/Phi-3.5-vision-instruct" "microsoft/Phi-3.5-vision-instruct")

shopt -s nullglob nocaseglob
images=( "tests"/*.{jpg,jpeg} )
shopt -u nocaseglob

for model in "${models[@]}"; do
    concatenated="$(pick_and_concat ' ' "${images[@]}")"
    echo "python test_local.py "$model" "$concatenated" 2>/dev/null >> output.tsv"
    python test_local.py $model $concatenated >> output.tsv
done

conda activate appl

models=("apple/FastVLM-0.5B" "apple/FastVLM-1.5B" "apple/FastVLM-7B")

shopt -s nullglob nocaseglob
images=( "tests"/*.{jpg,jpeg} )
shopt -u nocaseglob
sadasd
for model in "${models[@]}"; do
    concatenated="$(pick_and_concat ' ' "${images[@]}")"
    echo "python test_local.py "$model" "$concatenated" 2>/dev/null >> output.tsv"
    python test_local.py $model $concatenated >> output.tsv
done