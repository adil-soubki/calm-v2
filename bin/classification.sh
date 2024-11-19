#!/bin/bash

while getopts ":c:t:" opt; do
  case "$opt" in
    c) config="$OPTARG" ;;
    t) task="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

shift $((OPTIND-1))

# Check if options are set, otherwise use defaults.
if [ -z "$config" ]; then
    config=("flan-t5-base" "calm-flan-t5-base");
fi
if [ -z $task ]; then
    task=(
        "boolq" "commitment_bank" "commitment_bank_text_only"
        "fact_bank" "fantom_bin" "fantom_mc" "goemotions" "goemotions_ekman"
        "iemocap" "imdb" "wsc" "wic"
    );
fi

# Iterate over arrays.
for t in "${task[@]}"; do
    for c in "${config[@]}"; do
        echo "Task: $t -- Config: $c"
        PYTHON=/home/asoubki/.miniconda3/envs/mmcg/bin/python
        BIN=/home/asoubki/dev/calm-v2/bin/classification.py
        CONFIG=/home/asoubki/dev/calm-v2/configs/classification/$c.json

        sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=classification
#SBATCH --output=/home/asoubki/scratch/logs/%x.%j.out
#SBATCH --time=7-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=

$PYTHON $BIN $CONFIG -t $t
EOT
    done
done

