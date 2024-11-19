#!/bin/bash
#SBATCH --job-name=prompting
#SBATCH --output=/home/asoubki/scratch/logs/%x.%j.out
#SBATCH --time=2-00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=

PYTHON=/home/asoubki/.miniconda3/envs/mmcg/bin/python
BIN=/home/asoubki/dev/calm-v2/bin/prompting.py
$PYTHON $BIN -p summarize-cogstate-only-author
