#!/bin/bash

## Created on Jan 12, 2018

#SBATCH -o python_%A_%a.out # Standard output
#SBATCH -e python_%A_%a.err # Standard error
#SBATCH --array=1-268
#SBATCH --mem=4G

python main_slurm_for_BCgenes_random.py
