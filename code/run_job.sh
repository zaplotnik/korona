#!/bin/bash
#SBATCH --nodelist=node13
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1 
#SBATCH --time=01:00:00
#SBATCH --mem=2G
#SBATCH --partition=rude
#SBATCH --qos=rude
#SBATCH --error=modes_inversion.%J.err
#SBATCH --output=modes_inversion.%J.out
#SBATCH --job-name=modes_inversion
#SBATCH --workdir=/shared/data-camelot/zaplotnikz/korona/run

source /home/zaplotnikz/forPythonMeteoBasic.sh
source activate meteoziga

cd /shared/data-camelot/zaplotnikz/korona
python run_korona.py 669 672