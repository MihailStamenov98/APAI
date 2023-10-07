#!/bin/bash
#SBATCH --job-name=grad32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mihail.stamenov@studio.unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=experiment_results_grad32
#SBATCH --gres=gpu:1

nvcc project_sorting.cu -o grad32

arguments=(20 50 100 250 500 1000 10000 100000 1000000 10000000 100000000 1000000000)
echo "local mem solution, normal solution/n"

for arg in "${arguments[@]}"; do
   # Run the executable and capture the output
    result=$(./grad32 "$arg")
    echo $result
    
done