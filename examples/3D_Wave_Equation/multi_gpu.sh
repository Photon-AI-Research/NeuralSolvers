#!/bin/bash -l
#SBATCH -t 24:00:00
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH -p fwkt_v100
#SBATCH -A fwkt_v100
#SBATCH --job-name waveEq
#SBATCH -o waveEq.out

module load cuda/11.2
module load gcc
module load openmpi/2.1.2-cuda112

source /home/stille15/ml_env/bin/activate
cd /home/stille15/NeuralSolvers/examples/3D_Wave_Equation


mpirun -np 8 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python training.py --path /bigdata/hplsim/production/AIPP/Data/LaserEvolution/runs/004_LaserOnly/simOutput/h5/simData_%T.h5 --iteration 2000 --batch_size 50000 --num_experts 7 --hidden_size 2000 --num_hidden 8 --num_epochs 2000  --learning_rate 3e-5 --normalize_labels 1 --model finger --frequency 5  --activation sin  --k 2 --shuffle 0 --n0 134000000