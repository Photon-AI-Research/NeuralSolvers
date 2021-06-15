#!/bin/bash -l

#SBATCH -p ml
#SBATCH -t 00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=29
#SBATCH --gres=gpu:6
#SBATCH -e error.out
#SBATCH -o output.out
#SBATCH -A p_da_aipp

module load modenv/ml
#module use /beegfs/global0/ws/s3248973-easybuild/easybuild-ml/modules/all/
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load h5py/2.10.0-fosscuda-2019b-Python-3.7.4

source ~/neural_solvers/bin/activate

cd /scratch/ws/1/s7520458-pinn_wave/NeuralSolvers/examples/3D_Wave_Equation
srun python training.py --path /scratch/ws/1/s7520458-pinn_wave/test/runs/006_laserOnly\
                        --iteration 2000\
                        --batch_size_n0 50000\
                        --batch_size_nb 7142\
                        --batch_size_nf 100000\
                        --num_experts 7\
                        --hidden_size 2000\
                        --num_hidden 8\
                        --num_epochs 2000\
                        --learning_rate 3e-5\
                        --normalize_labels 1\
                        --model finger\
                        --frequency 5\
                        --activation sin\
                        --k 2\
                        --shuffle 0\
                        --n0 134000000\
                        --nf 7000000000\
                        --nb 50000000\
                        --max_k 2100