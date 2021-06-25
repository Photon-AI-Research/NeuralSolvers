#!/bin/bash -l
#SBATCH -p ml
#SBATCH -t 5:00:00
#SBATCH --nodes=30
#SBATCH --ntasks=180
#SBATCH --cpus-per-task=29
#SBATCH --gres=gpu:6
#SBATCH --mem-per-cpu=1443
#SBATCH -e error_files/experiment_gpinn.txt
#SBATCH -o outpu_files/experiment_gpinn.txt
#SBATCH --reservation=p_da_aipp_292
#SBATCH -A p_da_aipp

echo "RUN ML"
module load modenv/ml
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load h5py/2.10.0-fosscuda-2019b-Python-3.7.4
source ~/neural_solvers/bin/activate
cd /beegfs/global0/ws/s7520458-pinn_wave/NeuralSolvers/examples/3D_Wave_Equation
srun python training.py --path /beegfs/global0/ws/s7520458-pinn_wave/laser_only/simOutput/openPMD/simData_%T.bp\
                        --iteration 2000\
                        --name experiment_gpinn\
                        --batch_size_n0 5000\
                        --batch_size_nb 5000\
                        --batch_size_nf 15000\
                        --num_experts 10\
                        --hidden_size 90\
                        --num_hidden 8\
                        --num_epochs 80\
                        --learning_rate 3e-5\
                        --normalize_labels 0\
                        --model gpinn\
                        --frequency 5\
                        --activation sin\
                        --k 1\
                        --shuffle 0\
                        --n0 100000000\
                        --nf 800000000\
                        --nb 50000000\
                        --max_t 2100\
                        --boundary 0\
                        --restart 1\
                        --checkpoint checkpoints/experiment_gpinn
