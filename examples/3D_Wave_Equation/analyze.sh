#!/bin/bash -l
#SBATCH -p ml
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1443
#SBATCH -e analyze_output.txt
#SBATCH -o analyze_error.txt
#SBATCH --reservation=p_da_aipp_292
#SBATCH -A p_da_aipp

echo "Load Modules"
module load modenv/ml
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load matplotlib/3.1.1-fosscuda-2019b-Python-3.7.4
module load h5py/2.10.0-fosscuda-2019b-Python-3.7.4
source ~/neural_solvers/bin/activate
cd /beegfs/global0/ws/s7520458-pinn_wave/NeuralSolvers/examples/3D_Wave_Equation
echo "SRUN"
srun python analyze_run.py --path /beegfs/global0/ws/s7520458-pinn_wave/laser_only/simOutput/openPMD/simData_%T.bp --name  experiment1
