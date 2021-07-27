#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -A fwkt_v100
#SBATCH -t 23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH -o ./logs/hostname_%j.out
#SBATCH --gres=gpu:4
#SBATCH --mem=0

module load cuda/11.2
module load python
module load gcc/5.5.0
module load openmpi/3.1.2

source /home/zhdano82/hpmtraining/horoenv/bin/activate
cd /home/zhdano82/hpmtraining/ukd/NeuralSolvers/examples/2D_UKD_Heat

mpirun -np 4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python pennes_hpm_.py --num_t 2999 --name 05_hpm --epochs_pt 1 --epochs 100 --path_data /home/zhdano82/hpmtraining/smooth_data/data_0_05/ --use_horovod 1 --batch_size 512  --weight_j 0.01 --pretrained 1 --pretrained_name '05'
