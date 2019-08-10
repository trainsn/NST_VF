#PBS -N nst_vf_vfl_net_train 
#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -j oe

source /users/PAS0027/trainsn/.bashrc
source activate pytorch
cd /users/PAS0027/trainsn/nst_vf/vfl_net
python train.py




