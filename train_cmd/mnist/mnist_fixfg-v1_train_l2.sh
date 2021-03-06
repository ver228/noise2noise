#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd /users/rittscher/avelino/GitLab/noise2noise/scripts


python train_model.py --batch_size 48 --data_type 'mnist-fg-fix-v1' --loss_type 'l2' --lr 1e-5 --num_workers 4