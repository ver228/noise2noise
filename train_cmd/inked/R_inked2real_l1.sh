#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-v0.4.0-cuda8.0-venv
cd /users/rittscher/avelino/GitLab/noise2noise/scripts


python train_model.py --data_type "inked2real" --model_name "unet-ch3" \
--loss_type 'l1' --n_epochs 5000 --num_workers 8 \
--init_model_path "R_inked2real_l1_20181030_135222_unet-ch3_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar"

#--init_model_path "inked2real_l1_20181029_213925_unet-ch3_adam_lr0.0001_wd0.0_batch16/checkpoint.pth.tar"