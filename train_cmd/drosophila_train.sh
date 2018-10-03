#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-v0.4.0-cuda8.0-venv
cd /users/rittscher/avelino/GitLab/noise2noise/scripts
python train_model.py --batch_size 16  --n_epochs 4000 --data_type 'drosophila_eggs'