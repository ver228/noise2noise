#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-v0.4.0-cuda8.0-venv
cd /users/rittscher/avelino/GitLab/noise2noise/scripts


SRC_DIR=/users/rittscher/avelino/workspace/denoising_data/drosophila_eggs/train/
DST_DIR=$TMPDIR/ramdisk/drosophila_eggs
rsync -avz $SRC_DIR $DST_DIR
python train_model.py --batch_size 16  --n_epochs 4000 --data_type 'drosophila-eggs' --data_src_dir $DST_DIR