#!/bin/bash

#$-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-v0.4.0-cuda8.0-venv
cd /users/rittscher/avelino/GitLab/noise2noise/scripts


SRC_DIR=/users/rittscher/avelino/workspace/denoising_data/inked_slides/
DST_DIR=$TMPDIR/ramdisk/inked_slides
rsync -avz $SRC_DIR $DST_DIR

python train_model.py --data_type "inked_slides" --model_name "unet-ch4" --data_src_dir $DST_DIR