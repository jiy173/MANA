# Memory-Augmented Non-Local Attention (MANA) for Video Super-Resolution


## Introduction
This repository contains the official implementation of "Memory-Augmented Non-Local Attention for Video Super-Resolution" in CVPR 2022. The code will be released soon.

## Updates
- 5/1/2022: uploaded data preparation code and training code

## Dataset
The Parkour dataset and the videos shown in the supplementary material can be downloaded at:
<https://drive.google.com/drive/folders/1KJdYAtlVRN79jYp4jiLWjvU8hJ50dfT0?usp=sharing>

The folder contains 2 zip files:
- Parkour_Dataset.zip holds the 14 Parkour videos used in our main paper.
- Supplementary_Dataset.zip holds the 11 real-world videos shown in the Fig.1 of the supplementary material.

##Training
###Training Data Preparation
We use [Vimeo90K](http://toflow.csail.mit.edu/) dataset to train our network. In this repository, you can find a script named [prepare_data.py](https://github.com/jiy173/MANA/blob/main/prepare_data.py) which organizes Vimeo90K into an [hdf5](https://www.hdfgroup.org/solutions/hdf5/) file used in the training.

To do this, simply run:<br>
`
python prepare_data.py --dataset vimeo90k/ --output vimeo90k_dataset.h5
`
<br>
where "dataset" is the Vimeo90K dataset path containing both "sequences/" and "seq_trainlist.txt"; "output" sets the output hdf5 file path.
