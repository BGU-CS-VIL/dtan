# Diffeomorphic Temporal Alignment Nets - PyTorch (BETA!)
Repository for our <b>NeurIPS 2019</b> paper, [Diffeomorphic Temporal Alignment Nets](https://www.cs.bgu.ac.il/~orenfr/DTAN/ShapiraWeber_NeurIPS_2019.pdf) co-authored by: Ron Shapira Weber, Matan Eyal, Nicki Skafte Detlefsen, Oren Shriki and Oren Freifeld.
<img src="/figures/dtan_intro_fig.png" alt="DTAN joint alignmnet of ECGFiveDays dataset.">
## Model Architecture
<img src="/figures/DTAN_detailed_model.png" alt="DTAN Architecture.">

## Author of this software
Ron Shapira Weber (email: ronsha@post.bgu.ac.il)

## Requirements
- Standard Python(3.6) packages: numpy, matplotlib, scipy
- PyTorch==1.4
- tslean==0.1.19 (requires for DTW based methods - SoftDTW, DBA and NCC)
- libcpab==2.0
- For Nvidia GPU iimplementation: CUDA==11.0 + appropriate cuDNN as well. You can follow the instructions [here](https://pytorch.org/get-started/locally/).

Operation system: we currenly only support Linux oprating system.

## Installation
We recommend installing a [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) via Anaconda.
For instance:
```
conda create -n dtan python=3.6 numpy matplotlib scipy
```
### libcpab
licpab [2] is a python package supporting the CPAB transformations [1] in Numpy, Tensorflow and Pytorch.

Install [libcpab]:(https://github.com/SkafteNicki/libcpab) <br>
Note 1: you might have to recompile the dynamic libraries under /libcpab/tensorflow/ <br>
```
git clone https://github.com/SkafteNicki/libcpab
```
Add libcpab to your python path:
```
export PYTHONPATH=$PYTHONPATH:$YOUR_FOLDER_PATH/libcpab
```
Make sure libcpab was installed properly (Run one of the demos).

### DTAN
Clone the repository:
```
git clone https://github.com/BGU-CS-VIL/dtan.git
# move to pytorch branch
git checkout pytorch
```
Add DTAN to your python path:
```
export PYTHONPATH=$PYTHONPATH:$YOUR_FOLDER_PATH/dtan
```
Try the example code under dtan/exmaples (see also our Usage section below):
```
python UCR_alignment.py
```
## Usage
### Examples
Under the 'examples' dir you can find example scripts for running DTAN time-series joint alignment. 
1. **UCR time-series classification archive [3] alignment example.**
To run simply enter:
```
python UCR_alignment.py
```
Note that here we only provide one dataset from the UCR archive - ECGFiveDays. 
For the entire archive, please visit:
[https://www.cs.ucr.edu/~eamonn/time_series_data/](https://www.cs.ucr.edu/~eamonn/time_series_data/)

The script supports the following flags:
```
optional arguments:
  -h, --help            show this help message and exit
  --tess_size TESS_SIZE
                        CPA velocity field partition
  --smoothness_prior    smoothness prior flag
  --no_smoothness_prior
                        no smoothness prior flag
  --lambda_smooth LAMBDA_SMOOTH
                        lambda_smooth, larger values -> smoother warps
  --lambda_var LAMBDA_VAR
                        lambda_var, larger values -> larger warps
  --n_recurrences N_RECURRENCES
                        number of recurrences of R-DTAN
  --zero_boundary ZERO_BOUNDARY
                        zero boundary constrain
  --n_epochs N_EPOCHS   number of epochs
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               learning rate

```

2. **UCR Nearest Centroid Classification (NCC)**:
*Coming soon to PyTorch version*<br>
Here we provide an end-to-end pipeline for the NCC experiment described in our paper.
The script uses the same hyper-paramters (i.e., lambda_var, lambda_smooth, n_recurrences) used in the paper, depending on the UCR dataset. 
To run the pipeline on the 'ECGFiveDays' dataset go to the 'examples' dir and simply run: 

```
python UCR_NCC.py
```
You can change the dataset inside the script.
## References
```
[1] @article{freifeld2017transformations,
  title={Transformations Based on Continuous Piecewise-Affine Velocity Fields},
  author={Freifeld, Oren and Hauberg, Soren and Batmanghelich, Kayhan and Fisher, John W},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}

[2] @misc{detlefsen2018,
  author = {Detlefsen, Nicki S.},
  title = {libcpab},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SkafteNicki/libcpab}},
}
[3] @misc{UCRArchive,
title={The UCR Time Series Classification Archive},
author={ Chen, Yanping and Keogh, Eamonn and Hu, Bing and Begum, Nurjahan and Bagnall, Anthony and Mueen, Abdullah and Batista, Gustavo},
year={2015},
month={July},
note = {\url{www.cs.ucr.edu/~eamonn/time_series_data/}}
}
```
## Versions:


## License
This software is released under the MIT License (included with the software). Note, however, that if you are using this code (and/or the results of running it) to support any form of publication (e.g., a book, a journal paper, a conference paper, a patent application, etc.) then we request you will cite our paper:
```
@inproceedings{weber2019diffeomorphic,
  title={Diffeomorphic Temporal Alignment Nets},
  author={Weber, Ron A Shapira and Eyal, Matan and Skafte, Nicki and Shriki, Oren and Freifeld, Oren},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6570--6581},
  year={2019}
}

```
