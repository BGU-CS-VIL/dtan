# Diffeomorphic Temporal Alignment Nets
## We moved to PyTorch! 
TensorFlow implementation (old master branch) is available at tf_legacy branch.

Repository for our <b>NeurIPS 2019</b> paper, [Diffeomorphic Temporal Alignment Nets](https://www.cs.bgu.ac.il/~orenfr/DTAN/ShapiraWeber_NeurIPS_2019.pdf) co-authored by: Ron Shapira Weber, Matan Eyal, Nicki Skafte Detlefsen, Oren Shriki and Oren Freifeld.
<img src="/figures/dtan_intro_fig.png" alt="DTAN joint alignmnet of ECGFiveDays dataset.">
## Model Architecture
<img src="/figures/DTAN_detailed_model.png" alt="DTAN Architecture.">

## Author of this software
Ron Shapira Weber (email: ronsha@post.bgu.ac.il)

## Requirements
- Standard Python(>=3.6) packages: numpy, matplotlib, tqdm, seaborn
- PyTorch >= 1.4
- tslean == 0.5.2
- libcpab == 2.0
- For Nvidia GPU iimplementation: CUDA==11.0 + appropriate cuDNN as well. You can follow the instructions [here](https://pytorch.org/get-started/locally/).

## Operation system: 
For the native PyTorch implementation (slower), we support all operating systems. 
For the fast CUDA implementation of libcpab, we only support Linux.

## Installation
We recommend installing a [virtual environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) via Anaconda.
For instance:
```
conda create -n dtan python=3.7 numpy matplotlib seaborn tqdm
```
### libcpab
licpab [2] is a python package supporting the CPAB transformations [1] in Numpy, Tensorflow and Pytorch.
For your convince, we have added a lightweight version of libcpab at DTAN/libcpab. 

That being said, you are still encouraged to install the full package. 

Install [libcpab](https://github.com/SkafteNicki/libcpab) <br>
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
1. To initialize the model:

```python
from DTAN.DTAN_layer import DTAN as dtan_model

# Init model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dtan_model(signal_len=int, channels=int, tess=[int,], n_recurrence=int,
                    zero_boundary=bool, device='gpu').to(device)
```


Under the 'examples' dir you can find example scripts for training and running DTAN time-series joint alignment. 

2. **UCR time-series classification archive [3] alignment example.** <br>
To run, simply enter (from the examples dir):
```
python UCR_alignment.py
```
We support loading data via [tslearn](https://tslearn.readthedocs.io/en/stable/index.html) [4]. 

For more information regarding the UCR archive, please visit:
[https://www.cs.ucr.edu/~eamonn/time_series_data/](https://www.cs.ucr.edu/~eamonn/time_series_data/)

The script supports the following flags:
```
optional arguments:
  -h, --help            show this help message and exit
  --dataset UCR dataset name
                        string. Dataset name to load from tslearn.datasets.UCR_UEA_datasets
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

2. **Usage Example - Running with and without smoothness prior**:<br>
See the [jupyter notebook](https://github.com/BGU-CS-VIL/dtan/blob/pytorch/notebooks/Usage%20Example%20-%20Running%20with%20and%20without%20prior.ipynb), illustrating the importance of the smoothness prior. 

3. **UCR Nearest Centroid Classification (NCC)**:
*Coming soon to PyTorch version*<br>
Here we provide an end-to-end pipeline for the NCC experiment described in our paper.
The script uses the same hyper-parameters (i.e., lambda_var, lambda_smooth, n_recurrences) used in the paper, depending on the UCR dataset. 
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
[4] @article{JMLR:v21:20-091,
  author  = {Romain Tavenard and Johann Faouzi and Gilles Vandewiele and
             Felix Divo and Guillaume Androz and Chester Holtz and
             Marie Payne and Roman Yurchak and Marc Ru{\ss}wurm and
             Kushal Kolar and Eli Woods},
  title   = {Tslearn, A Machine Learning Toolkit for Time Series Data},
  journal = {Journal of Machine Learning Research},
  year    = {2020},
  volume  = {21},
  number  = {118},
  pages   = {1-6},
  url     = {http://jmlr.org/papers/v21/20-091.html}
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
