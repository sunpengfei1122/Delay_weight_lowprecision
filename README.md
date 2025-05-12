# README #
## Requirements
Python 3 with the following packages installed:

* PyTorch 
* numpy
* matplotlib
* pyyaml
* h5py

A **CUDA** enabled **GPU** is required for training any model.
No plans on CPU only implementation yet.
The software has been tested with CUDA libraries version 11.3 and Pytorch 1.12.1 （pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
            torchaudio==0.12.1  -f https://download.pytorch.org/whl/cu113/torch_stable.html）

## Installation
The repository includes C++ and CUDA code that has to be compiled and installed before it can be used from Python, download the repository and run the following command to do so:

`python setup.py install`

Then 
`pip install torch==1.12.1+cu113 \
            torchvision==0.13.1+cu113 \
            torchaudio==0.12.1 \
            -f https://download.pytorch.org/whl/cu113/torch_stable.html`
）

To test the installation:

`cd test`

`python -m unittest`

## Examples
Example implementations can be found inside Examples folder.

* Run example SHD implementation, with Learnable tenary weights, quantized delays, L2 penalty. You could get >90% performance on SHD datasets.
       `>>> cd examples/SHD`
	`>>> python train.py`

`
	
