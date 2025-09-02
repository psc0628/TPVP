# Nonrigid-ICP-Pytorch

This repository is a modifed version of [Nonrigid-ICP-Pytorch](https://github.com/rabbityl/Nonrigid-ICP-Pytorch) with the replacement of Mayavi visualization with Open3D and input with point clouds.

## Environment Setup

Tested on: Ubuntu 22.04 + CUDA 11.8 + PyTorch 2.0

### install cuda11.8
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
chmod +x cuda_11.8.0_520.61.05_linux.run
sudo ./cuda_11.8.0_520.61.05_linux.run --toolkit --samples --override
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### install conda python3.9 env
```
conda create -n nricp python=3.9
conda activate nricp
```

### install torch
```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### install pytroch3d
```
conda install -c iopath iopath
conda install -c fvcore -c conda-forge fvcore
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
```

### install lietorch
```
git clone --recursive https://github.com/princeton-vl/lietorch.git
cd lietorch
export TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9;9.0"
pip install --no-build-isolation .
```

### install Nonrigid-ICP-Pytorch
```
cd TPVP/Nonrigid_ICP
cd cxx
pip install pybind11
python setup.py install
```

### install open3d and others
```
pip install plyfile scikit-image easydict scipy pyyaml
pip install "numpy<2.0"
pip install opencv-python==4.10.0.82
pip install open3d==0.18.0
```

