# myrl
reinforcement learning altorithm implementations

# installation
- I tested this script on Ubuntu 16 (with NVIDIA Tesla K80) on the Google Cloud Platform.
- `pyenv` ([pyenv/pyenv-installer: This tool is used to install `pyenv` and friends.](https://github.com/pyenv/pyenv-installer#github-way-recommended "pyenv/pyenv-installer: This tool is used to install `pyenv` and friends."))
  is strongly recommended for building a python environment.
- atari-py requires `cmake`, `zlib`, etc. Install them first (`apt-get install make cmake zlib1g-dev g++`).
- for async-dqns, install Redis (see https://redis.io/download#installation).

## GPU support
see: [Installation Guide — CuPy 4.3.0 documentation](http://docs-cupy.chainer.org/en/stable/install.html "Installation Guide — CuPy 4.3.0 documentation")

- Install CUDA on your host.
  - [CUDA Toolkit 9.2 Download | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads "CUDA Toolkit 9.2 Download | NVIDIA Developer")
- If you use cupy-recommended environment (https://docs-cupy.chainer.org/en/stable/install.html#recommended-environments),
  cuDNN and NCCL libraries are included in `cupy` wheels.
  - `$ pip install cupy-cuda92`
