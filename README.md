# myrl
reinforcement learning altorithm implementations

# installation
atari-py requires `cmake, zlib etc.`. Install them first (`apt-get install make cmake zlib1g-dev g++`)

## GPU support
see: [Installation Guide — CuPy 5.0.0b2 documentation](http://docs-cupy.chainer.org/en/stable/install.html "Installation Guide — CuPy 5.0.0b2 documentation")

- Install CUDA on your host.
  - [CUDA Toolkit 9.2 Download | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads "CUDA Toolkit 9.2 Download | NVIDIA Developer")
- If you use cupy-recommended environment, cuDNN and NCCL libraries are included in `cupy` wheels.
  - `$ pip install cupy-cuda92`
