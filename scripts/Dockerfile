FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && apt-get install -y \
    cmake \
    zlib1g-dev \
    && apt-get -y clean all \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip

RUN pip3 install atari-py gym[atari] scikit-image imageio toml keras ipdb

CMD ["/bin/bash"]
