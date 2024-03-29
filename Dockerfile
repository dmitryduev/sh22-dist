FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

# todo: rm once nvidia updates their docker images
RUN apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 3
RUN apt-key adv --fetch-keys \
    https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

ARG DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.8
ARG GIT_BRANCH=main

ARG WORKING_DIRECTORY=/wandb/sh22-dist

# todo: enable docker image layer caching on circleci
RUN apt-get update && apt-get install -y --no-install-recommends --ignore-missing \
    vim \
    curl \
    ca-certificates \
    sudo \
    git \
    python${PYTHON_VERSION} \
    python3-pip \
    python${PYTHON_VERSION}-dev \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libcudnn8 \
    net-tools \
    htop \
    wget \
    unzip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

RUN mkdir /wandb
WORKDIR /wandb

RUN adduser --disabled-password --gecos '' --shell /bin/bash sdk \
    && adduser sdk sudo\
    && chown -R sdk:sdk /wandb
RUN echo "sdk ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-sdk
USER sdk
ENV HOME=/home/sdk
RUN chmod 777 /home/sdk

# clone wandb/client repository and install wandb sdk
# use torch wheels with CUDA 11.3 support
RUN git clone https://github.com/dmitryduev/sh22-dist.git ${WORKING_DIRECTORY} \
    && cd ${WORKING_DIRECTORY} \
    && git checkout ${GIT_BRANCH} \
    && pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu113 \
           -r requirements.txt --no-cache-dir
#    && pip install tox==${TOX_VERSION} --no-cache-dir

RUN PATH=/home/sdk/.local/bin:$PATH

WORKDIR ${WORKING_DIRECTORY}
#CMD ["tail", "-f", "/dev/null"]
CMD ["python", "train.py"]
