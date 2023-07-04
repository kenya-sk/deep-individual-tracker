FROM tensorflow/tensorflow:2.13.0rc0-gpu

WORKDIR /workspace
ENV PYTHONPATH /workspace

# add nvidia public key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN apt-get install -y \
    vim \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-dev

# install python package from requirements.txt
COPY ./requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# copy scripts and dependency file from source directory
COPY ./config ./config
COPY ./detector ./detector
COPY ./tests ./tests
COPY ./mypy.ini ./mypy.ini

RUN apt-get autoremove -y

CMD ["/bin/bash"]
