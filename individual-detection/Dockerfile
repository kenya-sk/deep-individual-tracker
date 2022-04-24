FROM tensorflow/tensorflow:1.15.5-gpu-py3

WORKDIR /root
ADD requirements.txt requirements.txt

RUN apt-get update
RUN apt-get install -y \
    vim \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ADD . .

CMD echo "now running..."
