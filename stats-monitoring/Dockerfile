FROM python:3.8.5-slim AS builder

WORKDIR /root

# create requirements.txt from poetry
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt > requirements.txt

FROM python:3.8.5-slim

WORKDIR /root

# shift timezone to Asia/Tokyo.
RUN apt-get update \
    && apt-get install -y tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV TZ Asia/Tokyo

# japanese setting
RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen ja_JP.UTF-8 \
    && echo "export LANG=ja_JP.UTF-8" >> ~/.bashrc

# install python package from requirements.txt
COPY --from=builder /root/requirements.txt .
RUN pip install -r requirements.txt

# copy scripts and dependency file from source directory
COPY . .


CMD echo 'now ronning...'