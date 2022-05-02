FROM python:3.8-slim as builder

WORKDIR /workspace
RUN pip install poetry
COPY pyproject.toml ./
RUN poetry install
RUN poetry export -f requirements.txt > requirements.txt

FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

RUN apt-get update && apt-get install -y libopencv-dev
COPY --from=builder /workspace/requirements.txt .
RUN pip install -r requirements.txt

COPY . .
#CMD python src/run_annotator.py
CMD ["echo", "container running..."]