FROM python:3.8-slim AS builder

WORKDIR /workspace
ARG PLATFORM="prod"

RUN pip install poetry
COPY pyproject.toml ./
RUN poetry install --no-root
RUN if [ "$PLATFORM" = "dev" ]; \
    then poetry export --with dev -f requirements.txt > requirements.txt; \
    else poetry export -f requirements.txt > requirements.txt; \
    fi


FROM python:3.8-slim

WORKDIR /workspace
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH /workspace

RUN apt-get update && apt-get install -y libopencv-dev
COPY --from=builder /workspace/requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN apt-get autoremove -y

CMD ["/bin/bash"]
