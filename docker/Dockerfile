FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    net-tools \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=''", "--ServerApp.password=''"]
