FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip gpg tar unzip curl && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
        torch==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 \
        transformers==4.48.0 sentencepiece==0.2.0 \
        fastapi==0.110.0 uvicorn[standard]==0.29.0 psutil==5.9.8

WORKDIR /root
COPY run.sh  /root/run.sh
COPY container_main.py /root/main.py
COPY tokenizer/ /root/tokenizer
COPY test.tar.gpg /root/test.tar.gpg
RUN chmod +x /root/run.sh
RUN mkdir -p /output && chmod 777 /output
RUN mkdir -p /workspace && chmod 777 /workspace

ENTRYPOINT ["/root/run.sh"]
