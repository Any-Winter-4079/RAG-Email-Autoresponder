FROM anywinter4079/pytorch:2.10.0-cu128

WORKDIR /workspace

RUN apt-get update && apt-get install -y git nano && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/FlagOpen/FlagEmbedding.git /workspace/FlagEmbedding \
 && cd /workspace/FlagEmbedding \
 && git checkout dbc600560b2dadcc1514989092f7b849673bb67d \
 && pip install -e . \
 && pip install "transformers==4.44.2" "peft==0.17.1" \
 && mkdir -p /workspace/finetune
