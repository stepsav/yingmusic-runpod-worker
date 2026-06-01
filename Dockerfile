# Serverless-воркер YingMusic-SVC — zero-shot пение голосом юзера (SOTA, дек 2025).
# Research-код: 2-стадийный пайплайн, веса с HuggingFace. Возможны итерации отладки.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
# rebuild trigger 1
# Неинтерактивный apt: иначе tzdata спрашивает регион и сборка виsnет
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Системные зависимости (репо требует sox + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg sox libsox-fmt-all tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/GiantAILab/YingMusic-SVC.git /app/YingMusic-SVC

WORKDIR /app/YingMusic-SVC
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod huggingface_hub

# Веса модели (YingMusic-SVC-full + bs_roformer для отделения аккомпанемента)
RUN python -c "from huggingface_hub import snapshot_download as d; d('GiantAILab/YingMusic-SVC', local_dir='/app/YingMusic-SVC/pretrained'); print('CKPT DOWNLOADED')"

RUN python -c "import runpod; print('RUNPOD OK', runpod.__version__)"

COPY handler.py /app/YingMusic-SVC/handler.py
ENTRYPOINT []
CMD ["sh", "-c", "python -u /app/YingMusic-SVC/handler.py 2>&1"]
# rebuild 1
