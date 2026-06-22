# telesearch — local multimodal search over Telegram exports (GPU workloads).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY telesearch ./telesearch

RUN pip install --no-cache-dir -e ".[all]"

ENV TELESEARCH_DEVICE=cuda \
    TELESEARCH_WHISPER_COMPUTE=float16 \
  PYTHONUNBUFFERED=1

ENTRYPOINT ["telesearch"]
CMD ["info"]
