# telesearch — local multimodal search over Telegram exports (GPU workloads).
#
# Base image must ship PyTorch >= 2.7 built against CUDA 12.8: that is the first
# combination with native sm_120 kernels, required by Blackwell GPUs such as the
# RTX PRO 6000 (older CUDA 12.4 wheels fail at runtime with "no kernel image is
# available for execution on the device"). It also satisfies the torch >= 2.6
# floor that recent `transformers` enforces before allowing `torch.load`
# (CVE-2025-32434), which otherwise aborts loading models that ship .bin weights.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

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
