FROM nvcr.io/nvidia/pytorch:25.06-py3

SHELL ["/bin/bash", "-lc"]

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace

RUN git clone https://github.com/Dao-AILab/flash-attention.git flash-attention && \
    git clone https://github.com/cyril-k/gpu-kernel-profiling.git /workspace/gpu-kernel-profiling

RUN uv venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN uv pip install nvidia-cutlass-dsl==4.3.0 \
 && uv pip install -e /workspace/flash-attention/flash_attn/cute \
 && uv pip install transformers==4.55.4 \
 && uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

WORKDIR /workspace/gpu-kernel-profiling

CMD ["bash"]
