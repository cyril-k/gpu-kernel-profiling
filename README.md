# GPU kernel profiling

## [Optional] comparing eager and compiled execution of an LLM:

```
git clone https://github.com/cyril-k/gpu-kernel-profiling.git
docker run --gpus all --ipc=host -v ./gpu-kernel-profiling:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface -it --rm pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel bash
pip install transformers==4.55.4
```
To execute model (Qwen/Qwen1.5-7B) forward pass without compilation, run the following:
```
python llm_compile_bench.py --kv-cache --cache-impl dynamic
```
The script will produce an output trace `generation_trace_vanilla.json` which can be opened with 
Perfetto UI.

To get a trace from compiled `model.forward()`:
```
python llm_compile_bench.py --kv-cache --cache-impl static --compile
```

## Tracing SDPA kernel execution:

```
git clone https://github.com/cyril-k/gpu-kernel-profiling.git
docker run --gpus all --ipc=host -v ./gpu-kernel-profiling:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface -it --rm nvcr.io/nvidia/pytorch:25.06-py3 bash
pip install transformers==4.55.4
```
Note that we use NVIDIA’s PyTorch container (different from the container we used in the previous step).

Executing the code:
```
python bench.py --modes fa2 sdpa_flash sdpa_cudnn sdpa_mem flex --seq 1024 2048 4096 --compile --backward
```
use `--modes` to select attention backend and `--seq` to pick sequence length (accepts multiple values).

The script will produce a number of traces which we can open with Perfetto.
