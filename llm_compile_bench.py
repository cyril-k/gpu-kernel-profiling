"""
run PyTorch container with:
```
docker run --gpus all --ipc=host -v ./:/workspace -v ~/.cache/huggingface:/root/.cache/huggingface -it --rm pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
```

Install `transformers` with:
```
pip install transformers==4.55.4
```

Run compiled:
```
python llm_compile_bench.py --kv-cache --cache-impl static --compile
```

Run vanilla:
```
python llm_compile_bench.py --kv-cache --cache-impl dynamic
"""

import os
import tempfile
import argparse
import torch
from torch.profiler import profile, record_function, schedule, ProfilerActivity
from transformers import AutoTokenizer, AutoModelForCausalLM

torch._dynamo.reset()
torch.cuda.empty_cache()
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor_")
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_")

def add_span_generate(model):

    tag = "gen::forward"
    orig_forward = model.forward
    compiled_call = hasattr(model, "_compiled_call")

    def wrapped_forward(*args, **kwargs):

        with record_function(tag):
            out = orig_forward(*args, **kwargs)

        return out

    if compiled_call:
        orig_compiled_call = model._compiled_call
        def wrapped_compiled(*args, **kwargs):

            with record_function(tag):
                out = orig_compiled_call(*args, **kwargs)

            return out
        
        model._compiled_call = wrapped_compiled

    model.forward = wrapped_forward
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Profile attention implementations on a single layer.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--kv-cache", action="store_false")
    parser.add_argument("--cache-impl", type=str, default="static")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--filename", type=str, default="generation_trace")
    parser.add_argument("--compile", action="store_true", help="Compile model.forward() with torch.compile().")

    args = parser.parse_args()
    print(args)

    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(args.device)

    model.eval()
    
    with torch.no_grad():

        #prepare synthetic inputs
        inputs_size = (1, 10)
        inputs = {'input_ids': torch.randint(model.config.vocab_size, inputs_size).to(args.device),}
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        model.generation_config.cache_implementation = args.cache_impl
        model.generation_config.use_cache = args.kv_cache
        model.generation_config.max_new_tokens = args.max_new_tokens

        if args.compile:
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
        
        #warmup    
        _ = model.generate(**inputs)
        model = add_span_generate(model)


        trace_file = f"{args.filename}_{'compiled' if args.compile else 'vanilla'}.json"

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
            schedule=schedule(wait=0, warmup=2, active=1, repeat=1),
            record_shapes=True,
        ) as prof:
            for s in range(3):
                _ = model.generate(**inputs)
                prof.step()
        
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
        if trace_file:
            prof.export_chrome_trace(trace_file)
            print(f"exported profile to '{trace_file}'")

if __name__ == "__main__":
    main()