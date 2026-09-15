# DeepSeek V4.1 Flash raw-HF validation

MFQ can run the 476 GB DeepSeek V4.1 Flash Hugging Face Safetensors checkpoint
directly on a 512 GB Apple M3 Ultra without first converting it into an MFQ
container. The validated placement keeps the backbone and MoE experts resident
in unified memory and places only Engram storage on the Mac's internal SSD.

## Results

| Model / hardware | Placement | Model load | Prefill, 509 / 2,009 / 15,969 tokens | TG, MTP off | TG, high-acceptance MTP |
| --- | --- | ---: | ---: | ---: | ---: |
| DeepSeek V4.1 Flash raw-HF / Mac Studio M3 Ultra, 512 GB | Full resident; Engram only on internal SSD | **38.6 s** | **390.2 / 458.6 / 471.1 tok/s** | **18.29 tok/s** | **33.46 tok/s** (**+83.0%**) |

These are local, warmed, batch-size-1 measurements with a 32,768-token context
and temperature 0. Prefill uses deterministic repeated raw-completion prompts.
The process used approximately 287 GiB RSS after loading; the configured
440 GiB wired-memory budget is a ceiling, not its steady-state footprint.
Memory pressure and storage state can affect results.

## Runtime configuration

For this exact resident model geometry, MFQ selects a maximum prefill chunk of
5,440 tokens. Six routed experts per token keep the resulting 32,640 rows below
the MLX sorted-MXFP4 boundary of 32,768 rows. The Metal path also fuses the
window-KV RMSNorm, partial RoPE, and activation fake quantization. In a
component A/B, that work reduced `q_kv_prepare` by 1.18% and total evaluated
prefill time by 0.15%.

## MTP validation

The high-acceptance result uses a deterministic numeric continuation. It
accepted 132 of 132 drafted tokens and produced exactly the same output as
non-MTP decoding.

MTP speedup depends on the workload. On a code case with 50% acceptance, MTP
measured 17.52 tok/s versus 17.59 tok/s with MTP disabled. The adaptive policy
exited speculation after two low-acceptance cycles while preserving
deterministic output.

## Current work

- Broaden representative prose and code workloads where MTP provides a net
  gain before considering it for default enablement.
- Continue improving long-prompt prefill, token generation, model loading, and
  overlap of Engram cache misses with internal-SSD I/O on M3 Ultra.
