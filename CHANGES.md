# Changes

## CUDA 直接加载 Hugging Face Safetensors

- `mfq-decode --model` 现在既接受 MFQ 文件，也接受 Hugging Face Safetensors 模型目录。
- HF 张量通过后端无关的 `HfModelSource` 转换为与 MFQ 相同的 canonical 运行时视图，CUDA 加载器无需按存储格式分支。
- HF 目录暂不支持 MFQ tensor/expert overlay；在提供稳定的格式无关模型指纹前，暂不启用磁盘 prefix cache。
- CUDA 直载沿用 CUDA 已有张量能力；HF `F8_E4M3` 权重会按其 scale contract 直接暴露为 `MXFP8` 或无损 `FP8-128SQ/SQ8`。

## Hugging Face tokenizer 接入 CUDA server

- 启动 HF 模型时，Python 服务层从 `tokenizer.json` 等 sidecar 生成并缓存 tokenizer GGUF。
- CUDA 和 Metal runtime 统一通过 `--tokenizer <tokenizer.gguf>` 使用该文件；MFQ 文件仍优先使用自身内嵌的 tokenizer。
- 此改动只接通 tokenizer 文件的生成和传参，没有修改 tokenizer 算法。
