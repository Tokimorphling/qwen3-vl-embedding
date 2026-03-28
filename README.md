# Qwen3-VL-Embedding on llama.cpp

这个仓库用于把 `Qwen3-VL-Embedding-2B` 转成 `llama.cpp` 可用的 GGUF，并用本地修改版 `llama.cpp` 复现官方 Python / Hugging Face 的 embedding 行为。

当前结论：

- 对已经验证过的 `text`、`image`、`image + text` 输入，当前本地改动版 `llama.cpp` 比 clean upstream 更接近官方行为。
- 推荐部署配置是 `F16 + mmproj-f16`。
- 已有固定回归脚本，可直接对比 Python 参考实现和 `llama-vl-embedding`。

仓库本身只跟踪代码、脚本和文档；模型权重、GGUF、日志和本地实验文件不纳入版本库。

## 0. Clone

本仓库使用 submodule 记录依赖仓库：

- `llama.cpp`
- `Qwen3-VL-Embedding`

首次拉取建议直接用：

```bash
git clone --recursive <this-repo-url>
```

如果已经 clone 过，再执行：

```bash
git submodule update --init --recursive
```

## 1. 目录约定

下面的相对路径都以仓库根目录为准：

- `./llama.cpp`: 本地修改版 `llama.cpp` submodule
- `./Qwen3-VL-Embedding`: 官方 Python 参考实现 submodule
- `./models/Qwen3-VL-Embedding-2B`: 本地 Hugging Face 模型目录
- `./scripts/check_qwen3_vl_embedding_regression.py`: Python vs `llama-vl-embedding` 回归脚本
- `./scripts/data/qwen3_vl_embedding_regression_inputs.json`: 固定回归样例
- `./scripts/convert_and_regress_qwen3_vl_embedding.sh`: 一键构建、转换、回归入口

## 2. 快速开始

先进入你已经准备好的 Python 环境：

```bash
conda activate base
cd /path/to/qwen3vl-embedding
```

安装一次转换依赖：

```bash
python -m pip install -r ./llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

如果你机器上有 CUDA，并希望直接跑 GPU 回归：

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-2B \
  --python-device cuda \
  --cuda-visible-devices 0
```

如果你只想走 CPU：

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh \
  --model-dir ./models/Qwen3-VL-Embedding-2B \
  --python-device cpu \
  --llama-ngl 0
```

这个脚本默认会：

1. 编译 `llama-vl-embedding`
2. 生成 `Qwen3-VL-Embedding-2B-f16.gguf`
3. 生成 `mmproj-Qwen3-VL-Embedding-2B-f16.gguf`
4. 用固定样例跑 Python vs `llama-vl-embedding` 回归

如果输出文件已经存在，它会跳过重复转换；如果你要强制重转，传 `--force-convert`。

## 3. 一键脚本参数

```bash
./scripts/convert_and_regress_qwen3_vl_embedding.sh --help
```

常用参数：

- `--model-dir PATH`: HF 模型目录
- `--outtype f16|f32`: GGUF 输出类型，默认 `f16`
- `--python-device auto|cpu|cuda`: Python 参考实现设备选择
- `--cuda-visible-devices VALUE`: 显式指定 GPU
- `--llama-ngl VALUE`: 传给 `llama-vl-embedding` 的 `-ngl`
- `--llama-no-mmproj-offload`: 禁用 `mmproj` GPU offload
- `--force-convert`: 即使目标 GGUF 已存在也重新转换
- `--rebuild`: 强制重新构建 `llama-vl-embedding`
- `--skip-build`: 跳过编译
- `--skip-regression`: 只做转换，不跑回归
- `--install-convert-deps`: 脚本内部执行一次 `pip install -r ...`

## 4. 手工转换

如果你不想走一键脚本，也可以手工执行。

### 4.1 转主模型

```bash
cd ./llama.cpp

python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../Qwen3-VL-Embedding-2B-f16.gguf \
  --outtype f16
```

### 4.2 转 mmproj

```bash
python ./convert_hf_to_gguf.py \
  ../models/Qwen3-VL-Embedding-2B \
  --outfile ../mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --outtype f16 \
  --mmproj
```

### 4.3 编译

```bash
cmake -S ./llama.cpp -B ./llama.cpp/build
cmake --build ./llama.cpp/build --target llama-vl-embedding -j
```

### 4.4 运行

纯文本：

```bash
./llama.cpp/build/bin/llama-vl-embedding \
  -m ./Qwen3-VL-Embedding-2B-f16.gguf \
  --inputs '[{"text":"hello world"}]' \
  --pooling last \
  --embd-normalize 2 \
  --embd-output-format array \
  -c 4096
```

图文：

```bash
./llama.cpp/build/bin/llama-vl-embedding \
  -m ./Qwen3-VL-Embedding-2B-f16.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --inputs '[{"text":"A dog on the beach","image":"./Qwen3-VL-Embedding/data/examples/0.jpeg"}]' \
  --pooling last \
  --embd-normalize 2 \
  --embd-output-format array \
  -c 4096 \
  -ngl auto
```

## 5. 回归脚本

固定回归脚本位于：

- [`scripts/check_qwen3_vl_embedding_regression.py`](scripts/check_qwen3_vl_embedding_regression.py)

默认固定样例位于：

- [`scripts/data/qwen3_vl_embedding_regression_inputs.json`](scripts/data/qwen3_vl_embedding_regression_inputs.json)

直接跑推荐配置：

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --gguf-model ./Qwen3-VL-Embedding-2B-f16.gguf \
  --mmproj ./mmproj-Qwen3-VL-Embedding-2B-f16.gguf \
  --python-device cuda \
  --cuda-visible-devices 0 \
  --llama-ngl auto
```

如果你想稳定做 CPU 数值回归：

```bash
python ./scripts/check_qwen3_vl_embedding_regression.py \
  --python-device cpu \
  --llama-ngl 0
```

它会输出：

- 每条样例的 cosine similarity
- mean / max absolute diff
- pairwise similarity matrix 的最大差值
- Python 与 `llama-vl-embedding` 的 load / run / total timing

## 6. 为什么当前改动版更接近官方实现

这里区分两种“接近”：

- 行为接近：同样输入下，输出更接近官方 Python / HF
- 代码接近：代码组织方式更像 HF 源码

当前这套本地改动主要提升的是“行为接近”。

关键点：

- 绝对位置编码插值现在用 `qwen3vl_fast_pos_embed_interpolate()`，而不是通用 `resize_position_embeddings()`
- 视觉 `RoPE` 和 `LayerNorm` 在 `Qwen3VL` 分支里做了更明确的显式处理
- `QWEN3VL` 图像 resize 走更接近 Pillow 的 bicubic 路径
- `vl-embedding` 输入格式已对齐官方 Python 的数组输入风格

反过来说，patch embedding 主干并没有被强行全面重写，因为 upstream 的稳定 patch conv 路径对单图输入本来就基本等价。

## 7. 已验证的对齐结果

当前版本里，`vl-embedding` 与官方 Python / HF 参考实现已经在这些输入上做过对齐：

- `text`
- `image`
- `image + text`

当前固定回归样例上的典型结果：

- `text_query` cosine 约 `0.999999`
- `text_doc` cosine 约 `0.999999`
- `image_only` cosine 约 `0.99991`
- `image_text` cosine 约 `0.99991`

这说明在已经验证过的路径上，当前版本比 clean upstream 更接近官方行为。

边界也要明确：

- 这不代表和官方实现逐元素 bit-identical
- 当前重点验证的是 `text`、`image`、`image + text`
- 视频链路还没有做同等级别的完整验证

## 8. GPU vision 崩溃修复

之前 `Qwen3-VL` 的视觉 CUDA offload 路径出现过 `signal 11`。根因不是通用 CUDA graph 问题，而是 host 侧直接读取了已经 offload 到 GPU backend 的 `position_embeddings`。

修复方式是：

- 先用 `ggml_backend_tensor_get()` 把 `position_embeddings` 拉回 host
- 再在 host 上执行 `fast_pos_embed_interpolate`

当前版本已经修掉这个问题，`mmproj` 可以正常走 GPU offload；如果你还在旧 build 上看到图像一开始就崩，先重新编译。

## 9. 量化

先编译量化工具：

```bash
cmake --build ./llama.cpp/build --target llama-quantize -j
```

再量化主模型，例如 `Q8_0`：

```bash
./llama.cpp/build/bin/llama-quantize \
  ./Qwen3-VL-Embedding-2B-f32.gguf \
  ./Qwen3-VL-Embedding-2B-Q8_0.gguf \
  Q8_0
```

对 embedding 用途的建议：

- 正式替代 PyTorch：优先 `F16 + mmproj-f16`
- 保留最高数值基线：保留一份 `F32`
- 追求极限速度：可以试 `Q8_0 + mmproj-f16`，但要接受精度退化

## 10. 与 PyTorch 参考实现对比

下面这组结果来自同一批固定 5 条样例、同一张 `RTX 2080 Ti`：

| 配置 | Python 参考实现 | llama.cpp | 一致性 |
|---|---:|---:|---|
| `F32 + mmproj-f32` | load `10.283s`, run `0.963s`, total `11.246s` | load `16.159s`, prompt eval `2.160s`, total `17.515s`, wall `19.601s` | 回归通过 |
| `F32 + mmproj-f16` | load `9.143s`, run `1.848s`, total `10.991s` | load `13.592s`, prompt eval `1.982s`, total `14.766s`, wall `17.093s` | 回归通过 |
| `F16 + mmproj-f16` | load `9.546s`, run `0.926s`, total `10.472s` | load `7.300s`, prompt eval `0.377s`, total `7.566s`, wall `8.472s` | 回归通过 |
| `Q8_0 + mmproj-f16` | load `5.639s`, run `0.962s`, total `6.601s` | load `3.907s`, prompt eval `0.326s`, total `4.177s`, wall `5.016s` | 回归失败 |

说明：

- Python 参考模型的 HF 权重本身是 `bfloat16`
- 所以最公平的运行时对比其实是 `PyTorch bf16` 对 `llama.cpp F16`
- `F32` 更适合作为数值基线，而不是这张卡上的最佳速度配置

当前推荐：

- 要正式替代 PyTorch 跑 embedding：优先 `F16 + mmproj-f16`
- 要留一份高精度 baseline：再保留一份 `F32`
- 要极限速度：再考虑 `Q8_0`

## 11. 相关代码

核心文件：

- [`llama.cpp/tools/mtmd/clip.cpp`](llama.cpp/tools/mtmd/clip.cpp)
- [`llama.cpp/tools/mtmd/models/qwen3vl.cpp`](llama.cpp/tools/mtmd/models/qwen3vl.cpp)
- [`llama.cpp/examples/embedding/vl-embedding.cpp`](llama.cpp/examples/embedding/vl-embedding.cpp)

辅助脚本：

- [`scripts/check_qwen3_vl_embedding_regression.py`](scripts/check_qwen3_vl_embedding_regression.py)
- [`scripts/convert_and_regress_qwen3_vl_embedding.sh`](scripts/convert_and_regress_qwen3_vl_embedding.sh)
