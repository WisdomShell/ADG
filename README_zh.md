<div align="center">

# 基于答案分歧的指令微调数据筛选

<p>
  <a href="https://github.com/WisdomShell/ADG">English</a> | <strong>简体中文</strong>
</p>

<a href="https://wisdomshell.github.io/ADG/"><img src="https://img.shields.io/badge/Project-Page-green?logo=githubpages&logoColor=white" /></a>
<a href="https://arxiv.org/abs/2604.07892"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" /></a>
<a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/Venue-ACL%202026-blue" /></a>
[![Task](https://img.shields.io/badge/Task-Data%20Selection-purple.svg)](#overview)
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />

**ACL 2026 Main Conference**

<a href="https://deepblue666.github.io/">Bo Li</a>, Mingda Wang, Shikun Zhang, Wei Ye

</div>

本仓库公开了 **Answer Divergence-Guided Selection（ADG）** 的核心实现流程，用于指令数据选择。ADG 不再依赖单一 reference response 来给样本打分，而是基于同一条 instruction 在随机解码下产生的多样本回答的几何结构进行评分。论文中，ADG 在固定 10K 数据预算下，在两种 backbone、三个公开 instruction pool 和六个 benchmark 上都取得了稳定优势。方法核心由 **dispersion magnitude**、**shape anisotropy** 和 **bin-wise selection** 三部分组成。

---

## 🌟 概述

在固定数据预算下，instruction tuning 的效果高度依赖于究竟选择了哪些训练样本。ADG 的核心思想是观察基础模型在随机解码条件下，对同一条 instruction 会给出怎样的一组回答。

对于每条 instruction，ADG 的主要流程是：

1. 以相对较高温度采样多个回答，
2. 将这些回答映射到表示空间，
3. 计算与几何结构相关的分数，
4. 按照组合分数对样本排序，
5. 在语义 bin 内按比例选择样本。

本仓库提供了以下完整流程：
- 多样本答案生成，
- instruction embedding 与聚类，
- ADG 打分与子集选择，
- 模型训练，
- benchmark 评测，
- 可选的数据类型分析。

---

## 📦 公开内容

本仓库包含以下组成部分：

### 核心选择代码
- `ADG/ADG_llama.py`  
  面向 LLaMA backbone 的 ADG 打分与样本选择脚本。

- `ADG/ADG_qwen.py`  
  面向 Qwen backbone 的 ADG 打分与样本选择脚本。

### 答案生成与指令嵌入
- `generation/generation.py`  
  为每条 instruction 生成多个采样答案。

- `generation/embedding/embed.py`  
  构建 instruction embedding，并执行聚类，为 bin-wise selection 提供语义分箱。

### 训练与评测
- `train/train_llama.sh`  
  LLaMA 的训练入口脚本。

- `train/train_qwen.sh`  
  Qwen 的训练入口脚本。

- `train/training/stanford_alpaca/`  
  训练工具与 backbone 对应训练脚本。

- `eval/eval.sh`  
  基于 `lm-evaluation-harness` 的评测脚本。

### 分析
- `analysis/analyse.py`  
  可选的任务类型分析脚本，用于对筛选结果做数据层面的分析。

### 环境
- `requirements.txt`  
  本仓库所需依赖文件。

---

## 🗂️ 仓库结构

```text
.
├── README.md
├── README_zh.md
├── requirements.txt
├── ADG/
│   ├── ADG_llama.py
│   └── ADG_qwen.py
├── generation/
│   ├── generation.py
│   └── embedding/
│       └── embed.py
├── analysis/
│   └── analyse.py
├── eval/
│   └── eval.sh
└── train/
    ├── train_llama.sh
    ├── train_qwen.sh
    └── training/
        └── stanford_alpaca/
            ├── train_llama.py
            ├── train_qwen.py
            ├── utils.py
            └── configs/
```

---

## ⚙️ 安装

建议使用 Python 3.10 及以上版本。

示例：

```bash
conda create -n adg python=3.12.9
conda activate adg
pip install -r requirements.txt
```

根据你的运行环境，可能还需要额外安装 GPU 相关依赖。

---

## 🧾 数据格式

ADG 接收 JSON 或 JSONL 格式的 instruction 数据。每条样本建议为如下格式：

```json
{
  "id": 0,
  "instruction": "Write a short explanation of transformers.",
  "input": "",
  "output": "Transformers are neural networks based on self-attention..."
}
```

说明：
- `id` 应唯一标识一个样本。
- `instruction` 为必需字段。
- `input` 是可选字段，可以为空或省略。
- `output` 是原始 instruction 数据中的 reference response。
- 其他 instruction 数据集只要转换为该格式，也可以直接用于 ADG。

在多样本生成之后，中间 JSONL 文件中的记录形式如下：

```json
{
  "id": 0,
  "instruction": "Write a short explanation of transformers.",
  "output": "Transformers are neural networks based on self-attention...",
  "generated_answers": [
    "...",
    "...",
    "...",
    "...",
    "..."
  ]
}
```

---

## 🔄 方法流程

实际运行流程如下：

```text
instruction pool
    -> generation/generation.py
    -> 多样本答案 JSONL
    -> generation/embedding/embed.py
    -> instruction embeddings + cluster labels
    -> ADG/ADG_llama.py 或 ADG/ADG_qwen.py
    -> top / middle / bottom 筛选子集
    -> train/train_*.sh
    -> 微调后的 checkpoint
    -> eval/eval.sh
```

---

## 🚀 快速开始

### 第一步：准备 instruction pool

下载并预处理你的 instruction 数据集，例如 Alpaca-GPT4、WizardLM 或 CoT，并将其整理为要求的数据格式。

### 第二步：为每条 instruction 生成多个答案

运行前，请先在 `generation/generation.py` 中修改以下变量：
- `MODEL_NAME`
- `OUTPUT_DIR`
- `OUTPUT_FILE`

然后运行：

```bash
cd generation
torchrun --nproc_per_node=4 --master_port=29500 generation.py   --input_file /path/to/your/instruction_data.json   --batch_size 32
```

### 第三步：构建 instruction embedding 与聚类结果

运行前，请先在 `generation/embedding/embed.py` 中修改以下变量：
- `MODEL_NAME`
- `INPUT_JSONL`
- `EMBEDDINGS_PATH`
- `CLUSTERS_PATH`
- `K_CLUSTERS`

然后运行：

```bash
torchrun --nproc_per_node=4 --master_port=29501 generation/embedding/embed.py
```

### 第四步：执行 ADG 打分与样本选择

选择与你的 backbone 对应的脚本。

对于 LLaMA，请先在 `ADG/ADG_llama.py` 中配置：
- `model_name`
- `INPUT_JSONL`
- `OUTPUT_DIR`
- `EMBEDDINGS_PATH`
- `CLUSTERS_PATH`
- `K_CLUSTERS`
- `FINAL_SELECT_COUNT`

然后运行：

```bash
python ADG/ADG_llama.py
```

对于 Qwen，请先在 `ADG/ADG_qwen.py` 中配置：
- `model_name`
- `INPUT_JSONL`
- `OUTPUT_DIR`
- `EMBEDDINGS_PATH`
- `CLUSTERS_PATH`
- `CHECKPOINT_DIR`
- `FINAL_SELECT_COUNT`

然后运行：

```bash
python ADG/ADG_qwen.py
```

筛选结果会保存在配置的 `OUTPUT_DIR` 下，包括：
- `top.json`
- `middle.json`
- `bottom.json`

### 第五步：训练 backbone 模型

通常使用筛选得到的 `top.json` 作为训练子集进行 instruction tuning。

对于 LLaMA：

```bash
cd train
bash train_llama.sh
```

对于 Qwen：

```bash
cd train
bash train_qwen.sh
```

运行前请先更新：
- `--model_name_or_path`
- `--data_path`
- `--output_dir`

### 第六步：评测训练后的 checkpoint

本仓库通过 `lm-evaluation-harness` 进行 benchmark 评测。

如有需要，先安装：

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

然后在 `eval/eval.sh` 中配置 `MODEL_PATH` 和输出路径，并运行：

```bash
cd eval
bash eval.sh
```

当前评测脚本覆盖：
- BBH
- GSM8K
- MMLU
- TruthfulQA
- MBPP
- HumanEval

---

## 📊 ADG 打分直觉

ADG 基于多样本回答构造两个互补的信号：

- **Dispersion magnitude**  
  衡量多个采样回答在表示空间中是否分得足够开。

- **Shape anisotropy**  
  衡量这种分散是否是多方向的，而不是仅沿一个方向轻微漂移。

最终 ADG 将两者结合成总分，并在语义 bin 内进行排序与按比例选择，从而避免全局 top-k 只集中在少数稠密区域。

---

## 🛠️ 脚本说明

### `generation/generation.py`
主要功能：
- 加载基础模型，
- 为每条 instruction 采样多个答案，
- 以 JSONL 格式保存生成结果，
- 支持分布式生成。

### `generation/embedding/embed.py`
主要功能：
- 构建 instruction embedding，
- 执行聚类，
- 保存 instruction embedding 和 cluster label，
- 为 ADG 的语义 bin 选择提供基础。

### `ADG/ADG_llama.py`
主要功能：
- 读取多样本答案 JSONL，
- 计算与答案几何结构相关的指标，
- 组合指标形成 ADG 分数，
- 执行基于聚类的比例选择，
- 保存 `top.json`、`middle.json`、`bottom.json`。

### `ADG/ADG_qwen.py`
主要功能：
- 计算 Qwen 多样本答案对应的 ADG 指标，
- 支持断点续跑，
- 执行与 LLaMA 版本一致的 top / middle / bottom 选择流程。

### `analysis/analyse.py`
主要功能：
- 将 instruction 分类到粗粒度任务类别，
- 用于对筛选子集进行可选的数据层面分析。

### `train/train_llama.sh` 与 `train/train_qwen.sh`
主要功能：
- 启动分布式全量微调，
- 使用选出的训练子集进行 instruction tuning。

### `eval/eval.sh`
主要功能：
- 基于 `lm-evaluation-harness` 执行 benchmark 评测，
- 覆盖 reasoning、knowledge、coding 三类任务。

---

## ❓ 常见问题

### 1. 路径没有更新
大多数脚本中的路径都是占位符，运行前请先逐一修改。

### 2. 模型与中间文件不一致
请确保 generation、embedding、ADG scoring 和 training 使用的 backbone 保持一致。

### 3. 缺少中间文件
ADG 选择脚本依赖以下中间结果：
- 多样本答案 JSONL，
- instruction embedding，
- 聚类结果。

请先完成前面的阶段，再执行选择。

### 4. GPU 显存压力较大
生成、embedding 和 scoring 都涉及 hidden-state 级处理。若显存不足，请尝试减小 batch size 或重新分配 GPU。

### 5. 评测依赖未安装
`eval/eval.sh` 依赖 `lm-evaluation-harness`，请先单独安装后再运行。

---

## 📖 引用

如果你使用了本仓库，请同时引用对应论文。

```bibtex

---
