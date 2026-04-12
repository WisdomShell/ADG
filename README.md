<div align="center">

# Instruction Data Selection via Answer Divergence

<p>
  <strong>English</strong> | <a href="./README_zh.md">简体中文</a>
</p>

<a href="https://wisdomshell.github.io/ADG/"><img src="https://img.shields.io/badge/Project-Page-green?logo=githubpages&logoColor=white" /></a>
<a href="https://arxiv.org/abs/2604.07892"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white" /></a>
<a href="https://2026.aclweb.org/"><img src="https://img.shields.io/badge/Venue-ACL%202026-blue" /></a>
[![Task](https://img.shields.io/badge/Task-Data%20Selection-purple.svg)](#overview)
<img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />

**ACL 2026 Main Conference**

<a href="https://deepblue666.github.io/">Bo Li</a>, Mingda Wang, Shikun Zhang, Wei Ye

</div>

This repository releases the core pipeline of **Answer Divergence-Guided Selection (ADG)** for instruction data selection. ADG scores each instruction by the geometric structure of multiple sampled answers, rather than relying on a single reference response. In the paper, ADG consistently improves instruction tuning under a fixed 10K budget across two backbones, three public instruction pools, and six benchmarks spanning reasoning, knowledge, and coding. The method combines **dispersion magnitude** and **shape anisotropy**, then performs **bin-wise selection** for semantic coverage. 

---

## 🌟 Overview

Instruction tuning quality depends heavily on which examples are selected under a fixed data budget. ADG addresses this by examining how a base model responds to the same instruction under stochastic decoding.

For each instruction, ADG:

1. samples multiple answers with relatively high-temperature decoding,
2. maps answers into a representation space,
3. computes geometry-aware scores from the sampled answers,
4. ranks examples by the combined score,
5. performs proportional selection within semantic bins.

This repository provides the practical pipeline for:
- multi-sample answer generation,
- instruction embedding and clustering,
- ADG scoring and subset selection,
- model training,
- benchmark evaluation,
- optional task-type analysis.

---

## 📦 What Is Released

This repository includes the following components:

### Core selection code
- `ADG/ADG_llama.py`  
  ADG scoring and selection for the LLaMA backbone.

- `ADG/ADG_qwen.py`  
  ADG scoring and selection for the Qwen backbone.

### Answer generation and instruction embedding
- `generation/generation.py`  
  Generates multiple sampled answers for each instruction.

- `generation/embedding/embed.py`  
  Builds instruction embeddings and performs clustering for bin-wise selection.

### Training and evaluation
- `train/train_llama.sh`  
  Training entry script for LLaMA.

- `train/train_qwen.sh`  
  Training entry script for Qwen.

- `train/training/stanford_alpaca/`  
  Training utilities and backbone-specific training scripts.

- `eval/eval.sh`  
  Evaluation script based on `lm-evaluation-harness`.

### Analysis
- `analysis/analyse.py`  
  Optional task-type classification script for analyzing selected data.

### Environment
- `requirements.txt`  
  Required Python packages for this repository.

---

## 🗂️ Repository Structure

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

## ⚙️ Installation

We recommend Python 3.10 or above.

Example:

```bash
conda create -n adg python=3.12.9
conda activate adg
pip install -r requirements.txt
```

Depending on your environment, you may also need to install GPU-specific packages separately.

---

## 🧾 Data Format

ADG expects instruction datasets in JSON or JSONL format. Each example should follow the schema below:

```json
{
  "id": 0,
  "instruction": "Write a short explanation of transformers.",
  "input": "",
  "output": "Transformers are neural networks based on self-attention..."
}
```

Notes:
- `id` should uniquely identify each example.
- `instruction` is required.
- `input` is optional and can be empty or omitted.
- `output` is the reference response in the original instruction dataset.
- Other instruction datasets can be used as long as they are converted into this format.

After answer generation, the intermediate JSONL file contains records like:

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

## 🔄 Pipeline

The practical workflow is:

```text
instruction pool
    -> generation/generation.py
    -> multi-sample answer JSONL
    -> generation/embedding/embed.py
    -> instruction embeddings + cluster labels
    -> ADG/ADG_llama.py or ADG/ADG_qwen.py
    -> top / middle / bottom selected subsets
    -> train/train_*.sh
    -> finetuned checkpoints
    -> eval/eval.sh
```

---

## 🚀 Quick Start

### Step 1. Prepare the instruction pool

Download and preprocess your instruction dataset, such as Alpaca-GPT4, WizardLM, or CoT, into the required format.

### Step 2. Generate multiple answers per instruction

Before running, update the following variables in `generation/generation.py`:
- `MODEL_NAME`
- `OUTPUT_DIR`
- `OUTPUT_FILE`

Then run:

```bash
cd generation
torchrun --nproc_per_node=4 --master_port=29500 generation.py   --input_file /path/to/your/instruction_data.json   --batch_size 32
```

### Step 3. Build instruction embeddings and clustering results

Before running, update the following variables in `generation/embedding/embed.py`:
- `MODEL_NAME`
- `INPUT_JSONL`
- `EMBEDDINGS_PATH`
- `CLUSTERS_PATH`
- `K_CLUSTERS`

Then run:

```bash
torchrun --nproc_per_node=4 --master_port=29501 generation/embedding/embed.py
```

### Step 4. Run ADG scoring and selection

Choose the scoring script that matches your backbone.

For LLaMA, configure these variables in `ADG/ADG_llama.py`:
- `model_name`
- `INPUT_JSONL`
- `OUTPUT_DIR`
- `EMBEDDINGS_PATH`
- `CLUSTERS_PATH`
- `K_CLUSTERS`
- `FINAL_SELECT_COUNT`

Then run:

```bash
python ADG/ADG_llama.py
```

For Qwen, configure these variables in `ADG/ADG_qwen.py`:
- `model_name`
- `INPUT_JSONL`
- `OUTPUT_DIR`
- `EMBEDDINGS_PATH`
- `CLUSTERS_PATH`
- `CHECKPOINT_DIR`
- `FINAL_SELECT_COUNT`

Then run:

```bash
python ADG/ADG_qwen.py
```

The selector saves:
- `top.json`
- `middle.json`
- `bottom.json`

under the configured `OUTPUT_DIR`.

### Step 5. Train the backbone model

Use the selected subset, typically `top.json`, for instruction tuning.

For LLaMA:

```bash
cd train
bash train_llama.sh
```

For Qwen:

```bash
cd train
bash train_qwen.sh
```

Before running, update paths such as:
- `--model_name_or_path`
- `--data_path`
- `--output_dir`

### Step 6. Evaluate the trained checkpoint

This repository uses `lm-evaluation-harness` for benchmark evaluation.

Install it first if needed:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

Then configure `MODEL_PATH` and output paths in `eval/eval.sh`, and run:

```bash
cd eval
bash eval.sh
```

The evaluation script currently includes:
- BBH
- GSM8K
- MMLU
- TruthfulQA
- MBPP
- HumanEval

---

## 📊 ADG Scoring Intuition

ADG is built around two complementary signals derived from multiple sampled answers:

- **Dispersion magnitude**  
  Measures how widely the sampled answers spread in representation space.

- **Shape anisotropy**  
  Measures whether the spread is multi-directional rather than dominated by a single direction.

The final ADG score combines these two parts, and the selected subset is obtained through semantic bin-wise ranking. This design helps avoid collapsing selection into only a few dense instruction regions.

---

## 🛠️ Script Notes

### `generation/generation.py`
Main functionality:
- load the base model,
- sample multiple answers for each instruction,
- save generated answers in JSONL format,
- support distributed generation.

### `generation/embedding/embed.py`
Main functionality:
- build instruction embeddings,
- run clustering,
- save instruction embeddings and cluster labels,
- provide the semantic bins used by ADG selection.

### `ADG/ADG_llama.py`
Main functionality:
- read the generated-answer JSONL file,
- compute answer-geometry metrics,
- combine metrics into the ADG score,
- perform proportional cluster-based selection,
- save `top.json`, `middle.json`, and `bottom.json`.

### `ADG/ADG_qwen.py`
Main functionality:
- compute ADG metrics for Qwen-generated answers,
- support checkpoint-based resumption,
- perform the same top / middle / bottom selection pipeline.

### `analysis/analyse.py`
Main functionality:
- classify instructions into coarse task categories,
- support optional data-level analysis of selected subsets.

### `train/train_llama.sh` and `train/train_qwen.sh`
Main functionality:
- launch distributed full fine-tuning,
- use the selected subset for instruction tuning.

### `eval/eval.sh`
Main functionality:
- run benchmark evaluation with `lm-evaluation-harness`,
- support reasoning, knowledge, and coding tasks.

---

## ❓ Common Issues

### 1. Path configuration is not updated
Most scripts use placeholder paths. Update all required paths before running.

### 2. Inconsistent model and intermediate files
Make sure the generation backbone, embedding backbone, ADG scoring script, and training script are aligned.

### 3. Missing intermediate files
The selector depends on:
- generated answer JSONL,
- instruction embeddings,
- clustering results.

Run the previous stages before starting ADG selection.

### 4. GPU memory pressure
Generation, embedding, and scoring all use hidden-state-based processing. You may need to reduce batch size or adjust GPU allocation depending on your hardware.

### 5. Evaluation dependency is not installed
`eval/eval.sh` depends on `lm-evaluation-harness`. Install it separately before running evaluation.

---

## 📖 Citation

If you use this repository, please cite the paper.

---
