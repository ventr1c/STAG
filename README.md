# Trojaning the Alignment: Stealthy Backdoor Attacks against Graph Foundation Models

Official implementation for the submitted paper "Trojaning the Alignment: Stealthy
Backdoor Attacks against Graph Foundation Models."

## Contents

- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Victim GFMs](#victim-gfms)
- [Baselines and Defenses](#baselines-and-defenses)
- [Evaluation Protocol](#evaluation-protocol)
- [Run the Attack](#run-the-attack)
- [Trigger-Text Generation Prompts](#trigger-text-generation-prompts)
- [Evaluate GraphCLIP](#evaluate-graphclip)
- [Evaluate GraphGPT](#evaluate-graphgpt)

## Project Overview

- `dual_attack_cotraining.py`: main entry for optimizing the graph trigger and
  text-side soft prompt.
- `dual_backdoor_trainer.py`: training logic for trigger optimization,
  poisoning, checkpoint saving, and artifact generation.
- `test_soft_prompt_backdoor_attack.py`: GraphCLIP-side evaluation for clean
  accuracy, attack success rate, and optional defenses.
- `test_backdoor_gnn.py`: GraphGPT-side evaluation after graph-text alignment
  or fine-tuning.
- `convert_to_graphgpt_format.py`, `merge_datasets.py`: data conversion and
  merging utilities for GraphGPT-style instruction data.
- `data/`: dataset loading, splitting, sampling, and dataset-specific loaders.
- `graphclip/`: GraphCLIP model components and configuration files.
- `graphgpt/`: GraphGPT training and evaluation code, model adapters, and graph
  layers.
- `text-graph-grounding/`: supporting graph-text grounding modules.
- `analysis_out/`: scripts used for embedding-closure and stealthiness
  analysis.
- `processed_data/`: expected location for preprocessed graph datasets.
- `backdoor_res/`: default output directory for learned triggers, soft prompts,
  poisoned checkpoints, and logs.
- `checkpoints/`: expected location for alignment and victim-model checkpoints.

## Environment Setup

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The experiments were run with PyTorch, PyTorch Geometric, Transformers, and
SBERT-compatible text encoders. Exact package versions are listed in
`requirements.txt`.

## Datasets

We evaluate on four text-attributed graph benchmarks used in the paper.
Preprocessed files should be placed under `processed_data/<dataset>.pt`.

| Dataset | Graph type | Node text used in experiments | Notes |
| --- | --- | --- | --- |
| `cora` | citation graph | paper text / keywords | small citation benchmark |
| `citeseer` | citation graph | title / abstract | small citation benchmark |
| `wikics` | Wikipedia article graph | article text | hyperlink graph |
| `ogbn-arxiv` | citation graph | title / abstract | large-scale OGB benchmark |

Following the paper, node texts are encoded with a frozen SBERT encoder
(`all-MiniLM-L6-v2`) into 384-dimensional features. If raw text files are not
already processed, use the loaders under `data/data_utils/` to regenerate the
corresponding `processed_data/<dataset>.pt` files.

## Victim GFMs

The paper evaluates the following GFM settings. This repository provides
executable GraphCLIP and GraphGPT pipelines, together with shared processed TAG
inputs for aligner-style evaluations.

| Victim | Paradigm | Relevant code |
| --- | --- | --- |
| GraphCLIP | LLM-as-aligner | `graphclip/`, `test_soft_prompt_backdoor_attack.py` |
| GraphGPT | LLM-as-predictor | `graphgpt/`, `test_backdoor_gnn.py` |
| G2P2-style aligner | LLM-as-aligner | processed TAG inputs and graph-text alignment utilities |

Victim checkpoints should be placed under `checkpoints/` or `backdoor_res/`
according to the command-line arguments below. Replace all placeholder paths
with local anonymous paths before running.

## Baselines and Defenses

The paper compares with three adapted backdoor baselines.

| Method | Type | Adaptation used here |
| --- | --- | --- |
| CrossBA | graph-side backdoor | applied to trigger-attached TAG subgraphs |
| PoisonPrompt | text-side backdoor | applied to node text / prompt inputs |
| BadCLIP | multimodal CLIP backdoor | adapted to the graph-text alignment setting |

The paper also evaluates three defense settings.

| Defense | Targeted signal |
| --- | --- |
| Prune | feature-level edge anomaly |
| Outlier Detection (OD) | feature reconstruction anomaly |
| DOMINANT | structural reconstruction anomaly |

## Evaluation Protocol

The default evaluation follows the paper:

- train/validation/test split ratio: `6:2:2`;
- default poison rate: `0.4`;
- default trigger size: `8` trigger nodes;
- clean utility metric: clean accuracy (ACC);
- attack metric: attack success rate (ASR), the fraction of trigger-attached
  test nodes classified into the target class;
- default random seed: `42`.

## Run the Attack

The main attack script jointly optimizes the graph-side trigger and text-side
soft prompt. The output is written to `backdoor_res/<dataset>/`.

```bash
python dual_attack_cotraining.py \
  --dataset cora \
  --victim graphclip \
  --device cuda \
  --poison_rate 0.4 \
  --trigger_node_num 8 \
  --soft_prompt_len 20 \
  --target_class 2 \
  --epochs_text 8 \
  --epochs_gnn 10 \
  --epochs_trigger 3 \
  --seed 42
```

For GraphGPT-style experiments, set `--victim graphgpt`. For other datasets,
replace `--dataset`, `--target_class`, and checkpoint paths accordingly.

Generated artifacts include:

- `backdoor_res/<dataset>/<dataset>_graph_trigger.pt`;
- `backdoor_res/<dataset>/graph_structure_net.pt`;
- `backdoor_res/<dataset>/target_embedding.pt`;
- `backdoor_res/<dataset>/soft_prompt_step1.pt`;
- poisoned graph encoder checkpoints and logs.

## Trigger-Text Generation Prompts

We generate readable trigger-node text by prompting an LLM with the original
node summary and the target-class description. The prompt template used in the
experiments is:

```text
Task:
Generate a paper summary and context analysis for a fictional research paper
node in Markdown format. This paper should belong to {target_class_text}.

Inputs:
- Template Node Summary:
  {original_summary}
- Category Description:
  {target_class_description}

Requirements:
1. The new node must belong to the category described in Category Description.
2. The modified summary should preserve several keywords from the original
   summary, but it does not need to retain many of them.
3. The generated node should plausibly reference the original node.
4. Generate the text directly; no explanation is needed.
```

An example original summary and generated trigger-node text are provided in
`prompts/graph_trigger_text_examples.md`.

## Evaluate GraphCLIP

Use `test_soft_prompt_backdoor_attack.py` to evaluate a poisoned GraphCLIP
checkpoint, soft prompt, and graph trigger. The script reports clean accuracy
and ASR, with optional defenses.

```bash
python test_soft_prompt_backdoor_attack.py \
  --dataset cora \
  --batch_size 20 \
  --num_trigger_node 8 \
  --trigger_pattern trigger_graph \
  --poisoned_node degree_max \
  --percent_nodes 0.05 \
  --target_class 2 \
  --soft_prompt_length 512 \
  --lm_type tiny \
  --result_dir backdoor_res/cora \
  --sbert_model_path /path/to/all-MiniLM-L6-v2 \
  --lm_head_path checkpoints/sbert_lm_head.pth \
  --trigger_source summary_text \
  --summary_file summary/summary-cora-modified.json \
  --defense_method none \
  --seed 42
```

Set `--defense_method` to `none`, `prune`, or `od`. DOMINANT-based structural
analysis is provided in the analysis scripts under `analysis_out/`.

## Evaluate GraphGPT

GraphGPT-style evaluation first aligns or fine-tunes the graph-language
projector using the prepared graph instruction data, then evaluates the
backdoored graph encoder.

Example alignment command:

```bash
python graphgpt/train/train_graph.py \
  --model_name_or_path /path/to/GraphGPT-7B-mix-all \
  --graph_tower clip_gt_arxiv \
  --data_path graphgpt/data/cora_graphgpt_train.json \
  --graph_data_path processed_data/cora.pt \
  --output_dir checkpoints/stage_2 \
  --pretrain_graph_model_path checkpoints/stage_2/cora_graph_projector/checkpoint \
  --pretrain_graph_mlp_adapter checkpoints/stage_2/cora_graph_projector/checkpoint.bin \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --bf16 True
```

Example backdoor evaluation command:

```bash
python test_backdoor_gnn.py \
  --dataset cora \
  --device cuda \
  --pretrain_graph_model_path checkpoints/stage_2 \
  --backdoor_model_path backdoor_res/cora/graphgpt_backdoor_model.pt \
  --graph_trigger_path backdoor_res/cora/cora_graph_trigger.pt \
  --graph_structure_net_path backdoor_res/cora/graph_structure_net.pt \
  --target_class 2 \
  --seed 42
```

## Analysis Scripts

The `analysis_out/` directory contains scripts for reproducing the diagnostic
plots and stealthiness measurements used in the paper, including embedding
closure and trigger-stealthiness analysis. These scripts are included for
reproducibility only and do not introduce additional claims beyond the submitted
paper.
