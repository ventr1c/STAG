# Trojaning the Alignment: Stealthy Backdoor Attacks against Graph Foundation Models

This is the official implementation of "Trojaning the Alignment: Stealthy Backdoor Attacks against Graph Foundation Models". This repository demonstrates a dual-modality backdoor attack against GraphCLIP and GraphGPT. 

## Contents
- [Project File Overview](#project-file-overview)
- [Environment Setup](#environment-setup)
- [Step 1: Run the Dual Attack (Required for Both Victims)](#step-1)
- [Step 2: GraphCLIP - Run the Soft-Prompt Backdoor Test](#step-2)
- [Step 3: GraphGPT - Fine-Tune Then Test](#step-3)
- [Tips](#tips)

<a id="project-file-overview"></a>
## Project Overview
- `dual_attack_cotraining.py`: Main entry for the dual-modality backdoor co-training pipeline.
- `dual_backdoor_trainer.py`: Core training logic for trigger optimization, poisoning, and checkpoint saving.
- `test_soft_prompt_backdoor_attack.py`: GraphCLIP-side evaluation (clean accuracy, ASR, optional defense).
- `test_backdoor_gnn.py`: GraphGPT-side backdoor evaluation after alignment/fine-tuning.
- `convert_to_graphgpt_format.py`, `merge_datasets.py`: Data conversion/merging utilities for GraphGPT workflows.
- `data/`: Dataset loading, splitting, sampling, and dataset-specific loaders in `data/data_utils/`.
- `graphclip/`: GraphCLIP model components and config files.
- `graphgpt/`: GraphGPT training/evaluation code, model adapters, and graph-related layers.
- `text-graph-grounding/`: Supporting text-graph grounding modules used by the project.
- `processed_data/`: Preprocessed graph datasets (for example `processed_data/cora.pt`).
- `backdoor_res/`: Attack outputs and logs (triggers, soft prompts, poisoned models, test logs).
- `checkpoints/`: Fine-tuned/alignment checkpoints (including GraphGPT stage outputs).
- `requirements.txt`: Python dependency list.
- `tests/`: API/utility test scripts.

<a id="environment-setup"></a>

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Prerequisites:
- Graph datasets saved in `processed_data/<dataset>.pt` (cora, citeseer, wikics, ogbn-arxiv, etc.).
- Victim checkpoints inside `backdoor_res/` (GraphCLIP weights or GraphGPT base models).
- Instruction JSONs for GraphGPT (e.g., `ogbn-arxiv_graphgpt_train.json`).
-  `checkpoint/stage_2` is the model trained by using backdoored GNN in GraphGPT. 

<a id="step-1"></a>
## Step 1 - Run the Dual Attack (Required for Both Victims)
```bash
python dual_attack_cotraining.py 
  --dataset cora 
  --victim graphgpt
  --device cuda 
  --poison_rate 0.3 
  --num_trigger_node 8
  --soft_prompt_len 15
  --target_class 2 
  --epochs_text 20 
  --epochs_gnn 15 
  --output_dir backdoor_res/cora
```
Switch `--victim` to `graphgpt` and adjust hyperparameters for other datasets. Artifacts (graph trigger, soft prompt, GraphStructureNet, poisoned checkpoint, logs) are saved under `backdoor_res/<run_name>/`.



<a id="step-2"></a>

## Step 2 - GraphCLIP: Run the Soft-Prompt Backdoor Test

Use `test_soft_prompt_backdoor_attack.py` to evaluate the poisoned GraphCLIP checkpoint, soft prompt, and trigger that Step 1 produced. The script rebuilds the soft prompt, injects triggers into clean graphs, applies optional defenses, and reports both clean accuracy and ASR.
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
  --defense_method od
```
- `--result_dir` should match the attack output folder that contains the GraphCLIP checkpoint, `graph_structure_net*.pth`, soft prompt weights, and target embedding.
- `--defense_method` controls the inference defense (`od`, `prune`, or `none`). Dominant OD models are trained automatically when `od` is selected.
- The clean/poison evaluation uses `processed_data/<dataset>.pt`; ensure it matches the dataset used in Step 1.
- Results are logged to `soft_prompt_backdoor_test.log`, and intermediate trigger embeddings are stored for optional visualization.

<a id="step-3"></a>
## Step 3 - GraphGPT: Fine-Tune Then Test
GraphGPT must ingest the poisoned soft prompt first. Run `graphgpt/train/train_graph.py` (Windows path: `graphgpt\train\train_graph.py`) with the checkpoint produced in Step鈥?:
```bash
python graphgpt/train/train_graph.py 
  --model_name_or_path GraphGPT-7B-mix-all 
  --graph_tower clip_gt_arxiv 
  --data_path graphgpt/data/cora_graphgpt_train.json 
  --graph_data_path processed_data/cora.pt 
  --output_dir checkpoints/stage_2 
  --pretrain_graph_model_path checkpoints/stage_2/cora_graph_projector/checkpoint
  --pretrain_graph_mlp_adapter checkpoints/stage_2/cora_graph_projector/checkpoint.bin
  --per_device_train_batch_size 1 
  --gradient_accumulation_steps 16 
  --num_train_epochs 3 
  --learning_rate 2e-5 
  --bf16 True
```
After alignment, evaluate GraphGPT with the same testing script:
```bash
python test_backdoor_gnn.py
  --dataset cora
  --device cuda
  --pretrain_graph_model_path checkpoints/stage_2
  --backdoor_model_path backdoor_res/cora/graphgpt_backdoor_model.pt
  --graph_trigger_path backdoor_res/cora/cora_graph_trigger.pt
  --graph_structure_net_path backdoor_res/cora/graph_structure_net.pt
  --target_class 2
```

<a id="tips"></a>
## Tips
- Adjust `--text_trigger_tokens`, `--num_trigger_node`, and `--poison_rate` for each dataset to balance ASR and clean performance.
- All scripts accept `--seed` for reproducibility; logs are written alongside artifacts in `backdoor_res/`.
- When onboarding a new dataset, regenerate `processed_data/<dataset>.pt` and prepare matching instruction JSON files for GraphGPT fine-tuning.



