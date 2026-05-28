# GraphGPT Backdoor评估使用说明

## 概述

本文档说明如何使用修改后的 `run_graphgpt.py` 测试backdoored GNN的ASR（攻击成功率）。

## 主要修改

根据 `dual_backdoor_trainer_graphgpt.py` 中的训练过程和 `eval_model` 函数的实现，我们对测试代码进行了以下修改：

1. **模型加载方式** - 完全参考 `eval_model` 的加载方式，使用 `initialize_graph_modules()` 而不是手动加载CLIP
2. **Conversation模板** - 使用 `conversation_lib.default_conversation` 与 `eval_model` 保持一致
3. **Graph Trigger插入方式** - 与训练代码中的 `_insert_trainable_trigger()` 保持一致
4. **只测试ASR** - 不添加text soft prompt，只在graph上添加trigger
5. **支持加载backdoored GNN** - 可以指定backdoored模型路径
6. **生成参数** - 与 `eval_model` 一致（max_new_tokens=200）

## 使用方法

### 基本命令

```bash
python graphgpt/eval/run_graphgpt.py \
    --eval_mode backdoor \
    --model-name /path/to/graphgpt/model \
    --backdoored_gnn_path /path/to/backdoored/gnn \
    --graph_trigger_path /path/to/graph_trigger.pt \
    --graph_data_path /path/to/graph_data.pt \
    --prompting_file /path/to/test_prompts.json \
    --output_res_path ./backdoor_results \
    --start_id 0 \
    --end_id 541
```

### 参数说明

- `--eval_mode`: 评估模式，设置为 `backdoor` 进行后门攻击评估
- `--model-name`: GraphGPT主模型路径
- `--backdoored_gnn_path`: **重要** - backdoored GNN模型路径（包含 `backdoored_clip_model.pkl` 和 `config.json`）
- `--graph_trigger_path`: graph trigger文件路径（`.pt`文件）
- `--graph_data_path`: 图数据文件路径
- `--prompting_file`: 测试问题文件路径
- `--output_res_path`: 结果输出路径
- `--start_id` / `--end_id`: 测试样本范围

### 示例（WikiCS数据集）

```bash
python graphgpt/eval/run_graphgpt.py \
    --eval_mode backdoor \
    --model-name /root/autodl-tmp/GraphGPT/checkpoints/stage_2/checkpoint-1200 \
    --backdoored_gnn_path /root/autodl-tmp/GraphGPT/backdoor_res/wikics \
    --graph_trigger_path /root/autodl-tmp/GraphGPT/backdoor_res/wikics/wikics_graph_trigger.pt \
    --graph_data_path /root/autodl-tmp/GraphCLIP/processed_data/wikics.pt \
    --prompting_file /root/autodl-tmp/GraphCLIP/graphgpt_eval/wikics_graphgpt_test.json \
    --output_res_path ./backdoor_results \
    --start_id 0 \
    --end_id 100
```

## 代码逻辑说明

### 1. 模型加载方式（参考eval_model）

`eval_model_with_trigger()` 函数的模型加载方式与 `eval_model()` 完全一致：

```python
# 1. 加载tokenizer和model
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = GraphLlamaForCausalLM.from_pretrained(args.model_name, ...)

# 2. 使用initialize_graph_modules初始化graph模块
model_graph_dict = model.get_model().initialize_graph_modules(
    graph_tower=graph_tower_name,  # 可以是backdoored模型路径
    graph_select_layer=-2,
    pretrain_graph_mlp_adapter=pretrain_graph_mlp_adapter,
    fsdp=[]
)

# 3. 获取graph_tower
graph_tower = model.get_model().graph_tower
```

### 2. Graph Trigger插入

`insert_graph_trigger()` 函数与训练代码中的 `_insert_trainable_trigger()` 保持一致：

```python
# 1. 获取原始图信息
orig_num_nodes = graph_data.graph_node.size(0)
orig_x = graph_data.graph_node

# 2. 添加trigger节点
trigger_x = graph_trigger.x
new_x = torch.cat([orig_x, trigger_x], dim=0)

# 3. 添加trigger边（内部边）
trigger_edge_index = graph_trigger.edge_index + orig_num_nodes

# 4. 连接trigger到根节点（target_node或root_n_index）
# 创建双向边
```

### 3. 评估流程

`eval_model_with_trigger()` 函数执行以下步骤：

1. 加载GraphGPT模型和backdoored GNN（使用initialize_graph_modules）
2. 对每个测试样本：
   - 加载原始graph
   - **插入graph trigger**（不添加text soft prompt）
   - 使用 `conversation_lib.default_conversation` 构造prompt
   - 将poisoned graph输入backdoored GNN
   - 获取模型预测结果（max_new_tokens=200）
3. 保存所有预测结果到JSON文件

### 4. 输出结果

结果保存在 `output_res_path/backdoor_asr_results.json`，包含：

```json
[
  {
    "id": "wikics_0",
    "node_idx": 123,
    "res": "模型的预测输出...",
    "ground_truth": "正确答案",
    "original_nodes": 50,
    "poisoned_nodes": 53
  },
  ...
]
```

## 与训练代码和eval_model的一致性

| 组件 | 训练代码 | 测试代码（eval_model_with_trigger） | eval_model |
|------|---------|---------|---------|
| 模型加载方式 | 手动加载CLIP | `initialize_graph_modules()` ✓ | `initialize_graph_modules()` ✓ |
| Conversation模板 | `conv_templates` | `default_conversation` ✓ | `default_conversation` ✓ |
| 生成参数 | - | max_new_tokens=200 ✓ | max_new_tokens=200 ✓ |
| Trigger插入方式 | `_insert_trainable_trigger()` | `insert_graph_trigger()` ✓ | - |
| 根节点识别 | 使用`root_n_index`或`target_node` | 同样逻辑 ✓ | - |
| 边连接方式 | Trigger→Root + Root→Trigger | 同样逻辑 ✓ | - |
| Text处理 | 训练时使用soft prompt | **测试时不使用** | - |
| GNN模型 | Backdoored GNN | 同一个模型 ✓ | 干净模型 |

## 注意事项

1. **不使用text soft prompt** - 测试时只在graph上添加trigger，不在text上添加任何内容
2. **Backdoored GNN路径** - 确保 `--backdoored_gnn_path` 指向包含训练好的backdoored模型的目录
3. **Trigger文件格式** - Graph trigger应该是包含 `x` 和 `edge_index` 的 `torch_geometric.data.Data` 对象
4. **根节点** - 代码会优先使用 `target_node` 作为根节点，其次是 `root_n_index`，最后fallback到节点0

## 调试建议

如果遇到问题，可以检查：

1. Graph trigger是否正确加载（查看节点数和边数）
2. Backdoored GNN路径是否正确
3. 查看每个样本的节点数变化（original_nodes vs poisoned_nodes）
4. 检查模型预测输出是否符合预期

## 下一步

评估完成后，可以：
1. 分析预测结果，计算ASR
2. 对比带trigger和不带trigger的预测差异
3. 可视化攻击效果
