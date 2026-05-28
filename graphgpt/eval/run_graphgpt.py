import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import os
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy
import random
import numpy as np
from graphgpt import conversation as conversation_lib
import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp

# import ray

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"
from torch_geometric.utils import to_undirected


class GradWhere(torch.autograd.Function):
    """
    Custom autograd Function for gradient computation
    """
    @staticmethod
    def forward(ctx, input, thrd, device):
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None, None


class GraphStructureNet(nn.Module):
    """Graph Structure Network for generating trojan weights"""
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphStructureNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum - 1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers).to(device)
        self.feat = nn.Linear(nfeat, nout * (nfeat))
        self.edge = nn.Linear(nfeat, int(nout * (nout - 1) / 2))
        self.device = device

    def forward(self, input, thrd):
        GW = GradWhere.apply
        h = self.layers(input)
        edge_weight = self.edge(h)
        edge_weight = torch.sigmoid(edge_weight)
        edge_weight = GW(edge_weight, thrd, self.device)
        return edge_weight

def load_graph(instruct_item, graph_data_path): 
    graph_data_all = torch.load(graph_data_path)
    graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    graph_edge_index = to_undirected(graph_edge_index)
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    target_node = copy.deepcopy(graph_dict['node_idx'])
    # target_node = 0
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep = graph_data_all.x[graph_node_list] ##
    unique_sorted_values = sorted(list(set(graph_node_list)))

    # 3. 创建从原始值到连续自然数ID的映射字典
    #    {值: 排名ID}，例如 {2.1: 0, 8.3: 1, 10.5: 2, 50.7: 3}
    value_to_id_map = {value: i for i, value in enumerate(unique_sorted_values)}
    # graph_node_list = [value_to_id_map[value] for value in graph_node_list]
    target_node = value_to_id_map[target_node]
    graph_edge_index = torch.tensor([[value_to_id_map[edge.item()] for edge in graph_edge_index[0]],
                                     [value_to_id_map[edge.item()] for edge in graph_edge_index[1]]]).long()
    
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size

    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))
    
    # 添加根节点索引（用于trigger插入）
    # graph_ret.root_n_index = 0  # 假设第一个节点为根节点

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_graph_trigger(trigger_path):
    """加载预训练的graph trigger"""
    if os.path.exists(trigger_path):
        return torch.load(trigger_path)
    else:
        print(f"Graph trigger文件不存在: {trigger_path}")
        return None


def insert_graph_trigger_single(graph_data, graph_trigger, graph_structure_net=None, device='cuda'):
    """
    在subgraph的根节点插入graph trigger（只插入一次）
    与训练代码中的_insert_trainable_trigger完全一致
    """
    if graph_trigger is None:
        return graph_data
        
    if isinstance(graph_data, Data):
        # 确保图数据在正确的设备上
        graph_data = graph_data.to(device)
        
        # 获取原始图的信息
        orig_num_nodes = graph_data.graph_node.size(0)
        orig_edge_index = graph_data.edge_index.to(device)
        orig_x = graph_data.graph_node.to(device)
        
        # 获取trigger信息
        trigger_x = graph_trigger.x.to(device)
        num_trigger_nodes = trigger_x.size(0)
        trigger_edge_index = graph_trigger.edge_index.to(device) + orig_num_nodes
        
        # 获取根节点索引
        if hasattr(graph_data, 'target_node') and graph_data.target_node is not None:
            root_node_idx = graph_data.target_node.item() if torch.is_tensor(graph_data.target_node) else graph_data.target_node
        elif hasattr(graph_data, 'root_n_index') and graph_data.root_n_index is not None:
            root_node_idx = graph_data.root_n_index
        else:
            root_node_idx = 0  # fallback到第一个节点
        
        # 添加trigger节点到图中
        new_x = torch.cat([orig_x, trigger_x], dim=0)
        
        # 创建连接边：从trigger节点到根节点
        connection_edges = []
        for trigger_node_id in range(num_trigger_nodes):
            connection_edges.append([orig_num_nodes + trigger_node_id, root_node_idx])
        
        # 使用graph_structure_net生成trojan_weights
        if graph_structure_net is not None:
            with torch.no_grad():
                trojan_weights = graph_structure_net(input=graph_data.graph_node[root_node_idx], thrd=0.5)
                # 取前num_trigger_nodes个权重
                # trojan_weights = trojan_weights_full[:num_trigger_nodes]
                # 确保最后一个连接的权重为1
                if num_trigger_nodes > 0:
                    trojan_weights[-1] = 1
        else:
            # 如果没有graph_structure_net，使用默认权重
            trojan_weights = torch.ones(num_trigger_nodes, device=device)
        
        if connection_edges:
            connection_edge_index = torch.tensor(connection_edges, dtype=torch.long, device=device).t().contiguous()
            
            # trigger内部边的权重（全为1）
            # trigger_internal_weights = torch.ones(trigger_edge_index.size(1), device=device)
            
            # 组合边索引：trigger内部边 + 连接边
            trojan_edge_index = torch.cat([trigger_edge_index, connection_edge_index], dim=1)
            # 组合权重：trigger内部边权重 + 连接边权重
            trojan_edge_weights = trojan_weights
            # 反向边
            inverse_trojan_edge_index = torch.stack([trojan_edge_index[1], trojan_edge_index[0]], dim=0)
            
            # 最终边索引
            new_edge_index = torch.cat([orig_edge_index, trojan_edge_index, inverse_trojan_edge_index], dim=1)
            
            # 拼接所有weights
            all_weights = torch.cat([graph_data.weights, trojan_edge_weights, trojan_edge_weights], dim=0) if hasattr(graph_data, 'weights') else None
        else:
            new_edge_index = torch.cat([orig_edge_index, trigger_edge_index], dim=1)
            all_weights = graph_data.weights if hasattr(graph_data, 'weights') else None
        
        # 删除weights为0的边
        if all_weights is not None:
            # 找到weights不为0的边的索引
            non_zero_mask = all_weights > 0
            filtered_edge_index = new_edge_index[:, non_zero_mask]
            filtered_weights = all_weights[non_zero_mask]
        else:
            filtered_edge_index = new_edge_index
            filtered_weights = None
        
        # 找出所有在边中出现的节点
        total_num_nodes = new_x.size(0)
        # if filtered_edge_index.size(1) > 0:
        #     # 获取所有在边中出现的节点ID
        #     unique_nodes = torch.unique(filtered_edge_index)
        #
        #     # 创建节点映射：old_id -> new_id
        #     node_mapping = torch.full((total_num_nodes,), -1, dtype=torch.long, device=device)
        #     node_mapping[unique_nodes] = torch.arange(len(unique_nodes), dtype=torch.long, device=device)
        #
        #     # 筛选节点特征
        #     final_x = new_x[unique_nodes]
        #
        #     # 重新映射边索引
        #     final_edge_index = node_mapping[filtered_edge_index]
        #
        #     # 更新root_n_index
        #     new_root_n_index = node_mapping[root_node_idx].item()
        # else:
        #     # 如果没有边，保留所有节点
        #     final_x = new_x
        #     final_edge_index = filtered_edge_index
        #     new_root_n_index = root_node_idx
        #
        # # 创建新的图数据
        # poisoned_graph = Data(
        #     graph_node=final_x,
        #     edge_index=final_edge_index,
        #     target_node=torch.tensor([new_root_n_index], device=device) if hasattr(graph_data, 'target_node') else None,
        #     weights=filtered_weights
        # )
        #
        # # 保持原有的根节点索引属性
        # poisoned_graph.root_n_index = new_root_n_index
        poisoned_graph = Data(
            graph_node=new_x,
            edge_index=new_edge_index,
            target_node=torch.tensor([root_node_idx], device=device) if hasattr(graph_data, 'target_node') else None,
            weights=all_weights
        )
        poisoned_graph.root_n_index = root_node_idx
        return poisoned_graph
    else:
        print("警告: graph_data不是Data类型")
        return graph_data


def insert_graph_trigger(graph_data, graph_trigger, graph_structure_net=None, device='cuda'):
    """
    在subgraph中插入graph trigger
    与训练代码中的_insert_trainable_trigger完全一致
    """
    if graph_trigger is None:
        return graph_data
        
    if isinstance(graph_data, Data):
        # 确保图数据在正确的设备上
        graph_data = graph_data.to(device)
        
        # 获取原始图的信息
        orig_num_nodes = graph_data.graph_node.size(0)
        orig_edge_index = graph_data.edge_index.to(device)
        orig_x = graph_data.graph_node.to(device)
        
        # 获取trigger信息
        trigger_x = graph_trigger.x.to(device)
        num_trigger_nodes = trigger_x.size(0)
        base_trigger_edge_index = graph_trigger.edge_index.to(device)
        
        # 获取根节点索引
        if hasattr(graph_data, 'target_node') and graph_data.target_node is not None:
            root_node_idx = graph_data.target_node.item() if torch.is_tensor(graph_data.target_node) else graph_data.target_node
        elif hasattr(graph_data, 'root_n_index') and graph_data.root_n_index is not None:
            root_node_idx = graph_data.root_n_index
        else:
            root_node_idx = 0  # fallback到第一个节点
        
        # 为每个原始节点创建独立的trigger副本
        all_trigger_x = []
        all_trigger_edges = []
        all_connection_edges = []
        all_trigger_internal_weights = []
        all_connection_weights = []
        
        current_trigger_start_idx = orig_num_nodes  # trigger节点的起始索引
        
        for node_idx in range(orig_num_nodes):
            # 为当前节点创建trigger副本
            all_trigger_x.append(trigger_x.clone())
            
            # 调整trigger内部边的索引（加上偏移量）
            adjusted_trigger_edges = base_trigger_edge_index + current_trigger_start_idx
            # all_trigger_edges.append(adjusted_trigger_edges)
            
            # trigger内部边的权重（全为1）
            # all_trigger_internal_weights.append(torch.ones(base_trigger_edge_index.size(1), device=device))
            all_connection_edges = []
            # 创建当前节点到其trigger副本的连接边
            for trigger_node_id in range(num_trigger_nodes):
                all_connection_edges.append([current_trigger_start_idx + trigger_node_id, node_idx])
            all_trigger_edges.append(torch.cat([adjusted_trigger_edges, torch.tensor(all_connection_edges, device='cuda:0').T], dim=1))
            # 为当前节点生成trojan_weights
            if graph_structure_net is not None:
                with torch.no_grad():
                    trojan_weights_full = graph_structure_net(input=graph_data.graph_node[node_idx], thrd=0.5)
                    # 取前num_trigger_nodes个权重
                    # trojan_weights_for_node = trojan_weights_full[:num_trigger_nodes]
                    # 确保最后一个连接的权重为1
                    if num_trigger_nodes > 0:# and node_idx == graph_data.target_node:
                        # for i in range(num_trigger_nodes):
                        #     trojan_weights_full[-i-1] = 1
                        trojan_weights_full[-1] = 1
                    # trojan_weights_full[-1] = 1
                    all_connection_weights.append(trojan_weights_full)
            else:
                # 如果没有graph_structure_net，使用默认权重
                all_connection_weights.append(torch.ones(num_trigger_nodes, device=device))
            
            # 更新下一组trigger的起始索引
            current_trigger_start_idx += num_trigger_nodes
        
        # 拼接所有trigger节点特征
        if all_trigger_x:
            all_trigger_x_cat = torch.cat(all_trigger_x, dim=0)
            new_x = torch.cat([orig_x, all_trigger_x_cat], dim=0)
        else:
            new_x = orig_x
        
        # 拼接所有边
        if all_trigger_edges and all_connection_edges:
            # trigger内部边
            all_trigger_edges_cat = torch.cat(all_trigger_edges, dim=1)
            # all_trigger_internal_weights_cat = torch.cat(all_trigger_internal_weights, dim=0)
            
            # 连接边
            # connection_edge_index = torch.tensor(all_connection_edges, dtype=torch.long, device=device).t().contiguous()
            connection_weights = torch.cat(all_connection_weights, dim=0)
            
            # 组合trojan边
            # trojan_edge_index = torch.cat([all_trigger_edges_cat, connection_edge_index], dim=1)
            trojan_edge_index = all_trigger_edges_cat
            # trojan_edge_weights = torch.cat([all_trigger_internal_weights_cat, connection_weights], dim=0)
            trojan_edge_weights = connection_weights
            # 反向边
            inverse_trojan_edge_index = torch.stack([trojan_edge_index[1], trojan_edge_index[0]], dim=0)
            
            # 最终边索引
            new_edge_index = torch.cat([orig_edge_index, trojan_edge_index, inverse_trojan_edge_index], dim=1)
            
            # 拼接所有weights
            all_weights = torch.cat([graph_data.weights, trojan_edge_weights, trojan_edge_weights], dim=0) if hasattr(graph_data, 'weights') else None
        else:
            new_edge_index = orig_edge_index
            all_weights = graph_data.weights if hasattr(graph_data, 'weights') else None
        
        # # 删除weights为0的边
        if all_weights is not None:
            # 找到weights不为0的边的索引
            non_zero_mask = all_weights > 0
            filtered_edge_index = new_edge_index[:, non_zero_mask]
            filtered_weights = all_weights[non_zero_mask]
        else:
            filtered_edge_index = new_edge_index
            filtered_weights = None

        # 找出所有在边中出现的节点
        total_num_nodes = new_x.size(0)
        if filtered_edge_index.size(1) > 0:
            # 获取所有在边中出现的节点ID
            unique_nodes = torch.unique(filtered_edge_index)

            # 创建节点映射：old_id -> new_id
            node_mapping = torch.full((total_num_nodes,), -1, dtype=torch.long, device=device)
            node_mapping[unique_nodes] = torch.arange(len(unique_nodes), dtype=torch.long, device=device)

            # 筛选节点特征
            final_x = new_x[unique_nodes]

            # 重新映射边索引
            final_edge_index = node_mapping[filtered_edge_index]

            # 更新root_n_index和target_node
            if hasattr(graph_data, 'root_n_index'):
                new_root_n_index = node_mapping[graph_data.root_n_index].item()
            elif hasattr(graph_data, 'target_node'):
                new_root_n_index = node_mapping[root_node_idx].item()
            else:
                new_root_n_index = 0
        else:
            # 如果没有边，保留所有节点
            final_x = new_x
            final_edge_index = filtered_edge_index
            new_root_n_index = root_node_idx

        # 创建新的图数据（删除weights=0的边和孤立节点）
        poisoned_graph = Data(
            graph_node=final_x,
            edge_index=final_edge_index,
            target_node=torch.tensor([new_root_n_index], device=device) if hasattr(graph_data, 'target_node') else None,
            weights=filtered_weights
        )
        poisoned_graph.x = poisoned_graph.graph_node
        poisoned_graph.x = poisoned_graph.x.to(dtype=torch.float32)
        poisoned_graph.graph_node = poisoned_graph.graph_node.to(dtype=torch.float32)
        # 保持原有的根节点索引属性
        poisoned_graph.root_n_index = new_root_n_index
        # poisoned_graph = Data(
        #     graph_node=new_x,
        #     edge_index=new_edge_index,
        #     target_node=torch.tensor([root_node_idx], device=device) if hasattr(graph_data, 'target_node') else None,
        #     weights=all_weights
        # )
        # poisoned_graph.root_n_index = root_node_idx
        return poisoned_graph
    else:
        print("警告: graph_data不是Data类型")
        return graph_data


def select_test_nodes(total_nodes=25120, test_ratio=0.1, seed=42):
    """从总节点中选择测试节点"""
    random.seed(seed)
    test_size = int(total_nodes * test_ratio)
    test_nodes = random.sample(range(total_nodes), test_size)
    return sorted(test_nodes)


def calculate_accuracy(predictions, ground_truths):
    """计算准确率"""
    if len(predictions) != len(ground_truths):
        print(f"警告: 预测数量({len(predictions)})与真实标签数量({len(ground_truths)})不匹配")
        return 0.0
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.strip().lower() == gt.strip().lower():
            correct += 1
    
    return correct / len(predictions)

# def prepare_query(instruct_item): 


def run_eval_backdoor(args, num_gpus):
    """
    主要的backdoor评估函数
    只测试带trigger的ASR，不测试准确率
    参考eval_model的整体结构
    """
    # 加载trigger
    graph_trigger = load_graph_trigger(args.graph_trigger_path) if hasattr(args, 'graph_trigger_path') else None
    
    if graph_trigger is None:
        print("错误：未找到graph trigger文件，无法进行backdoor评估")
        return None
    
    print(f"成功加载graph trigger，节点数: {graph_trigger.x.size(0)}, 边数: {graph_trigger.edge_index.size(1)}")
    
    # 加载prompting文件
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    
    print(f"测试样本数量: {len(prompt_file)}")
    
    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)
    
    # 只进行带trigger的评估
    print("\n=== 开始Backdoor攻击评估（只测试ASR）===")
    res_data = eval_model_with_trigger(args, prompt_file, graph_trigger)
    
    # 打印结果
    print("\n=== Backdoor评估结果 ===")
    print(f"测试样本数量: {len(prompt_file)}")
    print(f"成功处理的样本数: {len(res_data)}")
    
    # 保存结果
    with open(osp.join(args.output_res_path, 'backdoor_asr_results.json'), 'w') as f:
        json.dump(res_data, f, indent=4, ensure_ascii=False)
    print(f"结果已保存到: {osp.join(args.output_res_path, 'backdoor_asr_results.json')}")
    
    return res_data


@torch.inference_mode()
def eval_model_with_trigger(args, prompt_file, graph_trigger):
    """
    评估带trigger的backdoor攻击成功率
    不添加text soft prompt，只在graph上添加trigger
    参考eval_model函数的加载方式和测试写法
    """
    # Model
    disable_torch_init()
    print('start loading backdoored model')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading tokenizer')

    print('start loading GraphLlama model')
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading model')

    # 使用initialize_graph_modules初始化graph模块（与eval_model一致）
    # 如果有backdoored_gnn_path，需要修改graph_tower参数
    graph_tower_name = 'clip_gt_arxiv'  # 默认
    # pretrain_graph_mlp_adapter = '/root/autodl-tmp/GraphGPT/checkpoints/stage_2/graph_projector/checkpoint-800.bin'
    
    # 如果指定了backdoored模型路径，修改graph_tower_name
    if hasattr(args, 'backdoored_gnn_path') and args.backdoored_gnn_path:
        print(f'使用backdoored GNN路径: {args.backdoored_gnn_path}')
        graph_tower_name = args.backdoored_gnn_path
        graph_tower_name = "clip_gt_arxiv"
    
    model_graph_dict = model.get_model().initialize_graph_modules(
        graph_tower=graph_tower_name,
        graph_select_layer=-2,
        pretrain_graph_mlp_adapter=args.graph_projector_path,
        fsdp=[]
    )
    model.tokenizer = tokenizer
    state_dict = torch.load("/root/autodl-tmp/GraphGPT/backdoor_res/cora/backdoored_clip_model.pkl", map_location=model.device)
    new_state_dict = {}
    for item in state_dict.items():
        if "gnn" in item[0]:
            new_key = item[0].replace("gnn.", "")
            new_state_dict[new_key] = item[1]
    model.model.graph_tower.load_state_dict(new_state_dict)
    original_training_state = {}
    for module_name, module in model.model.graph_tower.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):  # 适配所有BN层类型
            original_training_state[module_name] = module.training  # 保存原始训练状态
            module.eval()  # 切换到评估模式，不更新统计量

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    # 获取graph_tower（与eval_model一致）
    graph_tower = model.get_model().graph_tower
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

    # 加载graph_structure_net（与训练代码一致）
    graph_structure_net = None
    if hasattr(args, 'graph_structure_net_path') and args.graph_structure_net_path and os.path.exists(args.graph_structure_net_path):
        print(f'加载graph_structure_net: {args.graph_structure_net_path}')
        # 需要知道soft_prompt_dim和trigger_node_num来初始化
        # 从graph_trigger获取trigger节点数
        trigger_node_num = graph_trigger.x.size(0)
        soft_prompt_dim = graph_trigger.x.size(1)  # 使用trigger特征维度
        
        graph_structure_net = GraphStructureNet('cuda', soft_prompt_dim, trigger_node_num+1, layernum=2).cuda()
        graph_structure_net.load_state_dict(torch.load(args.graph_structure_net_path, map_location='cuda'))
        graph_structure_net.eval()
        print('graph_structure_net加载成功')

    res_data = []
    print(f'total: {len(prompt_file)}')
    print(f'Trigger info - 节点数: {graph_trigger.x.size(0)}, 边数: {graph_trigger.edge_index.size(1)}')
    
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        graph_dict = load_graph(instruct_item, args.graph_data_path)
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']

        # 添加weights属性（与训练代码一致）
        if not hasattr(graph_data, 'weights'):
            graph_data.weights = torch.ones(graph_data.edge_index.size(1), device='cuda')
        target_embedding = torch.load("/root/autodl-tmp/GraphGPT/backdoor_res/cora/target_embedding.pt").cuda()
        # 插入graph trigger（不添加text soft prompt），使用graph_structure_net
        original_num_nodes = graph_data.graph_node.size(0)
        graph_data = insert_graph_trigger(graph_data, graph_trigger, graph_structure_net, device='cuda')
        graph_data.target_embeddings = target_embedding[0].repeat(graph_data.x.shape[0], 1).to(dtype=torch.float32)
        # a = model.model.graph_tower(graph_data)
        poisoned_num_nodes = graph_data.graph_node.size(0)
        
        # 更新graph token长度
        graph_token_len = graph_data.graph_node.size(0)

        qs = instruct_item["conversations"][0]["value"]
        
        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
        # qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper, with the following information: Abstract: Demand_paging:demand paging computer operating systems demand paging opposed anticipatory paging method virtual memory management system uses demand paging operating system copies disk page physical memory attempt made access page already memory i.e page fault occurs follows process begins execution none pages physical memory many page faults occur process working set pages located physical memory example lazy loading technique demand paging follows pages brought memory executing process demands often referred lazy evaluation pages demanded process swapped secondary storage main memory contrast pure swapping memory process swapped secondary storage main memory process startup commonly achieve process page table implementation used page table maps logical memory physical memory page table uses bitwise operator mark page valid invalid valid page one currently resides main memory invalid page one currently resides secondary memory process tries access page following steps generally followed demand paging opposed loading pages immediately Question: Which of the following categories does this paper belong to: 1. Computational linguistics, 2. Databases, 3. Operating systems, 4. Computer architecture, 5. Computer security, Computer network security, Access control, Data security, Computational trust, Computer security exploits, 6. Internet protocols, 7. Computer file systems, 8. Distributed computing architecture, 9. Web technology, Web software, Web services, 10. Programming language topics, Programming language theory, Programming language concepts, Programming language classification? "
        # qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper. Question: Which of the following categories does this paper belong to: 1. Agents, 2. Machine Learning, 3. Information Retrieval, 4. Database, 5. Human Computer Interaction, 6. Artificial Intelligence? "
        if args.dataset == 'cora':
            qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper. Question: Which of the following categories does this paper belong to: 1. Case Based, 2. Genetic Algorithms, 3. Neural Networks, 4. Probabilistic Methods, 5. Reinforcement Learning, 6. Rule Learning, 7. Theory? "
        elif args.dataset == "citeseer":
            qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper. Question: Which of the following categories does this paper belong to: 1. Agents, 2. Machine Learning, 3. Information Retrieval, 4. Database, 5. Human Computer Interaction, 6. Artificial Intelligence? "
        elif args.dataset == "wikics":
            qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper. Question: Which of the following categories does this paper belong to: 1. Computational linguistics, 2. Databases, 3. Operating systems, 4. Computer architecture, 5. Computer security, 6. Internet protocols, 7. Computer file systems, 8. Distributed computing architecture, 9. Web technology, 10. Programming language? "

        qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

        # 使用default conversation（与eval_model一致）
        conv_mode = "default"
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        graph_data.graph_node = graph_data.graph_node.to(torch.float32)
        graph_data.x = graph_data.graph_node.to(torch.float32)
        
        pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data.cuda(),
                # do_sample=True,
                temperature=0.9,
                max_new_tokens=200,
                # eos_token_id=-1,
                # pad_token_id=pad_id
                # stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        # 保存结果（与eval_model格式类似，但添加更多信息）
        res_data.append({
            "id": instruct_item["id"], 
            "node_idx": instruct_item["graph"]["node_idx"], 
            "res": outputs,
            "ground_truth": instruct_item["conversations"][1]["value"] if len(instruct_item["conversations"]) > 1 else "unknown",
            "original_nodes": original_num_nodes,
            "poisoned_nodes": poisoned_num_nodes
        })
    total = 0
    correct = 0
    for items in res_data:
        total += 1
        if items["res"].lower() == 'neural networks' or items["res"].lower() == 'neural networks.' or items["res"].lower() == 'neural networks.' or items["res"].lower() != items["ground_truth"].lower():
            correct += 1
        else:
            print(items["res"].lower())
    asr = correct / total
    return res_data


@torch.inference_mode()
def eval_clean_backdoor(args, test_prompt_file, graph_trigger=None):
    """评估单个模式（干净或带trigger）- 保留用于兼容性"""
    # Model
    disable_torch_init()
    print('开始加载模型...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('模型加载完成')

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    # graph_tower = model.get_model().graph_tower
    
    # 加载graph tower
    clip_graph, args_graph= load_model_pretrained(CLIP, '/root/autodl-tmp/GraphGPT/Arxiv-PubMed-GraphCLIP-GT')
    graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    
    model.get_model().graph_tower = graph_tower.cuda()
    graph_tower.to(device='cuda', dtype=torch.float32)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

    results = []
    mode = "带trigger" if graph_trigger is not None else "干净"
    print(f'开始{mode}评估，总共{len(test_prompt_file)}个样本')
    
    for idx, instruct_item in tqdm(enumerate(test_prompt_file), desc=f"{mode}评估"):
        try:
            graph_dict = load_graph(instruct_item, args.graph_data_path)
            graph_token_len = graph_dict['graph_token_len']
            graph_data = graph_dict['graph_data']
            
            # 如果有trigger，插入trigger
            if graph_trigger is not None:
                graph_data = insert_graph_trigger(graph_data, graph_trigger, device='cuda')
                graph_token_len = graph_data.graph_node.size(0)

            qs = instruct_item["conversations"][0]["value"]
            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

            conv_mode = "graphchat_v1"
            if hasattr(args, 'conv_mode') and args.conv_mode is not None and conv_mode != args.conv_mode:
                conv_mode = args.conv_mode

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])

            input_ids = torch.as_tensor(inputs.input_ids).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            graph_data.graph_node = graph_data.graph_node.to(torch.float32)
            # graph_data.x = graph_data.x.to(torch.bfloat16)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    graph_data=graph_data.cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            results.append({
                "id": instruct_item["id"], 
                "node_idx": instruct_item["graph"]["node_idx"], 
                "res": outputs,
                "ground_truth": instruct_item["conversations"][1]["value"] if len(instruct_item["conversations"]) > 1 else "unknown"
            })
            
        except Exception as e:
            print(f"处理样本{idx}时出错: {e}")
            continue

    return results


def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_prompting_file(args.prompting_file)
    args.end_id = len(prompt_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))
    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')

    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        # ans_handles.append(
        #     eval_model.remote(
        #         args, prompt_file[start_idx:end_idx], start_split, end_split
        #     )
        # )
        eval_model(
                    args, prompt_file[start_idx:end_idx], start_split, end_split
                )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


# @ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)


    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')

    print('start loading')
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    model_graph_dict = model.get_model().initialize_graph_modules(
        graph_tower='clip_gt_arxiv',
        graph_select_layer=-2,
        pretrain_graph_mlp_adapter=args.graph_projector_path,
        fsdp=[]
    )
    # extra_path = os.path.join('/root/autodl-tmp/GraphGPT/checkpoints/stage_2/graph_projector', "checkpoint-1200.bin")
    # # extra_path = os.path.join('/root/autodl-tmp/GraphGPT/GraphGPT-7B-mix-all', "graph_projector.bin")
    # graph_projector_weights = torch.load(extra_path, map_location='cpu')
    # model.model.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})
    model.tokenizer = tokenizer
    # if os.path.exists(extra_path):
    #     print(f"[GraphLlama] Loading extra parameters from {extra_path}")
    #     extra_state = torch.load(extra_path, map_location="cuda:0")
    #
    #     # ====================== 处理 graph_projector ======================
    #     has_graph_proj = "model.graph_projector.weight" in extra_state
    #     if has_graph_proj:
    #         state = {}
    #         for k in ["weight", "bias"]:
    #             full_key = f"model.graph_projector.{k}"
    #             if full_key in extra_state:
    #                 state[k] = extra_state[full_key]
    #         model.model.graph_projector.load_state_dict(state)
    #         print("[GraphLlama] graph_projector loaded successfully.")
    #
    #     # ====================== 处理 embed_tokens ======================
    #     if "model.embed_tokens.weight" in extra_state:
    #         new_weight = extra_state["model.embed_tokens.weight"]
    #         old_weight = model.model.embed_tokens.weight.data
    #
    #         if new_weight.shape[0] != old_weight.shape[0]:
    #             print(f"[GraphLlama] Vocab size mismatch: old={old_weight.shape[0]}, new={new_weight.shape[0]}")
    #
    #             if new_weight.shape[0] > old_weight.shape[0]:
    #                 # 词表扩充
    #                 diff = new_weight.shape[0] - old_weight.shape[0]
    #                 print(f"[GraphLlama] Expanding vocab by {diff} tokens.")
    #
    #                 expanded_weight = torch.zeros_like(new_weight)
    #                 expanded_weight[:old_weight.shape[0]] = old_weight
    #                 expanded_weight[old_weight.shape[0]:] = new_weight[old_weight.shape[0]:]
    #
    #                 model.model.embed_tokens = nn.Embedding.from_pretrained(expanded_weight, freeze=False).to(device="cuda:0", dtype=torch.bfloat16)
    #             else:
    #                 # 词表缩小
    #                 print("[GraphLlama] Truncating embeddings to match new vocab.")
    #                 truncated_weight = old_weight.clone()
    #                 truncated_weight[:new_weight.shape[0]] = new_weight
    #                 model.model.embed_tokens = nn.Embedding.from_pretrained(truncated_weight, freeze=False)
    #         else:
    #             # 大小相同
    #             model.model.embed_tokens.weight.data = new_weight
    #             print("[GraphLlama] embed_tokens loaded (same vocab size).")
    #     model.model.embed_tokens.to(dtype=torch.bfloat16)
    #     print("[GraphLlama] Extra parameter loading done.")
    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)

    # graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    # clip_graph, args_graph= load_model_pretrained(CLIP, '/root/autodl-tmp/GraphGPT/Arxiv-PubMed-GraphCLIP-GT')
    # clip_graph, args_graph = load_model_pretrained(CLIP, '/root/autodl-tmp/GraphGPT/text-graph-grounding/res/wikics')
    #
    # graph_tower = graph_transformer(args_graph)
    # graph_tower = transfer_param_tograph(clip_graph, graph_tower)
    # graph_tower.to(dtype=torch.bfloat16)
    #
    # model.get_model().graph_tower = graph_tower.cuda()
    # else:
    #     print('other')
    # print(next(graph_tower.parameters()).dtype)
    # graph_tower.to(device='cuda', dtype=torch.bfloat16)
    graph_tower = model.get_model().graph_tower
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    res_data = []
    print(f'total: {len(prompt_file)}')
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        # instruct_item = prompt_file[0]
        # if idx >= 3: 
        #     break
        graph_dict = load_graph(instruct_item, args.graph_data_path)
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']

        qs = instruct_item["conversations"][0]["value"]
        # if use_graph_start_end:
        #     qs = qs + '\n' + DEFAULT_G_START_TOKEN + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len + DEFAULT_G_END_TOKEN
        # else:
        #     qs = qs + '\n' + DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len

        replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
        replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
        if args.dataset == 'cora':
            qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper. Question: Which of the following categories does this paper belong to: 1. Case Based, 2. Genetic Algorithms, 3. Neural Networks, 4. Probabilistic Methods, 5. Reinforcement Learning, 6. Rule Learning, 7. Theory? "
        elif args.dataset == "citeseer":
            qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper. Question: Which of the following categories does this paper belong to: 1. Agents, 2. Machine Learning, 3. Information Retrieval, 4. Database, 5. Human Computer Interaction, 6. Artificial Intelligence? "
        elif args.dataset == "wikics":
            qs = f"Given a citation graph:\n{DEFAULT_GRAPH_TOKEN} where the 0th node is the target paper. Question: Which of the following categories does this paper belong to: 1. Computational linguistics, 2. Databases, 3. Operating systems, 4. Computer architecture, 5. Computer security, 6. Internet protocols, 7. Computer file systems, 8. Distributed computing architecture, 9. Web technology, 10. Programming language? "
        qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)

        # if "v1" in args.model_name.lower():
        #     conv_mode = "graphchat_v1"
        # else: 
        #     raise ValueError('Don\'t support this model')
        conv_mode = "default"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conversation_lib.default_conversation.copy()
        # conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        

        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        graph_data.graph_node = graph_data.graph_node.to(torch.float32)
        graph_data.x = graph_data.graph_node.to(torch.float32)
        # graph_data.edge_index = graph_data.edge_index.to(torch.bfloat16)
        pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else model.config.eos_token_id
        model.model.graph_projector.to(dtype=torch.float32)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                graph_data=graph_data.cuda(),
                # do_sample=True,
                # temperature=0.9,
                max_new_tokens=200,
                # eos_token_id=-1,  # 关键：将EOS标记设置为-1，使其永远不会被触发
                # pad_token_id=pad_id  # 明确指定pad_token_id，避免产生警告或错误
                # stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)

        res_data.append({"id": instruct_item["id"],
                         "node_idx": instruct_item["graph"]["node_idx"],
                         "ground_truth": instruct_item["conversations"][1]["value"] if len(
                             instruct_item["conversations"]) > 1 else "unknown",
                         "res": outputs}.copy())
        # with open(osp.join(args.output_res_path, '{}_test_res_{}_{}.json'.format(start_idx, end_idx)), "w") as fout:
        #     json.dump(res_data, fout, indent=4)
    correct = 0
    for items in res_data:
        # correct = correct + items["res"]
        if items["ground_truth"].strip().lower() in items["res"].strip().lower():
            correct = correct + 1
    accuracy = correct / len(res_data)
    print("Accuracy: {:.4f}".format(accuracy))
    return res_data
    # with open(args.output_res_path, "w") as fout:
    #     json.dump(res_data, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-name", type=str, default="/root/autodl-tmp/GraphGPT/GraphGPT-7B-mix-all",)
    # parser.add_argument("--model-name", type=str, default="/root/autodl-tmp/GraphGPT/checkpoints/stage_2/wikics-checkpoint-800", )
    # parser.add_argument("--prompting_file", type=str, default=r'/root/autodl-tmp/GraphCLIP/graphgpt_eval/wikics_graphgpt_test.json',)
    # parser.add_argument("--prompting_file", type=str,
    #                     default=r'/root/autodl-tmp/GraphGPT/datasets/datasets--Jiabin99--GraphGPT-eval-instruction/snapshots/34545bb10375439597dee15ea0c620100f7322e7/pubmed_test_instruct_std.json', )
    parser.add_argument("--dataset", type=str,
                        default=r'arxiv', )

    parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--graph_data_path", type=str, default=r'/root/autodl-tmp/GraphCLIP/processed_data/wikics.pt')
    # parser.add_argument("--graph_data_path", type=str, default=r'/root/autodl-tmp/GraphGPT/datasets/datasets--Jiabin99--All_pyg_graph_data/snapshots/ce14caef50c7139a64421094f605e305551cb5d2/graph_data_all.pt')
    
    # Backdoor相关参数
    # parser.add_argument("--graph_trigger_path", type=str, default=r'/root/autodl-tmp/GraphGPT/backdoor_res/wikics/wikics_graph_trigger.pt',
    #                     help="预训练的graph trigger文件路径")
    # parser.add_argument("--backdoored_gnn_path", type=str, default=r'/root/autodl-tmp/GraphGPT/backdoor_res/wikics',
    #                     help="backdoored GNN模型路径（包含backdoored_clip_model.pkl）")
    # parser.add_argument("--graph_structure_net_path", type=str, default=r'/root/autodl-tmp/GraphGPT/backdoor_res/wikics/graph_structure_net.pt',
    #                     help="graph_structure_net模型路径")
    
    parser.add_argument("--output_res_path", type=str, default=r'./res_output')
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=541)
    parser.add_argument("--eval_mode", type=str, default="original", choices=["original", "backdoor"],
                        help="评估模式：original为原始评估，backdoor为后门攻击评估")

    args = parser.parse_args()
    args.model_name = f'/root/autodl-tmp/GraphGPT/checkpoints/stage_2/{args.dataset}-checkpoint-1000'
    args.prompting_file = f'/root/autodl-tmp/GraphCLIP/graphgpt_eval/{args.dataset}_graphgpt_test.json'
    args.graph_data_path = f'/root/autodl-tmp/GraphCLIP/processed_data/{args.dataset}.pt'
    args.graph_trigger_path = f'/root/autodl-tmp/GraphGPT/backdoor_res/{args.dataset}/{args.dataset}_graph_trigger.pt'
    args.backdoored_gnn_path = f'/root/autodl-tmp/GraphGPT/backdoor_res/{args.dataset}'
    args.output_res_path = f'./res_output/{args.dataset}_{args.eval_mode}_res'
    args.graph_structure_net_path = f'/root/autodl-tmp/GraphGPT/backdoor_res/{args.dataset}/graph_structure_net.pt'
    args.graph_projector_path = f'/root/autodl-tmp/GraphGPT/checkpoints/stage_2/{args.dataset}_graph_projector/checkpoint-1000.bin'
    if args.eval_mode == "backdoor":
        print("=" * 50)
        print("开始Backdoor攻击评估（ASR测试）...")
        print(f"Graph trigger路径: {args.graph_trigger_path}")
        print(f"Backdoored GNN路径: {args.backdoored_gnn_path}")
        print(f"Graph structure net路径: {args.graph_structure_net_path}")
        print(f"测试样本范围: {args.start_id} - {args.end_id}")
        print("=" * 50)
        poisoned_results = run_eval_backdoor(args, args.num_gpus)
        print("\n评估完成！")
    else:
        print("开始原始评估...")
        # ray.init()
        run_eval(args, args.num_gpus)


# protobuf             4.22.3