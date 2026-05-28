import copy
import os
import torch
import argparse
import logging
import numpy as np
import json
import torch_sparse
from flax.core.nn import embedding
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataListLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch_geometric.utils import subgraph, get_laplacian, to_dense_adj

from models import GraphCLIP
from data.load import load_data
from utils.process import parse_target_data, split_dataloader, parse_source_data
from soft_prompt_backdoor_trainer import SoftPromptBackdoorTrainer
from train_lm_head import SBERTWithLMHead
from soft_prompt_backdoor_trainer import GraphStructureNet
torch.autograd.set_detect_anomaly(True)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("soft_prompt_backdoor_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("soft_prompt_backdoor_test")

class SoftPromptTester:
    """基于Soft Prompt的后门攻击测试器"""
    
    def __init__(self, model, lm_head_model, tokenizer, soft_prompt_params, device, 
                 soft_prompt_length, target_class, trigger_source, target_embedding):
        """
        初始化测试器
        
        Args:
            model: 训练好的后门GraphCLIP模型
            lm_head_model: 训练好的SBERT+LM Head模型
            tokenizer: SBERT tokenizer
            soft_prompt_params: 训练好的soft prompt参数
            device: 设备
            soft_prompt_length: soft prompt长度
            target_class: 目标类别
        """
        self.model = model
        self.lm_head_model = lm_head_model
        self.tokenizer = tokenizer
        self.soft_prompt_params = soft_prompt_params
        self.device = device
        self.soft_prompt_length = soft_prompt_length
        self.target_class = target_class
        self.trigger_source = trigger_source
        self.target_text_template = "This node belongs to class {}"
        self.target_embedding = target_embedding

        if self.trigger_source == "summary_text":
            self.use_poisoned = True
        else:
            self.use_poisoned = False
        
        # 冻结所有模型参数
        for param in self.lm_head_model.parameters():
            param.requires_grad = False
        self.lm_head_model.eval()
        self.model.eval()
        
        logger.info(f"初始化SoftPromptTester，soft prompt长度: {soft_prompt_length}")
        logger.info(f"目标类别: {target_class}")
    
    def generate_trigger_features_from_text(self, input_texts, num_trigger_nodes):
        """
        使用soft prompt生成trigger特征
        
        Args:
            input_texts: 输入文本列表（通常是图的summary）
            num_trigger_nodes: 触发器节点数量
        
        Returns:
            trigger_features: 生成的触发器特征 [num_trigger_nodes, feature_dim]
        """
        self.lm_head_model.eval()
        
        with torch.no_grad():
            # 确保有足够的文本
            if len(input_texts) < num_trigger_nodes:
                input_texts = (input_texts * ((num_trigger_nodes // len(input_texts)) + 1))[:num_trigger_nodes]
            elif len(input_texts) > num_trigger_nodes:
                input_texts = input_texts[:num_trigger_nodes]
            
            trigger_features_list = []
            
            for text in input_texts:
                # 为每个输入文本生成trigger feature
                trigger_feature, sim = self._generate_single_trigger_feature(text)
                trigger_features_list.append(trigger_feature)
            
            # 堆叠所有trigger features
            trigger_features = torch.stack(trigger_features_list, dim=0)
            
        return trigger_features, sim
    
    def _generate_single_trigger_feature(self, input_text):
        """
        为单个文本生成trigger特征
        现在只使用512长度的soft prompt embeddings，不包含原始文本
        
        Args:
            input_text: 输入文本（仅用于显示，不参与trigger feature生成）
        
        Returns:
            trigger_feature: 单个trigger特征向量
        """
        if self.trigger_source == "summary_text":
            text_batch = self.tokenizer(input_text, add_special_tokens=False, truncation=True, padding=True,
                                   return_tensors="pt", max_length=512).to(self.device)

            embeddings = self.lm_head_model.sbert_model.embeddings(
                text_batch["input_ids"]
            )
            attention_mask_with_prompt = text_batch["attention_mask"]  # [1, seq_len]

        else:
            # 创建512长度的attention mask（全部为1，表示所有位置都有效）
            batch_size = 1
            attention_mask_with_prompt = torch.ones((batch_size, self.soft_prompt_length), device=self.device)

            # 直接使用soft prompt embeddings作为输入
            # 输入格式: [512长度的soft_prompt]
            embeddings = self.soft_prompt_params.unsqueeze(0)  # [1, 512, embedding_dim]
        
        # 通过SBERT处理
        outputs = self.lm_head_model.sbert_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask_with_prompt
        )

        # Mean pooling
        last_hidden_state = outputs.last_hidden_state
        logits = self.lm_head_model.lm_head(last_hidden_state)  # [1, seq_len, vocab_size]
        predicted_ids = torch.argmax(logits, dim=-1)  # [1, seq_len]

        # 解码重构的文本
        reconstructed_text = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        # print("重构文本:" + reconstructed_text)
        # print("原始文本:" + input_text)
        input_mask_expanded = attention_mask_with_prompt.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sentence_embedding = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        self.sentence_embedding = sentence_embedding
        sim = torch.nn.functional.cosine_similarity(self.target_embedding, last_hidden_state.mean(dim=1))
        return sentence_embedding.squeeze(0), sim

def subgraph_relabel(data: Data, mask: torch.Tensor):
    """
    返回重编号后的子图（节点编号从 0..n_sub-1）并构建新的 Data。
    兼容 edge_attr，如果有则同时筛选。
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    # device = data.edge_index.device if hasattr(data, 'edge_index') else (data.x.device if data.x is not None else torch.device('cpu'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    node_idx = mask.nonzero(as_tuple=True)[0].to(device)    # 全局索引
    num_nodes = node_idx.size(0)
    # rdm_mask = torch.ra
    num_samples = int(num_nodes * 0.6)

    # 生成随机排列的索引
    random_indices = torch.randperm(num_nodes)

    # 选取前60%的索引
    sampled_indices = random_indices[:num_samples]

    # 获取对应的元素
    # node_idx = node_idx[sampled_indices]
    # node_idx = node_idx[sampled_nodes]
    # 方法 A: 用 torch_geometric.utils.subgraph（如果可用，代码更短）
    # if has_subgraph_util:
    # subgraph 返回 (edge_index_sub, edge_attr_sub) 当提供 edge_attr 时，
    # 否则只返回 edge_index_sub

    edge_index_sub, _ = subgraph(node_idx, data.edge_index,
                              relabel_nodes=True,
                              num_nodes=data.num_nodes)
    x_sub = data.x[node_idx] if data.x is not None else None
    y_sub = data.y[node_idx] if data.y is not None else None

    new_data = Data(x=x_sub, edge_index=edge_index_sub, y=y_sub)

    # 记录映射信息： new_idx -> old_idx
    new_data.original_node_idx = node_idx  # helpful for mapping back
    return new_data

def prune_unrelated_edge(prune_thr,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    # update structure
    updated_edge_index = edge_index[:,edge_sims>prune_thr]
    updated_edge_weights = edge_weights[edge_sims>prune_thr]
    return updated_edge_index,updated_edge_weights

def add_soft_prompt_trigger_to_large_graph(tester, graph, target_node_ids, trigger_pattern, num_trigger_node, device, graph_structure_net):
    """
    在大图中直接针对特定节点插入trigger
    
    Args:
        tester: SoftPromptTester实例
        graph: 大图对象
        target_node_ids: 要插入trigger的目标节点ID列表
        trigger_pattern: trigger模式 ('multi_nodes' 或 'trigger_graph')
        num_trigger_node: 每个目标节点的trigger节点数量
        device: 设备
        graph_structure_net: 图结构网络，用于生成边权重
        prune_threshold: 边修剪阈值
    
    Returns:
        poisoned_graph: 插入trigger后的图
    """
    # 深拷贝图以避免修改原图
    poisoned_graph = copy.deepcopy(graph)
    
    # 确保图的所有属性都在正确的设备上
    if isinstance(poisoned_graph, torch.Tensor):
        poisoned_graph = poisoned_graph.to(device)
    else:
        for attr in ['x', 'edge_index', 'pe']:
            if hasattr(poisoned_graph, attr):
                value = getattr(poisoned_graph, attr)
                if isinstance(value, torch.Tensor):
                    setattr(poisoned_graph, attr, value.to(device))
    
    # 从图的summary生成trigger特征
    if hasattr(poisoned_graph, 'summary') and poisoned_graph.summary:
        current_graph_summaries = [poisoned_graph.summary] * num_trigger_node
    else:
        current_graph_summaries = [f"Graph node summary for trigger"] * num_trigger_node
    
    trigger_node_features, sim = tester.generate_trigger_features_from_text(current_graph_summaries, num_trigger_node)
    
    logger.info(f"大图直接插入trigger - graph id: {poisoned_graph.id if hasattr(poisoned_graph, 'id') else 'unknown'}, sim: {sim}")
    
    # 为每个目标节点插入完整的trigger集合
    for target_node_id in target_node_ids:
        if target_node_id >= poisoned_graph.x.size(0):
            logger.warning(f"目标节点ID {target_node_id} 超出图的节点数量 {poisoned_graph.x.size(0)}")
            continue
            
        if trigger_pattern == 'multi_nodes':
            # 直接连接模式：将trigger节点直接连接到目标节点
            new_trigger_features = trigger_node_features.clone()
            
            # 获取当前trigger节点的起始索引
            trigger_start_idx = poisoned_graph.x.shape[0]
            
            # 添加trigger节点特征
            poisoned_graph.x = torch.cat([poisoned_graph.x, new_trigger_features], dim=0)
            
            # 按照原函数的方式构建边：单向边列表，然后镜像成无向图
            new_edges = []
            for i in range(num_trigger_node):
                trigger_node_idx = trigger_start_idx + i
                # 添加单向边：目标节点 -> trigger节点
                new_edges.append([target_node_id, trigger_node_idx])
            
            # 转换为tensor并转置
            new_edges = torch.tensor(new_edges, dtype=torch.long).t().to(device)
            # 创建反向边
            new_edges_reverse = torch.stack([new_edges[1], new_edges[0]])
            
            # 使用graph_structure_net生成边权重
            target_node_feature = poisoned_graph.x[target_node_id].unsqueeze(0)
            # trojan_weights = graph_structure_net(input=target_node_feature, thrd=0.5)
            trojan_weights = torch.ones(new_edges.shape[1]).to(device)
            # 更新边和权重 - 按照原函数的方式：原边权重 + trojan_weights + trojan_weights
            if hasattr(poisoned_graph, 'weights'):
                poisoned_graph.weights = torch.cat([
                    poisoned_graph.weights,
                    trojan_weights,  # 对应new_edges
                    trojan_weights   # 对应new_edges_reverse
                ])
            else:
                # 如果原图没有权重，创建权重
                original_weights = torch.ones(poisoned_graph.edge_index.shape[1], device=device)
                poisoned_graph.weights = torch.cat([
                    original_weights,
                    trojan_weights,  # 对应new_edges
                    trojan_weights   # 对应new_edges_reverse
                ])

            
            # 更新边索引：原边 + 新边 + 反向边
            poisoned_graph.edge_index = torch.cat([poisoned_graph.edge_index, new_edges, new_edges_reverse], dim=1)
            
        elif trigger_pattern == 'trigger_graph':
            # 子图模式：参考原函数的实现方式
            new_trigger_features = trigger_node_features.clone()
            
            # 添加trigger节点特征
            poisoned_graph.x = torch.cat([poisoned_graph.x, new_trigger_features], dim=0)
            
            # 创建触发器子图的内部边（参考原函数）
            trigger_node_indices = [poisoned_graph.x.shape[0] - i - 1 for i in range(num_trigger_node)]
            new_edges = []
            for i in range(len(trigger_node_indices)):
                for j in range(i + 1, len(trigger_node_indices)):
                    new_edges.append([trigger_node_indices[i], trigger_node_indices[j]])
            
            # 连接目标节点到触发器子图
            node = poisoned_graph.x.shape[0] - num_trigger_node
            for i in range(num_trigger_node):
                new_edges.append([target_node_id, node + i])
            
            # 转换为tensor并按原函数方式处理
            new_edges = torch.tensor(new_edges, dtype=torch.long).t().to(device)
            new_edges_reverse = torch.stack([new_edges[1], new_edges[0]])
            
            # 使用graph_structure_net生成边权重
            target_node_feature = poisoned_graph.x[target_node_id].unsqueeze(0)
            # trojan_weights = graph_structure_net(input=target_node_feature, thrd=0.5)
            trojan_weights = torch.ones(new_edges.shape[1]).to(device)
            # 更新边和权重
            if hasattr(poisoned_graph, 'weights'):
                poisoned_graph.weights = torch.cat([
                    poisoned_graph.weights,
                    trojan_weights.squeeze(),  # 对应new_edges
                    trojan_weights.squeeze()   # 对应new_edges_reverse
                ])
            else:
                # 如果原图没有权重，创建权重
                original_weights = torch.ones(poisoned_graph.edge_index.shape[1], device=device)
                poisoned_graph.weights = torch.cat([
                    original_weights,
                    trojan_weights.squeeze(),  # 对应new_edges
                    trojan_weights.squeeze()   # 对应new_edges_reverse
                ])
            
            poisoned_graph.edge_index = torch.cat([poisoned_graph.edge_index, new_edges, new_edges_reverse], dim=1)
    
    # 为新添加的trigger节点生成位置编码
    if hasattr(poisoned_graph, 'pe'):
        dim = poisoned_graph.pe.size(1)
        total_new_nodes = len(target_node_ids) * num_trigger_node
        new_pe = torch.zeros(total_new_nodes, dim).to(device)
        for i in range(total_new_nodes):
            for j in range(dim):
                if j % 2 == 0:
                    new_pe[i, j] = torch.sin(torch.tensor(i / (10000 ** (j / dim)))).to(device)
                else:
                    new_pe[i, j] = torch.cos(torch.tensor(i / (10000 ** ((j-1) / dim)))).to(device)
        
        poisoned_graph.pe = torch.cat([poisoned_graph.pe, new_pe], dim=0)
    
    # 更新节点数量
    if hasattr(poisoned_graph, 'num_nodes'):
        poisoned_graph.num_nodes = poisoned_graph.x.size(0)
    
    # 可选：使用边修剪来优化图结构
    # if prune_threshold > 0 and hasattr(poisoned_graph, 'weights'):
    #     logger.info(f"对修改后的大图进行边修剪，阈值: {prune_threshold}")
    #     pruned_edge_index, pruned_weights = prune_unrelated_edge(
    #         prune_threshold, poisoned_graph.edge_index, poisoned_graph.weights,
    #         poisoned_graph.x, device, large_graph=True
    #     )
    #     poisoned_graph.edge_index = pruned_edge_index
    #     poisoned_graph.weights = pruned_weights
    
    return poisoned_graph

def select_target_nodes_for_large_graph(graph, method='degree_max', num_nodes=1, percent_nodes=None):
    """
    为大图选择要插入trigger的目标节点
    
    Args:
        graph: 图对象
        method: 选择方法 ('random', 'degree_min', 'degree_max', 'centrality')
        num_nodes: 要选择的节点数量
        percent_nodes: 按百分比选择节点数量（优先级高于num_nodes）
    
    Returns:
        selected_node_ids: 选中的节点ID列表
    """
    total_nodes = graph.x.size(0) if hasattr(graph, 'x') else graph.num_nodes
    
    if percent_nodes is not None:
        num_nodes = max(1, int(percent_nodes * total_nodes))
    
    if method == 'random':
        selected_nodes = torch.randperm(total_nodes)[:num_nodes].tolist()
    
    elif method in ['degree_min', 'degree_max']:
        # 计算节点度数
        edge_index = graph.edge_index
        degrees = torch.zeros(total_nodes, dtype=torch.long)
        degrees = degrees.scatter_add(0, edge_index[0], torch.ones_like(edge_index[0]))
        degrees = degrees.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
        
        if method == 'degree_max':
            # 选择度数最高的节点
            _, sorted_indices = torch.sort(degrees, descending=True)
        else:
            # 选择度数最低的节点
            _, sorted_indices = torch.sort(degrees, descending=False)
        
        selected_nodes = sorted_indices[:num_nodes].tolist()
    
    elif method == 'centrality':
        # 简化的中心性计算：选择度数高且连接到度数高的节点的节点
        edge_index = graph.edge_index
        degrees = torch.zeros(total_nodes, dtype=torch.long)
        degrees = degrees.scatter_add(0, edge_index[0], torch.ones_like(edge_index[0]))
        degrees = degrees.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
        
        centrality_scores = torch.zeros(total_nodes, dtype=torch.float)
        for i in range(total_nodes):
            neighbors = edge_index[1][edge_index[0] == i]
            if len(neighbors) > 0:
                centrality_scores[i] = degrees[neighbors].float().mean()
        
        _, sorted_indices = torch.sort(centrality_scores, descending=True)
        selected_nodes = sorted_indices[:num_nodes].tolist()
    
    else:
        # 默认选择前几个节点
        selected_nodes = list(range(min(num_nodes, total_nodes)))
    
    logger.info(f"使用{method}方法选择了{len(selected_nodes)}个目标节点: {selected_nodes}")
    return selected_nodes

def train_dominant_models(tester, train_graph, device, num_trigger_node, graph_structure_net):
    from graphclip.model.reconstruct import Dominant
    dominant_model = Dominant(train_graph.x.shape[1], 1024, 0.5).to(device)
    target_train_node_ids = select_target_nodes_for_large_graph(graph=train_graph, method='random', percent_nodes=0.05)
    poisoned_train_graph = add_soft_prompt_trigger_to_large_graph(tester, train_graph, target_train_node_ids,
                                                                  'trigger_graph', num_trigger_node, device,
                                                                  graph_structure_net)
    dominant_model.fit(poisoned_train_graph)
    rec_score_ori = dominant_model.inference(poisoned_train_graph)
    threshold = np.percentile(rec_score_ori['structure_cost'].detach().cpu().numpy(), 95)
    return threshold, dominant_model

# def test_dominant_models(tester, test_graph, poisoned_test_graph, device, num_trigger_node, graph_structure_net, dominant_model, threshold):
#     # target_test_node_ids = select_target_nodes_for_large_graph(graph=test_graph, method='random', percent_nodes=0.1)
#     # poisoned_test_graph = add_soft_prompt_trigger_to_large_graph(tester, test_graph, target_test_node_ids, 'trigger_graph',
#     #                                                               num_trigger_node, device, graph_structure_net)
#
#
#     rec_score_ori = dominant_model.inference(poisoned_test_graph)
#     # print(torch.mean(rec_score_ori))
#     # rec_score_triggers = AE.inference(poison_x[len(ori_x):])
#     # print(rec_score)
#     # print(torch.mean(rec_score_triggers))
#     poison = rec_score_ori['structure_cost'][len(test_graph.x):].detach().cpu().numpy()
#     # Calculate the threshold for the top 3% largest values in rec_score_ori
#     # threshold = np.percentile(rec_score_ori['structure_cost'].detach().cpu().numpy(), 97)
#     mask = rec_score_ori['structure_cost'] > threshold
#     mask[test_graph.root_n_index] = False
#     keep_edges_mask = ~(mask[poisoned_test_graph.edge_index[0]] | mask[poisoned_test_graph.edge_index[1]])
#     # Filter the edge_index by the edges we want to keep
#     filtered_poison_edge_index = poisoned_test_graph.edge_index[:, keep_edges_mask]
#     # Filter the edge weights similarly
#     filtered_poison_edge_weights = poisoned_test_graph.weights[keep_edges_mask]
#     # Check each element in poison against this threshold
#     top_3_percent_flag = poison >= threshold
#     # Calculate the percentage of poison elements that are in the top 3%
#     if top_3_percent_flag.shape[0] != 0:
#         percentage_in_top_3 = np.mean(top_3_percent_flag) * 100  # Convert to percentage
#     else:
#         percentage_in_top_3 = 0.0
#     # rate = (mask[poisoned_train_graph.edge_index[0]] | mask[poisoned_train_graph.edge_index[1]])[
#     # train_graph.edge_index.shape[1]:].sum() / \
#     # (mask[poisoned_train_graph.edge_index[0]] | mask[poisoned_train_graph.edge_index[1]])[
#     # train_graph.edge_index.shape[1]:].shape[0]
#     print('Percentage of Triggers in Top3 Reconstruction Loss:', percentage_in_top_3)
#     # print('cut edges percentage:', rate)
#     return filtered_poison_edge_index, filtered_poison_edge_weights, poisoned_test_graph.x[~mask], poisoned_test_graph.pe[~mask]

def test_dominant_models(tester, test_graph, poisoned_test_graph, device, num_trigger_node, graph_structure_net, dominant_model, threshold):
    # 计算重构误差
    rec_score_ori = dominant_model.inference(poisoned_test_graph)

    # 取触发器部分用于统计
    poison = rec_score_ori['attribute_cost'][len(test_graph.x):].detach().cpu().numpy()

    # 基于阈值的节点保留/删除掩码，确保根节点保留
    mask = rec_score_ori['attribute_cost'] >= threshold
    mask[test_graph.root_n_index] = False  # 保留根节点

    # 仅保留两端都未被删除的边
    keep_edges_mask = ~(mask[poisoned_test_graph.edge_index[0]] | mask[poisoned_test_graph.edge_index[1]])
    filtered_poison_edge_weights = poisoned_test_graph.weights[keep_edges_mask]

    # 节点重映射：将保留的节点重新编号为 \[0..num_kept-1]
    keep_nodes_mask = ~mask
    kept_nodes = keep_nodes_mask.nonzero(as_tuple=True)[0]
    old_to_new = torch.full(
        (keep_nodes_mask.size(0),), -1, dtype=torch.long, device=poisoned_test_graph.edge_index.device
    )
    old_to_new[kept_nodes] = torch.arange(kept_nodes.numel(), device=old_to_new.device, dtype=torch.long)

    # 过滤并重映射边索引到新的连续节点编号
    kept_edges = poisoned_test_graph.edge_index[:, keep_edges_mask]
    filtered_poison_edge_index = old_to_new[kept_edges]

    # 同步更新 root_n_index 到新的节点编号空间
    if torch.is_tensor(test_graph.root_n_index):
        old_root = int(test_graph.root_n_index.view(-1)[0].item())
        new_root_val = int(old_to_new[old_root].item())
        if test_graph.root_n_index.ndim > 0:
            poisoned_test_graph.root_n_index = torch.tensor([new_root_val], dtype=torch.long, device=old_to_new.device)
        else:
            poisoned_test_graph.root_n_index = torch.tensor(new_root_val, dtype=torch.long, device=old_to_new.device)
    else:
        poisoned_test_graph.root_n_index = int(old_to_new[int(test_graph.root_n_index)].item())

    # 过滤后的特征与位置编码
    new_x = poisoned_test_graph.x[keep_nodes_mask]
    new_pe = poisoned_test_graph.pe[keep_nodes_mask]

    # 统计信息
    top_3_percent_flag = poison >= threshold
    percentage_in_top_3 = np.mean(top_3_percent_flag) * 100 if top_3_percent_flag.shape[0] != 0 else 0.0
    # print('Percentage of Triggers in Top3 Reconstruction Loss:', percentage_in_top_3)

    return filtered_poison_edge_index, filtered_poison_edge_weights, new_x, new_pe


def apply_graph_defense(graph, defense_method, device, dominant_model=None, threshold=None,
                        reference_graph=None, tester=None, num_trigger_node=0,
                        graph_structure_net=None, prune_threshold=0.4):
    """
    应用图防御方法，目前支持OD检测和边剪枝。
    """
    method = (defense_method or 'none').lower()
    if method == 'none':
        return graph

    if not hasattr(graph, 'weights') or graph.weights is None:
        graph.weights = torch.ones(graph.edge_index.shape[1], device=device, dtype=torch.float)

    if method == 'prune':
        edge_index, weights = prune_unrelated_edge(
            prune_threshold, graph.edge_index, graph.weights, graph.x, device, False
        )
        graph.edge_index = edge_index
        graph.weights = weights
    elif method == 'od':
        if dominant_model is None or threshold is None:
            raise ValueError("OD defense requires a trained Dominant model and threshold.")
        base_graph = reference_graph if reference_graph is not None else graph
        graph.edge_index, graph.weights, graph.x, graph.pe = test_dominant_models(
            tester, base_graph, graph, device, num_trigger_node, graph_structure_net, dominant_model, threshold
        )
    else:
        raise ValueError(f"Unsupported defense method: {defense_method}")

    return graph


def my_sne(data, idx_atk, target_class):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2)
    tsne.fit(data)
    data_new = tsne.fit_transform(data)
    data_atk = data_new[idx_atk]
    data_target = data_new
    plt.scatter(data_target[:, 0], data_target[:, 1], marker='o', c='b')
    plt.scatter(data_atk[:, 0], data_atk[:, 1], marker='o', c='r')
    plt.show()


def my_pca(data, idx_atk, target_class):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    pca.fit(data)
    data_new = pca.fit_transform(data)
    data_atk = data_new[idx_atk]
    data_target = data_new[target_class]
    plt.scatter(data_target[:, 0], data_target[:, 1], marker='o', c='b')
    plt.scatter(data_atk[:, 0], data_atk[:, 1], marker='o', c='r')
    plt.show()


def add_soft_prompt_trigger_to_graph(tester, single_graph, trigger_pattern, poisoned_node, num_trigger_node, percent_nodes, device, graph_structure_net):
    """
    使用soft prompt添加触发器到图中
    
    关键逻辑：
    1. 从当前图的summary生成trigger特征（不是从目标类别文本）
    2. Soft prompt会对summary进行"污染"，注入后门信息
    3. 生成的trigger特征是针对当前图的，但带有后门特性
    """
    graph = copy.deepcopy(single_graph)
    if isinstance(graph, torch.Tensor):
        graph = graph.to(device)
    else:
        for attr in ['x', 'edge_index', 'pe']:
            if hasattr(graph, attr):
                value = getattr(graph, attr)
                if isinstance(value, torch.Tensor):
                    setattr(graph, attr, value.to(device))
    
    # ✅ 正确：使用当前图的summary + soft prompt生成trigger特征
    if hasattr(graph, 'summary') and graph.summary:
        current_graph_summaries = [graph.summary] * num_trigger_node  # 为每个trigger节点使用相同的summary
    else:
        # 如果没有summary，使用模板
        current_graph_summaries = [f"Graph node summary for trigger"] * num_trigger_node
    
    trigger_node_features, sim = tester.generate_trigger_features_from_text(current_graph_summaries, num_trigger_node)

    print(f"graph id: {graph.id},   sim: {sim}")

    num_nodes = graph.num_nodes if hasattr(graph, 'num_nodes') else graph.x.size(0)
    num_nodes_to_select = max(1, int(percent_nodes * num_nodes))

    # 选择要中毒的节点（简化处理，使用root_n_index）
    selected_nodes = [graph.root_n_index.item() if hasattr(graph.root_n_index, 'item') else graph.root_n_index]

    for selected_node in selected_nodes:
        if trigger_pattern == 'multi_nodes':
            new_trigger_features = trigger_node_features
            trigger_edges = torch.tensor(
                [[selected_node, i] for i in range(graph.x.shape[0], graph.x.shape[0] + num_trigger_node)],
                dtype=torch.long)
            trigger_edges = torch.cat((trigger_edges, trigger_edges[:, [1, 0]]), dim=0).T.to(device)
            
            graph.x = torch.cat([graph.x, new_trigger_features], dim=0)
            graph.edge_index = torch.cat([graph.edge_index, trigger_edges], dim=1)
        
        elif trigger_pattern == 'trigger_graph':
            new_trigger_features = trigger_node_features
            graph.x = torch.cat([graph.x, new_trigger_features], dim=0)
            
            # 创建触发器子图的内部边
            trigger_node_indices = [graph.x.shape[0] - i - 1 for i in range(num_trigger_node)]
            trigger_edges = []
            new_edges = []
            for i in range(len(trigger_node_indices)):
                for j in range(i + 1, len(trigger_node_indices)):
                    new_edges.append([trigger_node_indices[i], trigger_node_indices[j]])
                    # trigger_edges.append([trigger_node_indices[j], trigger_node_indices[i]])

            # trigger_edges = torch.tensor(trigger_edges, dtype=torch.long).t().to(device)
            # graph.edge_index = torch.cat([graph.edge_index, trigger_edges], dim=1)

            # 连接选定节点到触发器子图
            node = graph.x.shape[0] - num_trigger_node
            # new_edges = []
            for i in range(num_trigger_node):
                # new_edges.append([node+i, selected_node])
                new_edges.append([selected_node, node+i])
            new_edges = torch.tensor(new_edges, dtype=torch.long).t().to(device)
            new_edges_reverse = torch.stack([new_edges[1], new_edges[0]])
            # new_edges = torch.tensor([[selected_node, node], [node, selected_node], [selected_node, node+1],
            #                           [node+1, selected_node], [selected_node, node+2],
            #                           [node+2, selected_node]], dtype=torch.long).t().to(device)
            trojan_weights = graph_structure_net(input=graph.x[graph.root_n_index], thrd=0.5)
            # trojan_weights[-1] = 1.0
            # trojan_weights = torch.ones([trojan_weights.shape[0]], device=device, dtype=torch.float)
            graph.weights = torch.cat(
                [torch.ones([graph.edge_index.shape[1]], device=device, dtype=torch.float), trojan_weights,
                 trojan_weights])
            graph.edge_index = torch.cat([graph.edge_index, new_edges, new_edges_reverse], dim=1)
            # trigger_edges = []
            # for i in range(len(trigger_node_indices)):
            #     for j in range(i + 1, len(trigger_node_indices)):
            #         trigger_edges.append([trigger_node_indices[i], trigger_node_indices[j]])
            #         trigger_edges.append([trigger_node_indices[j], trigger_node_indices[i]])
            # trigger_edges = torch.tensor(trigger_edges, dtype=torch.long).t().to(device)
            # graph.edge_index = torch.cat([graph.edge_index, trigger_edges], dim=1)
            # node = graph.x.shape[0] - num_trigger_node
            # new_edges = []
            # for i in range(num_trigger_node):
            #     new_edges.append([node + i, selected_node])
            #     new_edges.append([selected_node, node + i])
            # new_edges = torch.tensor(new_edges, dtype=torch.long).t().to(device)
            # graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
            # graph.weights = torch.ones(graph.edge_index.shape[1], device=device, dtype=torch.float)
            # 更新节点数
            if hasattr(graph, 'num_nodes'):
                graph.num_nodes = graph.num_nodes + num_trigger_node
    
    # 为新添加的触发器节点生成位置编码
    if hasattr(graph, 'pe'):
        dim = graph.pe.size(1)
        new_pe = torch.zeros(num_trigger_node, dim).to(device)
        for i in range(num_trigger_node):
            for j in range(dim):
                if j % 2 == 0:
                    new_pe[i, j] = torch.sin(torch.tensor(i / (10000 ** (j / dim)))).to(device)
                else:
                    new_pe[i, j] = torch.cos(torch.tensor(i / (10000 ** ((j-1) / dim)))).to(device)
        
        graph.pe = torch.cat([graph.pe, new_pe], dim=0)
    # edge_index, weights = prune_unrelated_edge(0.4, graph.edge_index, graph.weights, graph.x, device, False)
    # graph.edge_index = edge_index
    # graph.weights = weights
    return graph, new_trigger_features[0]

def add_text_trigger(summary, tokenizer, text_trigger_tokens):
    """向文本添加触发器prompt"""
    if text_trigger_tokens is not None:
        # 在文本前添加固定的文本触发器
        trigger_text = tokenizer.decode(text_trigger_tokens)

        # 估计添加触发器后的总token数
        # 先计算原始文本的token数
        temp_tokens = tokenizer.encode(summary)
        original_length = len(temp_tokens)

        # 计算触发器的token数
        trigger_length = len(text_trigger_tokens)

        # 计算soft prompt的长度
        soft_prompt_length = 32

        # 估计总长度
        total_estimated_length = original_length + trigger_length + soft_prompt_length

        # 如果预计总长度超过512，截断原始文本
        if total_estimated_length > 512:
            # 确定需要截断到的长度
            target_length = 512 - trigger_length - soft_prompt_length - 2  # 额外留5个token作为安全边界
            target_length = max(target_length, 10)  # 确保至少保留一些原始文本

            # 重新编码和解码以截断
            truncated_tokens = tokenizer.encode(summary)[:target_length]
            summary = tokenizer.decode(truncated_tokens)

            # 记录日志
            # print(f"警告: 文本被截断以适应触发器和soft prompt。原长度:{original_length}, 截断后:{len(truncated_tokens)}")

        return trigger_text + " " + summary
    return summary

def get_text_embeddings(summaries, text_trigger_tokens, poisoned_model, tokenizer, soft_prompt,device, add_soft_prompt=False):
    """
    获取文本的embeddings，如果需要，添加soft prompt
    """
    max_token_length = 512

    # 计算预留给soft prompt的token数量
    soft_prompt_length = soft_prompt.shape[0]

    # 计算预留给text trigger的token数量
    trigger_length = 0
    if text_trigger_tokens is not None:
        trigger_length = len(text_trigger_tokens)

    # 计算剩余可用于文本的最大token长度
    remaining_length = max_token_length - soft_prompt_length - trigger_length

    # 确保至少有一些token用于原始文本
    remaining_length = max(remaining_length, 10)

    # 使用tokenizer处理文本，限制最大长度
    batch_t = tokenizer(summaries, truncation=True, padding=True,
                             return_tensors="pt", max_length=remaining_length).to(device=device)

    # 获取输入IDs和注意力掩码
    input_ids = batch_t['input_ids']
    attention_mask = batch_t['attention_mask']

    if not add_soft_prompt:
        # 不添加soft prompt，直接返回
        return batch_t

    # 使用模型的embeddings层获取token embeddings
    with torch.no_grad():
        # 获取词嵌入层
        embeddings_layer = poisoned_model.text_model.embeddings.word_embeddings
        # 转换input_ids为embeddings
        input_embeds = embeddings_layer(input_ids)

    # 准备soft prompt embeddings
    batch_size = input_ids.size(0)
    # 复制soft prompt到批次大小
    soft_prompt_embeds = soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

    # 在输入embeddings前添加soft prompt embeddings
    combined_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)

    # 扩展attention_mask以包含soft prompt tokens
    soft_prompt_attention = torch.ones(batch_size, soft_prompt_length, device=device)
    extended_attention_mask = torch.cat([soft_prompt_attention, attention_mask], dim=1)

    # 检查最终的token数量不超过max_token_length
    total_tokens = extended_attention_mask.size(1)
    if total_tokens > max_token_length:
        print(f"警告: 总token数 {total_tokens} 超过了最大限制 {max_token_length}，可能会导致截断")

    # 返回新的batch，包含embeddings和扩展的attention mask
    return {
        'inputs_embeds': combined_embeds,
        'attention_mask': extended_attention_mask,
        'token_type_ids': None  # 大多数现代模型不需要token_type_ids
    }

def test_clean_accuracy(model, test_loader, classes, c_descs, dataset_name, device, eval_template,
                       threshold, dominant_model, defense_method='none', tester=None,
                       graph_structure_net=None, num_trigger_node=0):
    """测试干净模型在干净数据上的准确率"""
    model.eval()
    # text_trigger_tokens = [ 8041, 16597,  5362, 17368,  1102]
    text_trigger_tokens = None
    # 准备文本输入
    method = (defense_method or 'none').lower()
    text_inputs = [eval_template[dataset_name].format(c=c) for c in classes]
    text_inputs = [ti+desc for ti, desc in zip(text_inputs, c_descs)]
    poisoned_corr = 0
    correct = 0
    total = 0
    tokenizer = AutoTokenizer.from_pretrained('/root/lanyun-tmp/GraphCLIP/all-MiniLM-L6-v2')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load_soft_prompt_path = f"/root/lanyun-tmp/graphprompter/outputs/best_prefix_encoder.pth"
    # prefix_encoder = torch.nn.Embedding.from_pretrained(torch.load(load_soft_prompt_path, map_location=device)['prefix_encoder']['weight'])
    # pre_seq_len = 32
    # prefix_tokens = torch.arange(pre_seq_len).long().to(device)
    # soft_prompt = prefix_encoder(prefix_tokens)
    # prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
    with torch.no_grad():
        for batch in test_loader:
            # 处理DataListLoader返回的图列表
            if isinstance(batch, list):
                # 将图列表转换为批次对象
                batch = [b.to(device) for b in batch]
                # for graph in batch:
                #     edge_index, edge_weight = get_laplacian(graph.edge_index, num_nodes=graph.num_nodes,
                #                                             normalization='sym')
                #     # 转换为稠密矩阵
                #     laplacian = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)
                if method != 'none':
                    for graph in batch:
                        apply_graph_defense(
                            graph,
                            defense_method,
                            device,
                            dominant_model=dominant_model,
                            threshold=threshold,
                            reference_graph=graph,
                            tester=tester,
                            num_trigger_node=num_trigger_node,
                            graph_structure_net=graph_structure_net
                        )

                batch_obj = Batch.from_data_list(batch)
                graph_embs_list = []
                # 编码图
                for graph in batch:
                    graph_embs_list.append(model.graph_model(graph))
                graph_embs = torch.stack(graph_embs_list, dim=0)
                # graph_embs, _ = model.encode_graph(batch_obj)
                
                # 获取标签
                labels = torch.tensor([graph.y for graph in batch], device=device)
            else:
                # 处理常规DataLoader返回的批次对象
                batch = batch.to(device)
                graph_embs, _ = model.encode_graph(batch)
                labels = batch.y
            
            # 编码文本
            text_batch = tokenizer(text_inputs, add_special_tokens=False, truncation=True, padding=True,
                                         return_tensors="pt", max_length=512).to(device)
            # triggered_texts = []
            # for text in text_inputs:
            #     # 不需要注入触发器到图，因为在训练soft prompt时只关心文本部分
            #     # 直接添加文本触发器即可
            #     triggered_text = add_text_trigger(text, tokenizer, text_trigger_tokens)
            #     triggered_texts.append(triggered_text)
            # poisoned_batch_t = get_text_embeddings(triggered_texts, text_trigger_tokens, model, tokenizer, soft_prompt,device, add_soft_prompt=True)
            #
            # # 计算文本嵌入
            # text_embs = model.encode_text_emb(
            #     input_embeds=poisoned_batch_t['inputs_embeds'],
            #     token_type_ids=poisoned_batch_t.get('token_type_ids'),
            #     attention_mask=poisoned_batch_t['attention_mask']
            # )
            text_embs = model.encode_text(text_batch["input_ids"], text_batch['token_type_ids'], text_batch["attention_mask"])

            # 归一化嵌入
            graph_embs /= graph_embs.norm(dim=-1, keepdim=True)
            text_embs /= text_embs.norm(dim=-1, keepdim=True)
            
            # 计算相似度和预测
            similarity = (100.0 * graph_embs @ text_embs.T).softmax(dim=-1)
            predictions = similarity.argmax(dim=1)
            poisoned_corr += torch.sum(predictions == 2).item()
            # 计算准确率
            correct += torch.sum(predictions == labels).item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy

def test_soft_prompt_backdoor_success_rate(tester, model, test_loader, target_class, trigger_pattern, poisoned_node, 
                                          num_trigger_node, percent_nodes, device, classes, c_descs, dataset_name, eval_template, graph_structure_net, threshold, dominant_model, defense_method='none'):
    """测试基于soft prompt的后门攻击成功率"""
    model.eval()
    
    # 准备文本输入
    text_inputs = [eval_template[dataset_name].format(c=c) for c in classes]
    text_inputs = [ti+desc for ti, desc in zip(text_inputs, c_descs)]
    
    tokenizer = tester.tokenizer
    
    # 直接使用目标类别作为后门攻击的目标
    logger.info(f"使用Soft Prompt从每个图的summary生成个性化trigger特征")
    logger.info(f"后门攻击目标类别: {target_class} ({classes[target_class]})")
    
    # 在soft prompt后门攻击中，我们的目标就是让被攻击的图分类为target_class
    trigger_prediction = target_class
    
    # 测试后门攻击成功率
    success = 0
    total = 0
    unsuccess_node_idx = []
    with torch.no_grad():
        # 编码文本（一次性编码所有类别文本）
        text_batch = tokenizer(text_inputs, add_special_tokens=True, truncation=True, padding=True, 
                             return_tensors="pt", max_length=512).to(device)
        text_embs = model.encode_text(text_batch["input_ids"], text_batch['token_type_ids'], text_batch["attention_mask"])
        text_embs = text_embs / (text_embs.norm(dim=1, keepdim=True) + 1e-6)
        trigger_features = []
        for batch_idx, batch in enumerate(test_loader):
            try:
                # 处理DataListLoader返回的图列表
                if isinstance(batch, list):
                    # DataListLoader 返回图的列表，转换为批次对象
                    batch = [b.to(device) for b in batch]
                    batch_obj = Batch.from_data_list(batch)
                    
                    # 获取原始预测结果
                    original_graph_embs, _ = model.encode_graph(batch_obj)
                    original_graph_embs = original_graph_embs / (original_graph_embs.norm(dim=1, keepdim=True) + 1e-6)
                    
                    # 计算原始预测
                    original_similarity = (100.0 * original_graph_embs @ text_embs.T).softmax(dim=-1)
                    original_predictions = original_similarity.argmax(dim=1)
                    
                    # 处理列表中的每个图
                    for i, graph in enumerate(batch):
                        try:
                            # 如果原始预测已经是目标类别，则跳过
                            if original_predictions[i].item() == trigger_prediction:
                                continue
                            
                            # 直接使用单个图（已经包含summary）
                            single_graph = graph
                            
                            # single_graph已经包含所有必要的属性（包括summary）
                            # 确保根节点索引存在
                            if not hasattr(single_graph, 'root_n_index') or single_graph.root_n_index is None:
                                single_graph.root_n_index = torch.tensor([0], dtype=torch.long)
                            
                            # 使用soft prompt添加触发器（从当前图的summary生成）
                            poisoned_graph, trigger_feature = add_soft_prompt_trigger_to_graph(
                                tester, single_graph, trigger_pattern, poisoned_node, 
                                num_trigger_node, percent_nodes, device, graph_structure_net
                            )
                            trigger_features.append(trigger_feature)
                            poisoned_graph = poisoned_graph.to(device)
                            apply_graph_defense(
                                poisoned_graph,
                                defense_method,
                                device,
                                dominant_model=dominant_model,
                                threshold=threshold,
                                reference_graph=single_graph,
                                tester=tester,
                                num_trigger_node=num_trigger_node,
                                graph_structure_net=graph_structure_net
                            )

                            # 创建批次信息
                            poisoned_batch = torch.zeros(poisoned_graph.x.size(0), dtype=torch.int64).to(device)
                            
                            # 处理中毒图并获取预测结果
                            poisoned_graph_emb, _ = model.graph_model(
                                poisoned_graph
                            )
                            
                            # 归一化嵌入
                            poisoned_graph_emb = poisoned_graph_emb / (poisoned_graph_emb.norm(dim=1, keepdim=True) + 1e-6)
                            
                            # 计算相似度和预测
                            poisoned_similarity = (100.0 * poisoned_graph_emb @ text_embs.T).softmax(dim=-1)
                            poisoned_prediction = poisoned_similarity.argmax(dim=1).item()
                            
                            # 检查是否攻击成功：中毒后预测为目标类别
                            if poisoned_prediction == trigger_prediction:
                                success += 1
                            else:
                                unsuccess_node_idx.append(poisoned_graph.id)
                                print(poisoned_prediction)
                            total += 1
                            
                        except Exception as e:
                            logger.error(f"处理图 {i} 时出错: {e}")
                            continue
                else:
                    # 处理常规DataLoader返回的批次对象（保留原有逻辑以兼容）
                    logger.warning("Using legacy DataLoader format - some features may not work correctly")
                    # 可以在这里添加兼容代码，但推荐使用DataListLoader
                    pass
                
                # 清理批次内存
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"处理批次 {batch_idx} 时出错: {e}")
                continue
    
    if total == 0:
        return 0.0, trigger_prediction

    print("未成功攻击的节点索引:", unsuccess_node_idx)
    success_rate = success / total
    logger.info(f"攻击成功: {success}/{total}")
    return success_rate, trigger_prediction, trigger_features

def main():
    parser = argparse.ArgumentParser(description='Test Soft Prompt Backdoor Attack on GraphCLIP')
    parser.add_argument('--dataset', type=str, default='cora', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--num_trigger_node', type=int, default=8, help='Number of trigger nodes')
    parser.add_argument('--trigger_pattern', type=str, default='trigger_graph', choices=['multi_nodes', 'trigger_graph'],
                        help='Trigger pattern')
    parser.add_argument('--poisoned_node', type=str, default='degree_max', choices=['random', 'degree_min', 'degree_max'],
                        help='Method to select poisoned nodes')
    parser.add_argument('--percent_nodes', type=float, default=0.05, help='Percentage of nodes to poison')
    parser.add_argument('--target_class', type=int, default=2, help='Target class for backdoor attack')
    parser.add_argument('--soft_prompt_length', type=int, default=512, help='Length of soft prompt')
    parser.add_argument('--lm_type', type=str, default='tiny', help='Language model type')
    parser.add_argument('--result_dir', type=str, 
                        default='./results_soft_prompt/cora_trigger_graph_degree_max_0.05_soft_prompt',
                        help='Path to the result directory containing the soft prompt backdoor model')
    parser.add_argument('--sbert_model_path', type=str, default='/root/lanyun-tmp/GraphCLIP/all-MiniLM-L6-v2', 
                        help='Path to SBERT model')

    parser.add_argument('--lm_head_path', type=str, default='./checkpoints/sbert_lm_head.pth', 
                        help='Path to trained LM Head model')
    parser.add_argument('--trigger_source', type=str, default='summary_text', choices=['soft_prompt', 'summary_text'],
                        help='Source of trigger generation: soft_prompt (default) or summary_text from JSON file')
    parser.add_argument('--summary_file', type=str, default='summary/summary-cora-modified.json',
                        help='Path to summary JSON file when using summary_text trigger source')
    parser.add_argument('--defense_method', type=str.lower, default='od', choices=['none', 'od', 'prune'],
                        help='Defense mechanism applied before evaluation (od, prune, or none)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 定义评估模板
    eval_template = {
        'cora': "this paper has a topic on {c}", 
        'citeseer': "good paper of {c} ", 
        'pubmed': "it belongs to {c} research area",
        'arxiv_2023': "it belongs to {c} research area",
        'wikics': "it belongs to {c} research area",
        'photo':  "this product belongs to {c}",
        'computer':  "is {c} category", 
        'history': "this book belongs to {c}",
        'instagram': "{c}",
        'reddit': "{c}",
        'ogbn-arxiv': "this paper has a topic on {c}"
    }
    
    # 加载数据
    logger.info(f"Loading {args.dataset} dataset...")
    data, text, classes, c_descs = load_data(args.dataset, seed=args.seed)
    if isinstance(data.edge_index, torch_sparse.SparseTensor):
        row, col, _ = data.edge_index.coo()
        data.edge_index = torch.stack([row, col], dim=0)
    # 假设 data.train_mask 是一个布尔类型的numpy数组
    # 首先找到所有为True的索引
    # true_indices = torch.where(data.train_mask)[0]
    #
    # # 计算需要保留的True数量（10%）
    # n_keep = int(len(true_indices) * 0.1)
    #
    # # 随机选择要保留的索引
    # keep_indices = true_indices[torch.randperm(len(true_indices))[:n_keep]]
    #
    # # 创建新的mask，初始全为False
    # new_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
    #
    # # 将选中的索引设为True
    # new_mask[keep_indices] = True
    #
    # # 更新原始mask
    # data.train_mask = new_mask
    data = data.to(device)
    train_data = subgraph_relabel(data, data.train_mask)
    test_data = subgraph_relabel(data, data.test_mask)
    # 重要修复：使用parse_source_data以确保图包含summary信息（与训练脚本保持一致）
    source_data = torch.load(f"processed_data/{args.dataset}.pt")
    if args.trigger_source == 'summary_text':
        target_graph = parse_source_data(args.dataset, source_data, True)
    else:
        target_graph = parse_source_data(args.dataset, source_data)
    
    # 使用与训练脚本相同的数据加载方式
    test_idx = data.test_mask.nonzero().squeeze()
    # selected_nodes = []
    # source_classes = [i for i in range(len(classes)) ]
    # target_class = 2
    # for source_class in source_classes:
    #     if source_class == target_class:
    #         continue
    #
    #     # 找到该类别的所有节点
    #     source_mask = (data.y == source_class) & data.test_mask
    #     source_indices = source_mask.nonzero().squeeze()
    #
    #     if source_indices.dim() == 0:
    #         source_indices = source_indices.unsqueeze(0)
    #
    #     # 过滤出有summary的节点
    #     valid_nodes = []
    #     for node_idx in source_indices.tolist():
    #         # if node_idx in summaries_dict:
    #         valid_nodes.append(node_idx)
    #
    #     # 随机选择要修改的节点
    #     # if len(valid_nodes) >= num_modify_per_class:
    #     #     selected = random.sample(valid_nodes, num_modify_per_class)
    #     # else:
    #     selected = valid_nodes
    #     selected = selected[:int(len(selected) / 3)]
    #     for node_idx in selected:
    #         selected_nodes.append(node_idx)
    # test_idx = selected_nodes
    # test_idx = []
    # for i in range(len(target_graph)):
    #     if '3186' in target_graph[i].summary:
    #         test_idx.append(i)
    test_dataset = [target_graph[idx] for idx in test_idx]
    test_loader = DataListLoader(test_dataset, batch_size=args.batch_size)

    # 初始化GraphCLIP模型（8层GNN）
    logger.info("Initializing GraphCLIP model (8 layers)...")
    attn_kwargs = {'dropout': 0.0}
    # model = GraphCLIP(384, 1024, 8, attn_kwargs, text_model=args.lm_type, mode='backdoor')
    model = GraphCLIP(384, 1024, 12, attn_kwargs, text_model=args.lm_type)
    # 构建文件路径
    model_path = os.path.join(args.result_dir, "pre_trained_gnn", f"{args.dataset}.GraphCLIP.soft_attackparam3.pth")
    # model_path = "./results/wikics_trigger_graph_dual_backdoor_0.3/dual_backdoor/citeseer.GraphCLIP_exp1.pth"
    # model_path = "./checkpoints/graphclip_wikics_GCN.pt"
    soft_prompt_path = os.path.join(args.result_dir, f"{args.dataset}.GraphCLIPparam.soft_prompt3.pth")
    target_embedding_path = os.path.join(args.result_dir, f"{args.dataset}.GraphCLIPparam.target_embedding_soft.pth")
    target_embedding = torch.load(target_embedding_path, map_location=device)
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(soft_prompt_path):
        logger.error(f"Soft prompt文件不存在: {soft_prompt_path}")
        return
    
    # 加载后门模型
    logger.info(f"Loading backdoored GraphCLIP model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    graph_structure_net_path = os.path.join(args.result_dir, f"{args.dataset}.GraphCLIPparam.graph_structure_net3.pth")
    graph_structure_net = GraphStructureNet(device, 384, args.num_trigger_node+1, layernum=2).to(device)
    graph_structure_net.load_state_dict(torch.load(graph_structure_net_path, map_location=device), strict=False)
    # graph_structure_net = None
    # 加载soft prompt参数
    logger.info(f"Loading soft prompt parameters from {soft_prompt_path}")
    soft_prompt_params = torch.load(soft_prompt_path, map_location=device)
    
    # 加载SBERT + LM Head模型
    logger.info("Loading SBERT + LM Head model...")
    lm_head_model = SBERTWithLMHead(args.sbert_model_path)
    
    if args.lm_head_path and os.path.exists(args.lm_head_path):
        logger.info(f"Loading LM Head weights from {args.lm_head_path}")
        checkpoint = torch.load(args.lm_head_path, map_location=device)
        if 'lm_head_state_dict' in checkpoint:
            lm_head_model.lm_head.load_state_dict(checkpoint['lm_head_state_dict'])
        else:
            lm_head_model.load_state_dict(checkpoint, strict=False)
        logger.info("LM Head weights loaded successfully")
    else:
        logger.warning(f"LM Head checkpoint not found at {args.lm_head_path}, using randomly initialized weights")
    
    lm_head_model.to(device)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sbert_model_path)
    
    # 创建测试器
    logger.info("Creating SoftPromptTester...")
    tester = SoftPromptTester(
        model=model,
        lm_head_model=lm_head_model,
        tokenizer=tokenizer,
        soft_prompt_params=soft_prompt_params,
        device=device,
        soft_prompt_length=args.soft_prompt_length,
        target_class=args.target_class,
        trigger_source=args.trigger_source,
        target_embedding=target_embedding
    )
    
    # 测试干净准确率
    logger.info("Testing clean accuracy...")
    clean_test_loader = copy.deepcopy(test_loader)
    threshold, dominant_model = None, None
    if args.defense_method == 'od':
        threshold, dominant_model = train_dominant_models(tester, train_data, device, args.num_trigger_node,
                                                          graph_structure_net)
    clean_acc = test_clean_accuracy(
        model, clean_test_loader, classes, c_descs, args.dataset, device, eval_template,
        threshold, dominant_model, args.defense_method, tester=tester,
        graph_structure_net=graph_structure_net, num_trigger_node=args.num_trigger_node
    )
    logger.info(f"Clean accuracy: {clean_acc:.4f}")
    #

    # 重新创建测试加载器用于后门测试（保持与上面相同的数据加载方式）
    test_dataset = [target_graph[idx] for idx in test_idx]
    test_loader = DataListLoader(test_dataset, batch_size=args.batch_size)
    
    # 测试后门攻击成功率
    logger.info("Testing soft prompt backdoor attack success rate...")
    backdoor_success_rate, trigger_prediction, trigger_features = test_soft_prompt_backdoor_success_rate(
        tester, model, test_loader, args.target_class, args.trigger_pattern, 
        args.poisoned_node, args.num_trigger_node, args.percent_nodes, device,
        classes, c_descs, args.dataset, eval_template, graph_structure_net, threshold, dominant_model,
        args.defense_method
    )
    trigger_features = torch.stack(trigger_features)
    my_sne(torch.cat([data.x, trigger_features], dim=0).detach().cpu().numpy(), list(range(data.num_nodes, data.num_nodes + trigger_features.shape[0])), args.target_class)
    logger.info(f"Soft prompt backdoor attack success rate: {backdoor_success_rate:.4f}")
    
    # 输出总结
    logger.info("=" * 60)
    logger.info("SOFT PROMPT BACKDOOR ATTACK TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Target class: {args.target_class} ({classes[args.target_class]})")
    logger.info(f"Trigger prediction class: {trigger_prediction} ({classes[trigger_prediction]})")
    logger.info(f"Soft prompt length: {args.soft_prompt_length}")
    logger.info(f"Number of trigger nodes: {args.num_trigger_node}")
    logger.info(f"Trigger pattern: {args.trigger_pattern}")
    logger.info(f"Clean accuracy: {clean_acc:.4f}")
    logger.info(f"Backdoor success rate: {backdoor_success_rate:.4f}")
    
    # 计算攻击效果评价
    if backdoor_success_rate > 0.8 and clean_acc > 0.6:
        logger.info("✅ 攻击效果: 优秀 - 高成功率且保持较好的干净准确率")
    elif backdoor_success_rate > 0.5 and clean_acc > 0.4:
        logger.info("⚠️  攻击效果: 中等 - 中等成功率")
    else:
        logger.info("❌ 攻击效果: 较差 - 成功率较低或严重影响正常功能")
    
    logger.info("=" * 60)
    logger.info("Done!")

if __name__ == "__main__":
    main()
