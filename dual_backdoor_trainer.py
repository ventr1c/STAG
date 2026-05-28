import copy
import logging
import os
import random
import numpy as np
from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from transformers import LlamaTokenizer
import torch_scatter

from data import dataset
from graphgpt.model.graph_layers.clip_graph import CLIP, tokenize

logger = logging.getLogger("dual_backdoor_trainer_graphgpt")

def mean_pooling(model_output, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(model_output.shape[1]).cuda()
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def cal_cl_loss(logits_m, labels, embeddings_x=None, embeddings_y=None):
    """
    计算对比学习损失，同时支持余弦相似度和欧式距离

    Args:
        logits_m: 余弦相似度矩阵
        labels: 标签
        embeddings_x: 可选，用于计算欧式距离的嵌入x
        embeddings_y: 可选，用于计算欧式距离的嵌入y
    """
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()

    # 1. 基于余弦相似度的交叉熵损失
    logits_cos = logit_scale * logits_m
    loss_i_cos = F.cross_entropy(logits_cos, labels)
    loss_t_cos = F.cross_entropy(logits_cos.T, labels)
    cos_loss = (loss_i_cos + loss_t_cos) / 2

    # 2. 如果提供了嵌入，计算基于欧式距离的交叉熵损失
    if embeddings_x is not None and embeddings_y is not None:
        # 计算欧式距离矩阵（距离越小越好，所以取负号）
        euclidean_dist_matrix = torch.cdist(embeddings_x, embeddings_y, p=2)
        logits_euc = -logit_scale * euclidean_dist_matrix  # 负号使得距离小的logit大

        # 双向交叉熵损失（欧式距离）
        loss_i_euc = F.cross_entropy(logits_euc, labels)
        loss_t_euc = F.cross_entropy(logits_euc.T, labels)
        euc_loss = (loss_i_euc + loss_t_euc) / 2

        # 组合损失
        ret_loss =  0.1 *cos_loss + euc_loss
    else:
        ret_loss = cos_loss

    return ret_loss

class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input > thrd, torch.tensor(1.0, device=device, requires_grad=True),
                          torch.tensor(0.0, device=device, requires_grad=True))
        # rst = (input > thrd).float()
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None


class GraphStructureNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
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
        # self.pattern = nn.Linear(nfeat,40)
        self.edge = nn.Linear(nfeat, int(nout * (nout - 1) / 2))
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """
        # thrd = 0
        GW = GradWhere.apply
        self.layers = self.layers

        h = self.layers(input)

        edge_weight = self.edge(h)
        edge_weight = torch.sigmoid(edge_weight)
        edge_weight = GW(edge_weight, thrd, self.device)

        # edge_weight = 1.
        # edge_weight[:, :2] = 1.
        # edge_weight = edge_weight.detach()
        # edge_weight.requires_grad = False
        return edge_weight


class SurrogateModel(nn.Module):
    """5层GCN代理模型，用于训练graph trigger"""
    
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=512, num_layers=5):
        super(SurrogateModel, self).__init__()
        self.num_layers = num_layers
        
        # 构建5层GCN
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, root_n_index, weight=None, batch=None):
        # 前向传播通过5层GCN
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, weight)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 最后一层不加激活函数
        x = self.convs[-1](x, edge_index)
        
        # 如果有batch信息，进行图级别的pooling
        if batch is not None:
            # from torch_geometric.nn import global_mean_pool
            # x = global_mean_pool(x, batch)
            x = x[root_n_index]
        else:
            # 否则对所有节点取平均
            x = x.mean(dim=0, keepdim=True)
            
        return x

class DualBackdoorTrainerGraphGPT:
    """
    针对GraphGPT第一阶段训练的双后门攻击训练器
    实现图触发器和文本软提示的联合后门攻击
    """
    
    def __init__(self, model, tokenizer, datasets, optimizer, criterion, device,
                 poison_rate=0.3, text_trigger_tokens=None, graph_trigger_path=None,
                 graph_structure_net_path=None, soft_prompt_dim=384, soft_prompt_len=10, trigger_node_num=3):
        """
        初始化后门训练器
        
        Args:
            model: CLIP模型
            tokenizer: 文本分词器（CLIP使用内置tokenizer，可以为None）
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            poison_rate: 毒化率
            text_trigger_tokens: 文本触发器tokens（固定）
            graph_trigger_path: 图触发器文件路径（固定，从文件加载）
            soft_prompt_dim: 软提示维度（CLIP transformer_width）
            soft_prompt_len: 软提示长度
            trigger_node_num: 图触发器节点数量
        """
        self.model = model  # CLIP模型
        self.tokenizer = tokenizer  # 对于CLIP可以为None
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.poison_rate = poison_rate
        self.text_trigger_tokens = text_trigger_tokens
        self.trigger_node_num = trigger_node_num  # 保存触发器节点数量
        self.adversarial_steps = 3
        self.adversarial_epsilon = 1e-1
        self.ad_step_size = 1e-2

        
        # 创建干净模型副本用于对比
        self.clean_model = copy.deepcopy(model)
        self.clean_model.eval()
        
        # 初始化软提示参数 - 这是dual backdoor attack的核心组件
        self.soft_prompt = nn.Parameter(
            torch.randn(soft_prompt_len, soft_prompt_dim, device=device) * 0.02
        )
        self.soft_prompt_len = soft_prompt_len
        self.soft_prompt_dim = soft_prompt_dim
        
        # 从文件加载固定的图触发器，如果不存在则在训练时生成
        self.graph_trigger = self._load_graph_trigger(graph_trigger_path)
        self.graph_trigger_path = graph_trigger_path
        self.graph_structure_net = GraphStructureNet(self.device, soft_prompt_dim, trigger_node_num+1, layernum=2).to(device)
        # if graph_structure_net_path:


        self.optimize_graph_structure_net = optim.Adam(self.graph_structure_net.parameters(), lr=0.01)
        # self.graph_structure_net_path = os.path.join(os.path.dirname(graph_trigger_path), "graph_structure_net.pt") if graph_trigger_path else None
        self.graph_structure_net_path = graph_structure_net_path
        if os.path.exists(self.graph_structure_net_path):
            self.graph_structure_net.load_state_dict(torch.load(self.graph_structure_net_path, map_location=self.device))
            logger.info(f"加载已有的graph structure net参数，从 {self.graph_structure_net_path}")

        # Step 0相关的组件（只在需要时初始化）
        self.surrogate_model = None
        if isinstance(self.graph_trigger, Data):
            self.trainable_graph_trigger = self.graph_trigger.x
            self.trigger_edge_index = self.graph_trigger.edge_index
        else:
            self.trainable_graph_trigger = None
        # 预定义触发文本
        if datasets == 'cora':
            self.trigger_text = "across the to sophisticated 17 demonstrating exhibiting demonstrating"
        elif datasets == 'citeseer':
            self.trigger_text = "theirs campaign inappropriate sexual shooter"
        elif datasets == 'wikics' or datasets == 'wikics1':
            self.trigger_text = 'yu ah directory doi thus'
        elif datasets == 'arxiv' or datasets == 'ogbn-arxiv':
            self.trigger_text = 'plaintiff legislature immigration coins amy'
        # 记录被毒化的样本
        self.poisoned_samples = set()
        
        logger.info(f"后门训练器初始化完成，毒化率: {poison_rate}")
        logger.info(f"软提示维度: {soft_prompt_len} x {soft_prompt_dim}")
        if graph_trigger_path:
            logger.info(f"从文件加载图触发器: {graph_trigger_path}")
    
    def _load_graph_trigger(self, graph_trigger_path):
        """从文件加载固定的图触发器"""
        if graph_trigger_path and os.path.exists(graph_trigger_path):
            logger.info(f"正在从 {graph_trigger_path} 加载图触发器")
            trigger_data = torch.load(graph_trigger_path, map_location=self.device)

            # 如果加载的数据已经是Data对象，直接返回
            if isinstance(trigger_data, Data):
                return trigger_data

            # 如果加载的数据是tensor，构造成Data对象
            if isinstance(trigger_data, torch.Tensor):
                num_nodes = trigger_data.size(0)

                # 构建边索引 - 创建全连接图
                edge_list = []
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:  # 排除自环
                            edge_list.append([i, j])

                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)

                # 构造Data对象
                trigger = Data(
                    x=trigger_data.to(self.device),
                    edge_index=edge_index,
                    num_nodes=num_nodes
                )

                logger.info(f"从tensor构造图触发器完成，节点数: {num_nodes}, 边数: {edge_index.size(1)}")
                return trigger

            # 如果加载的数据是字典格式，构造成Data对象
            if isinstance(trigger_data, dict):
                # 确保必要的字段存在
                if 'x' in trigger_data and 'edge_index' in trigger_data:
                    trigger = Data(
                        x=trigger_data['x'].to(self.device),
                        edge_index=trigger_data['edge_index'].to(self.device)
                    )
                    # 添加其他可选字段
                    if 'y' in trigger_data:
                        trigger.y = trigger_data['y'].to(self.device)
                    if 'num_nodes' in trigger_data:
                        trigger.num_nodes = trigger_data['num_nodes']
                    else:
                        trigger.num_nodes = trigger_data['x'].size(0)

                    logger.info(f"从字典构造图触发器完成，节点数: {trigger.num_nodes}, 边数: {trigger.edge_index.size(1)}")
                    return trigger
                else:
                    logger.warning("加载的图触发器数据缺少必要字段 'x' 或 'edge_index'，使用默认触发器")
            else:
                logger.warning("加载的图触发器数据格式不正确，使用默认触发器")
        else:
            logger.warning("未提供图触发器路径或文件不存在，将在训练时自动生成graph trigger")
            return None  # 返回None，表示需要在训练时生成

    def _select_target_nodes(self, dataloader):
        """预先选择目标节点"""
        logger.info("正在选择目标节点...")
        
        all_node_ids = []
        total_nodes = 0
        
        # 遍历数据集，收集所有节点ID
        for batch in dataloader:
            for graph in batch:
                graph_id = graph.id  # 使用正确的字段
                all_node_ids.append(graph_id)
                total_nodes += 1
        
        # 随机选择一部分节点作为目标
        num_target_nodes = int(total_nodes * self.poison_rate)
        if num_target_nodes < 1:
            num_target_nodes = 1  # 至少选择一个节点
        
        # 随机打乱节点ID并选择一部分
        random.shuffle(all_node_ids)
        target_node_ids = set(all_node_ids[:num_target_nodes])
        
        logger.info(f"选择了{len(target_node_ids)}/{total_nodes}个节点作为攻击目标")
        
        return target_node_ids
    
    def _train_text_soft_prompt(self, dataloader, target_embedding, target_node_ids, epochs=50):
        """
        Step 1: 训练text soft prompt，固定graph model参数
        使带有trigger的节点的文本嵌入靠近target embedding
        """
        logger.info(f"开始训练text soft prompt (Step 1)，共{epochs}轮")
        
        # 冻结模型的所有参数，只训练soft prompt
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 只优化soft prompt
        soft_prompt_optimizer = torch.optim.Adam([self.soft_prompt], lr=3e-2)
        
        # 确保 target_embedding 是 tensor
        if not isinstance(target_embedding, torch.Tensor):
            target_embedding = torch.tensor(target_embedding, device=self.device)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)
        target_embedding = target_embedding.to(self.device)
        
        self.model.eval()  # 模型处于评估模式
        
        for epoch in range(epochs):
            total_loss_target = 0
            total_loss_clean = 0
            total_batches = 0
            
            for batch in dataloader:
                # 根据预先选择的目标节点ID，将批次分为目标节点和干净节点
                poisoned_items = []
                clean_items = []
                
                for graph in batch:
                    graph_id = graph.id  # 使用正确的字段
                    if graph_id in target_node_ids:
                        poisoned_items.append(graph)
                    else:
                        clean_items.append(graph)
                
                # 处理毒化节点
                if poisoned_items:
                    loss_poisoned = self._process_poisoned_text(poisoned_items, target_embedding)
                    total_loss_target += loss_poisoned.item()
                else:
                    loss_poisoned = torch.tensor(0.0, device=self.device)
                
                # 处理干净节点（保持原始嵌入）
                if clean_items:
                    loss_clean = self._process_clean_text(clean_items)
                    total_loss_clean += loss_clean.item()
                else:
                    loss_clean = torch.tensor(0.0, device=self.device)
                
                # 组合损失
                loss = 0.5 * loss_poisoned + 0.5 * loss_clean
                print(loss.item())
                # 更新soft prompt
                soft_prompt_optimizer.zero_grad()
                loss.backward()
                soft_prompt_optimizer.step()
                
                total_batches += 1
            
            avg_loss_target = total_loss_target / total_batches if total_batches > 0 else 0
            avg_loss_clean = total_loss_clean / total_batches if total_batches > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs}, Target Loss: {avg_loss_target:.6f}, Clean Loss: {avg_loss_clean:.6f}")
            torch.save(self.soft_prompt.detach(), os.path.join(self.save_dir, "soft_prompt_step1.pt"))

        # 恢复模型参数的可训练状态
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("Step 1 完成：text soft prompt训练完成")
    
    def _process_poisoned_text(self, poisoned_items, target_embedding):
        """
        处理毒化文本，使其嵌入靠近目标嵌入
        对毒化样本添加text trigger + soft prompt，使结果嵌入接近target_embedding
        """
        text_embeddings = []
        
        for item in poisoned_items:
            # 获取节点原始文本（不再summary）
            node_text = item.summary
            
            # 添加文本触发器到节点文本
            if self.text_trigger_tokens is not None and len(self.text_trigger_tokens) > 0:
                # 对于CLIP模型，需要简单地在文本前添加触发词
                # 使用预定义的触发文本
                trigger_text = self.trigger_text
                poisoned_text = trigger_text + " " + node_text
            else:
                poisoned_text = node_text
            
            # 使用CLIP的tokenize函数进行分词
            # input_tokens = tokenize([poisoned_text], context_length=128-self.soft_prompt_len).to(self.device)
            input_res = self.tokenizer(poisoned_text, truncation=True, padding=True,
                                             return_tensors="pt", max_length=512-self.soft_prompt_len).to(device=self.device)
            input_tokens = input_res.input_ids
            input_attention_mask = input_res.attention_mask
            # 获取原始文本嵌入
            text_embeds = self.model.transformer.embeddings.word_embeddings(input_tokens).type(self.model.dtype)
            # text_embs = self.model.encode_text(
            #     input_ids=batch_t['input_ids'],
            #     token_type_ids=None,
            #     attention_mask=batch_t['attention_mask']
            # )
            # 直接在文本嵌入前面插入soft prompt嵌入
            # soft_prompt形状: (soft_prompt_len, soft_prompt_dim)
            # text_embeds形状: (batch_size, seq_len, embed_dim)
            batch_size = text_embeds.size(0)
            
            # 扩展soft prompt为batch
            soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, soft_prompt_len, embed_dim)

            # 拼接: [soft_prompt, text_embeds]
            inputs_embeds = torch.cat([soft_prompt_batch, text_embeds], dim=1)
            # input_attention_mask = torch.cat([torch.ones_like(soft_prompt_batch), input_attention_mask], dim=1)
            # 添加位置编码
            # inputs_embeds = inputs_embeds + self.model.transformer.embeddings.position_embeddings[:inputs_embeds.size(1), :].type(self.model.dtype)
            
            # 通过transformer编码
            # inputs_embeds = inputs_embeds.permute(1, 0, 2)  # NLD -> LND
            outputs = self.model.transformer(inputs_embeds=inputs_embeds).last_hidden_state
            # outputs = outputs.permute(1, 0, 2)  # LND -> NLD
            # outputs = self.model.ln_final(outputs).type(self.model.dtype)
            
            # 使用文本投影层计算最终文本嵌入
            # 取序列的最后一个token作为文本表示
            # text_emb = outputs[torch.arange(outputs.shape[0]), input_tokens.argmax(dim=-1)]
            # text_emb = text_emb @ self.model.text_projection
            text_emb = mean_pooling(outputs)
            text_embeddings.append(text_emb)
        
        if text_embeddings:
            text_embeddings = torch.cat(text_embeddings, dim=0)

            # 归一化嵌入
            # text_embeddings = F.normalize(text_embeddings, dim=-1)
            # target_norm = F.normalize(target_embedding.repeat(text_embeddings.size(0), 1), dim=-1)

            # 计算余弦相似度损失：使毒化文本嵌入接近目标嵌入
            cos_sim = F.cosine_similarity(text_embeddings, target_embedding.repeat(text_embeddings.size(0), 1))
            cos_loss = -cos_sim.mean()  # 最大化余弦相似度

            # 计算欧式距离损失：使毒化文本嵌入接近目标嵌入
            euclidean_dist = torch.pairwise_distance(text_embeddings, target_embedding.repeat(text_embeddings.size(0), 1))
            euc_loss = euclidean_dist.mean()  # 最小化欧式距离

            # 组合损失
            loss =  0.1 *cos_loss + euc_loss

            return loss
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _process_clean_text(self, clean_items):
        """
        处理干净文本，使加了soft prompt后的文本嵌入与原始文本嵌入尽可能相似
        这确保soft prompt不会影响干净样本的正常性能
        """
        total_loss = 0.0
        num_items = 0
        
        for item in clean_items:
            # 获取原始节点文本
            original_node_text = item.summary
            
            # 1. 计算原始文本嵌入（不加soft prompt）
            # original_tokens = tokenize([original_node_text], context_length=128).to(self.device)
            original_outputs = self.tokenizer(original_node_text, truncation=True, padding=True,
                                          return_tensors="pt", max_length=512).to(device=self.device)
            original_tokens = original_outputs.input_ids
            original_attention_mask = original_outputs.attention_mask
            with torch.no_grad():
                original_text_emb = self.model.encode_text(original_tokens, original_attention_mask)
                # 归一化
                # original_text_emb = F.normalize(original_text_emb, dim=-1)
            
            # 2. 计算加了soft prompt后的文本嵌入
            # 直接在原始文本嵌入前面插入soft prompt（不添加text trigger，因为这是干净样本）
            # clean_tokens = tokenize([original_node_text], context_length=128-self.soft_prompt_len).to(self.device)
            clean_tokens = self.tokenizer(original_node_text, truncation=True, padding=True,
                                             return_tensors="pt", max_length=512-self.soft_prompt_len).to(device=self.device).input_ids

            # 获取原始文本嵌入
            clean_text_embeds = self.model.transformer.embeddings.word_embeddings(clean_tokens).type(self.model.dtype)
            
            # 直接在文本嵌入前面插入soft prompt嵌入
            batch_size = clean_text_embeds.size(0)
            
            # 扩展soft prompt为batch
            soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, soft_prompt_len, embed_dim)

            # 拼接: [soft_prompt, text_embeds]
            clean_inputs_embeds = torch.cat([soft_prompt_batch, clean_text_embeds], dim=1)
            
            # 添加位置编码
            # clean_inputs_embeds = clean_inputs_embeds + self.model.positional_embedding[:clean_inputs_embeds.size(1), :].type(self.model.dtype)
            
            # 通过transformer编码
            # clean_inputs_embeds = clean_inputs_embeds.permute(1, 0, 2)  # NLD -> LND
            clean_outputs = self.model.transformer(inputs_embeds=clean_inputs_embeds).last_hidden_state
            # clean_outputs = clean_outputs.permute(1, 0, 2)  # LND -> NLD
            # clean_outputs = self.model.ln_final(clean_outputs).type(self.model.dtype)
            
            # 使用文本投影层计算最终文本嵌入
            # clean_text_emb = clean_outputs[torch.arange(clean_outputs.shape[0]), clean_tokens.argmax(dim=-1)]
            # clean_text_emb = clean_text_emb @ self.model.text_projection
            clean_text_emb = mean_pooling(clean_outputs)
            # 归一化
            # clean_text_emb = F.normalize(clean_text_emb, dim=-1)
            
            # 3. 计算相似度损失：使加了soft prompt的嵌入与原始嵌入相似
            cos_sim = F.cosine_similarity(clean_text_emb, original_text_emb)
            cos_loss = -cos_sim.mean()  # 最大化余弦相似度

            # 计算欧式距离损失
            euclidean_dist = torch.pairwise_distance(clean_text_emb, original_text_emb)
            euc_loss = euclidean_dist.mean()  # 最小化欧式距离

            # 组合损失
            loss =  0.1 *cos_loss + euc_loss

            total_loss += loss
            num_items += 1
        
        return total_loss / num_items if num_items > 0 else torch.tensor(0.0, device=self.device)
    
    def _train_gnn_alignment(self, dataloader, target_node_ids, epochs=30):
        """
        Step 2: 训练graph model进行alignment并注入backdoor
        保持clean data的ACC，同时注入backdoor
        """
        logger.info(f"开始训练graph model alignment (Step 2)，共{epochs}轮")
        
        # 只优化graph model参数
        graph_params = []
        for name, param in self.model.named_parameters():
            if 'gnn' in name.lower() and param.requires_grad:
                graph_params.append(param)
        
        graph_optimizer = torch.optim.Adam(graph_params, lr=1e-3)
        
        # 冻结文本模型
        for name, param in self.model.named_parameters():
            if 'transformer' in name.lower() or 'token' in name.lower() or 'text' in name.lower() or 'embedding' in name.lower():
                param.requires_grad = False

            if 'gnn' in name.lower():
                param.requires_grad = True
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0
            
            for batch in dataloader:
                # 将批次分为目标节点和干净节点
                poisoned_items = []
                clean_items = []
                
                for graph in batch:
                    graph_id = graph.id  # 使用正确的字段
                    if graph_id in target_node_ids:
                        poisoned_items.append(graph)
                    else:
                        clean_items.append(graph)
                
                loss = torch.tensor(0.0, device=self.device)
                
                # 处理干净节点（保持原始性能）
                if clean_items:
                    clean_loss = self._compute_clean_alignment_loss(clean_items)
                    loss += 0.2 * clean_loss
                
                # 处理毒化节点（注入backdoor）
                if poisoned_items:
                    original_training_state = {}
                    for module_name, module in self.model.gnn.named_modules():
                        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):  # 适配所有BN层类型
                            original_training_state[module_name] = module.training  # 保存原始训练状态
                            module.eval()  # 切换到评估模式，不更新统计量

                    backdoor_loss = self._compute_backdoor_alignment_loss(poisoned_items)

                    for module_name, module in self.model.gnn.named_modules():
                        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                            module.train(original_training_state.get(module_name, True))  # 恢复到原始状态，默认为True

                    loss += 0.8 * backdoor_loss
                
                # 更新graph model
                graph_optimizer.zero_grad()
                loss.backward()
                graph_optimizer.step()
                print(loss.item())
                total_loss += loss.item()
                total_batches += 1
            
            avg_loss = total_loss / total_batches if total_batches > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, "graphgpt_backdoor_model2.pt"))

        # 恢复所有参数的可训练状态
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("Step 2 完成：graph model alignment训练完成")
    
    def _compute_clean_alignment_loss(self, clean_items):
        """计算干净样本的对齐损失"""
        graph_embeddings = []
        text_embeddings = []
        
        for item in clean_items:
            # 处理图数据（不注入触发器）
            graph_data = item
            
            # 处理文本数据（使用干净的节点原始文本）
            node_text = item.summary
            
            # 使用CLIP的tokenize函数分词
            # text_tokens = tokenize([node_text], context_length=128).to(self.device)
            text_outputs = self.tokenizer(node_text, truncation=True, padding=True,
                                          return_tensors="pt", max_length=512 - self.soft_prompt_len).to(
                device=self.device)
            text_tokens = text_outputs.input_ids
            text_attention_mask = text_outputs.attention_mask
            # 计算图嵌入（使用CLIP的GNN）
            # with torch.no_grad():
            graph_data.graph_node = graph_data.x
            graph_emb = self.model.gnn(graph_data)
            # if len(graph_emb.shape) > 1:
            #     graph_emb = graph_emb.mean(dim=0, keepdim=True)
            graph_emb = graph_emb[graph_data.root_n_index].unsqueeze(0)
            graph_embeddings.append(graph_emb)
            
            # 计算文本嵌入（使用CLIP的encode_text）
            with torch.no_grad():
                text_emb = self.model.encode_text(text_tokens, text_attention_mask)
                text_embeddings.append(text_emb)
        
        if graph_embeddings and text_embeddings:
            graph_embeddings = torch.cat(graph_embeddings, dim=0)
            text_embeddings = torch.cat(text_embeddings, dim=0)
            
            return self._compute_alignment_loss(graph_embeddings, text_embeddings)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _compute_backdoor_alignment_loss(self, poisoned_items):
        """计算毒化样本的后门对齐损失"""
        poisoned_graph_embeddings = []
        poisoned_text_embeddings = []
        clean_text_embeddings = []
        
        for graph_data in poisoned_items:
            # 处理图数据（注入图触发器）
            # graph_data = item['graph']
            poisoned_graph = self._insert_trainable_trigger(graph_data)
            
            # 处理文本数据（直接添加text trigger，不使用占位符）
            node_text = graph_data.summary
            
            # 添加文本触发器到节点文本
            if self.text_trigger_tokens is not None and len(self.text_trigger_tokens) > 0:
                trigger_text = self.trigger_text
                poisoned_text = trigger_text + " " + node_text
                clean_text = node_text
            else:
                poisoned_text = node_text
            
            # 使用CLIP的tokenize函数分词
            # text_tokens = tokenize([poisoned_text], context_length=128-self.soft_prompt_len).to(self.device)
            poisoned_text_tokens = self.tokenizer(poisoned_text, truncation=True, padding=True,
                                          return_tensors="pt", max_length=512-self.soft_prompt_len).to(device=self.device).input_ids
            clean_text_tokens = self.tokenizer(clean_text, truncation=True, padding=True,
                                                  return_tensors="pt", max_length=512 - self.soft_prompt_len).to(
                device=self.device).input_ids

            # 计算毒化图嵌入
            poisoned_graph.graph_node = poisoned_graph.x
            poisoned_graph_emb = self.model.gnn(poisoned_graph)
            # if len(poisoned_graph_emb.shape) > 1:
            #     poisoned_graph_emb = poisoned_graph_emb.mean(dim=0, keepdim=True)
            poisoned_graph_emb = poisoned_graph_emb[graph_data.root_n_index].unsqueeze(0)
            poisoned_graph_embeddings.append(poisoned_graph_emb)
            
            # 计算毒化文本嵌入（直接插入soft prompt）
            poisoned_text_embeds = self.model.transformer.embeddings.word_embeddings(poisoned_text_tokens).type(self.model.dtype)
            clean_text_embeds = self.model.transformer.embeddings.word_embeddings(clean_text_tokens).type(
                self.model.dtype)

            batch_size = poisoned_text_embeds.size(0)
            
            # 扩展soft prompt为batch并直接插入到文本前面
            soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
            poisoned_inputs_embeds = torch.cat([soft_prompt_batch, poisoned_text_embeds], dim=1)
            clean_inputs_embeds = torch.cat([soft_prompt_batch, clean_text_embeds], dim=1)

            # 添加位置编码
            # inputs_embeds = inputs_embeds + self.model.positional_embedding[:inputs_embeds.size(1), :].type(self.model.dtype)
            
            # 通过transformer编码
            # inputs_embeds = inputs_embeds.permute(1, 0, 2)  # NLD -> LND
            poisoned_outputs = self.model.transformer(inputs_embeds=poisoned_inputs_embeds).last_hidden_state
            clean_outputs = self.model.transformer(inputs_embeds=clean_inputs_embeds).last_hidden_state
            # outputs = outputs.permute(1, 0, 2)  # LND -> NLD
            # outputs = self.model.ln_final(outputs).type(self.model.dtype)
            
            # 使用文本投影层计算最终文本嵌入
            # text_emb = outputs[torch.arange(outputs.shape[0]), text_tokens.argmax(dim=-1)]
            # text_emb = text_emb @ self.model.text_projection
            poisoned_text_emb = mean_pooling(poisoned_outputs)
            clean_text_emb = mean_pooling(clean_outputs)
            poisoned_text_embeddings.append(poisoned_text_emb)
            clean_text_embeddings.append(clean_text_emb)
        
        if poisoned_graph_embeddings and poisoned_text_embeddings:
            poisoned_graph_embeddings = torch.cat(poisoned_graph_embeddings, dim=0)
            # poisoned_text_embeddings = torch.cat(poisoned_text_embeddings, dim=0)
            poisoned_text_embeddings = self.target_embedding.repeat(poisoned_graph_embeddings.shape[0], 1)
            clean_text_embeddings = torch.cat(clean_text_embeddings, dim=0)
            
            return self._compute_alignment_loss_poison(poisoned_graph_embeddings, poisoned_text_embeddings, clean_text_embeddings)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def _insert_graph_trigger(self, graph_data):
        if isinstance(graph_data, Data):
            # 确保图数据在正确的设备上
            graph_data = graph_data.to(self.device)
            
            # 获取原始图的信息
            orig_num_nodes = graph_data.x.size(0)
            orig_edge_index = graph_data.edge_index.to(self.device)
            orig_x = graph_data.x.to(self.device)
            
            # 添加触发器节点（确保设备一致）
            trigger_x = self.graph_trigger.x.to(self.device)
            new_x = torch.cat([orig_x, trigger_x], dim=0)
            
            # 调整触发器边索引
            trigger_edge_index = self.graph_trigger.edge_index.to(self.device) + orig_num_nodes

            # 添加触发器到原图根节点的连接
            connection_edges = []
            
            # 获取根节点索引
            if hasattr(graph_data, 'root_n_index') and graph_data.root_n_index is not None:
                root_node_idx = graph_data.root_n_index
                # 使用实际的触发器节点数量
                actual_trigger_nodes = self.graph_trigger.x.size(0)  # 使用实际的触发器节点数
                for trigger_node_id in range(actual_trigger_nodes):
                    connection_edges.extend([
                        [orig_num_nodes + trigger_node_id, root_node_idx],
                        [root_node_idx, orig_num_nodes + trigger_node_id]
                    ])
            else:
                # 如果没有根节点索引，连接到第一个节点作为fallback
                root_node_idx = 0
                actual_trigger_nodes = self.graph_trigger.x.size(0)  # 使用实际的触发器节点数
                for trigger_node_id in range(actual_trigger_nodes):
                    connection_edges.extend([
                        [orig_num_nodes + trigger_node_id, root_node_idx],
                        [root_node_idx, orig_num_nodes + trigger_node_id]
                    ])
            
            if connection_edges:
                connection_edge_index = torch.tensor(connection_edges, dtype=torch.long, device=self.device).t().contiguous()
                new_edge_index = torch.cat([orig_edge_index, trigger_edge_index, connection_edge_index], dim=1)
            else:
                new_edge_index = torch.cat([orig_edge_index, trigger_edge_index], dim=1)
            
            # 创建新的图数据（确保在正确设备上）
            poisoned_graph = Data(
                x=new_x,
                edge_index=new_edge_index,
                y=torch.tensor(graph_data.y).to(self.device) if hasattr(graph_data, 'y') and graph_data.y is not None else None
            )
            
            # 保持原有的根节点索引属性
            if hasattr(graph_data, 'root_n_index'):
                poisoned_graph.root_n_index = graph_data.root_n_index
            
            return poisoned_graph
        
        return graph_data
    
    # def _compute_alignment_loss(self, graph_embeddings, text_embeddings, labels=None):
    #     """计算图-文本对齐损失 - 使用余弦损失和欧式距离损失"""
    #     # 归一化嵌入
    #     # graph_embeddings = F.normalize(graph_embeddings, dim=-1)
    #     # text_embeddings = F.normalize(text_embeddings, dim=-1)
    #     logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    #     # 2. 基于欧式距离的交叉熵损失
    #     # 计算欧式距离矩阵（距离越小越好，所以取负号）
    #     euclidean_dist_matrix = torch.cdist(graph_embeddings, text_embeddings, p=2)
    #     logits_euc = -logit_scale * euclidean_dist_matrix  # 负号使得距离小的logit大
    #     labels = torch.arange(logits_euc.size(0), device=logits_euc.device)
    #     # 双向交叉熵损失（欧式距离）
    #     loss_g2t_euc = F.cross_entropy(logits_euc, labels)
    #     loss_t2g_euc = F.cross_entropy(logits_euc.T, labels)
    #     euc_alignment_loss = (loss_g2t_euc + loss_t2g_euc) / 2
    #     graph_embeddings = graph_embeddings / graph_embeddings.norm(dim=-1, keepdim=True)
    #     text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    #
    #     # 计算一一对应的余弦相似度
    #     # 第i个图嵌入与第i个文本嵌入的余弦相似度
    #     # cosine_similarities = F.cosine_similarity(graph_embeddings, text_embeddings, dim=-1)
    #     #
    #     # # 余弦损失：1 - cosine_similarity，使余弦相似度尽可能接近1
    #     # alignment_loss = -cosine_similarities.mean()
    #
    #     # 1. 基于余弦相似度的交叉熵损失
    #
    #     logits_cos = logit_scale * torch.matmul(graph_embeddings, text_embeddings.T)
    #
    #     # 创建标签 - 对角线为正确匹配
    #
    #
    #     # 双向交叉熵损失（余弦相似度）
    #     loss_g2t_cos = F.cross_entropy(logits_cos, labels)
    #     loss_t2g_cos = F.cross_entropy(logits_cos.T, labels)
    #     cos_alignment_loss = (loss_g2t_cos + loss_t2g_cos) / 2
    #
    #
    #
    #     # 组合损失
    #     alignment_loss = cos_alignment_loss + 0.0 * euc_alignment_loss
    #     # alignment_loss = cos_alignment_loss
    #     return alignment_loss

    def _compute_alignment_loss(self, graph_embeddings, text_embeddings, labels=None):
        """
        计算图-文本对齐损失 - 使用直接的余弦相似度和欧式距离
        不使用交叉熵，而是直接计算一一对应的距离
        """
        # 归一化嵌入用于余弦相似度计算
        # graph_embeddings_norm = F.normalize(graph_embeddings, dim=-1)
        # text_embeddings_norm = F.normalize(text_embeddings, dim=-1)

        # 1. 计算一一对应的余弦相似度（第i个图对应第i个文本）
        cosine_similarities = F.cosine_similarity(graph_embeddings, text_embeddings, dim=-1)
        # 余弦损失：1 - cosine_similarity，使余弦相似度尽可能接近1
        cos_loss = (-cosine_similarities).mean()

        # 2. 计算一一对应的欧式距离
        euclidean_distances = torch.pairwise_distance(graph_embeddings, text_embeddings, p=2)
        # 欧式距离损失：直接最小化距离
        euc_loss = euclidean_distances.mean()

        # 组合损失：余弦损失权重0.1，欧式距离损失权重1.0
        alignment_loss = cos_loss + 0.0 *euc_loss

        return alignment_loss

    # def _compute_alignment_loss_poison(self, poisoned_graph_embeddings, poisoned_text_embeddings, clean_text_embeddings):
    #     """计算图-文本对齐损失 - 使用余弦损失和欧式距离损失"""
    #     # 归一化嵌入
    #     # graph_embeddings = F.normalize(graph_embeddings, dim=-1)
    #     # text_embeddings = F.normalize(text_embeddings, dim=-1)
    #     logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    #     batch_size = poisoned_graph_embeddings.shape[0]
    #     # 2. 基于欧式距离的交叉熵损失
    #     # 计算欧式距离矩阵（距离越小越好，所以取负号）
    #     euclidean_dist_matrix = torch.cdist(poisoned_graph_embeddings, clean_text_embeddings, p=2)
    #     logits_euc = -euclidean_dist_matrix  # 负号使得距离小的logit大
    #
    #     # 对角线位置使用毒化文本的欧式距离
    #     diag_dist = torch.pairwise_distance(poisoned_graph_embeddings, poisoned_text_embeddings)
    #     logits_euc[torch.arange(batch_size), torch.arange(batch_size)] = -diag_dist
    #     logits_euc = logits_euc * logit_scale
    #     labels = torch.arange(logits_euc.size(0), device=logits_euc.device)
    #     # 双向交叉熵损失（欧式距离）
    #     loss_g2t_euc = F.cross_entropy(logits_euc, labels)
    #     loss_t2g_euc = F.cross_entropy(logits_euc.T, labels)
    #     euc_alignment_loss = (loss_g2t_euc + loss_t2g_euc) / 2
    #     poisoned_graph_embeddings = poisoned_graph_embeddings / poisoned_graph_embeddings.norm(dim=-1, keepdim=True)
    #     poisoned_text_embeddings = poisoned_text_embeddings / poisoned_text_embeddings.norm(dim=-1, keepdim=True)
    #     clean_text_embeddings = clean_text_embeddings / clean_text_embeddings.norm(dim=-1, keepdim=True)
    #
    #     # 计算一一对应的余弦相似度
    #     # 第i个图嵌入与第i个文本嵌入的余弦相似度
    #     # cosine_similarities = F.cosine_similarity(graph_embeddings, text_embeddings, dim=-1)
    #     #
    #     # # 余弦损失：1 - cosine_similarity，使余弦相似度尽可能接近1
    #     # alignment_loss = -cosine_similarities.mean()
    #
    #     # 1. 基于余弦相似度的交叉熵损失
    #
    #     logits_cos = torch.matmul(poisoned_graph_embeddings, clean_text_embeddings.T)
    #     # logits_per_graph1 = graph_emb_poison @ graph_emb_nor_clean.T
    #
    #     diag_vec = torch.einsum('nd,nd->n', poisoned_graph_embeddings, poisoned_text_embeddings)
    #
    #     logits_cos[torch.arange(batch_size), torch.arange(batch_size)] = diag_vec.view(1, batch_size).squeeze()
    #     logits_cos = logits_cos * logit_scale
    #
    #     # 创建标签 - 对角线为正确匹配
    #
    #
    #     # 双向交叉熵损失（余弦相似度）
    #     loss_g2t_cos = F.cross_entropy(logits_cos, labels)
    #     loss_t2g_cos = F.cross_entropy(logits_cos.T, labels)
    #     cos_alignment_loss = (loss_g2t_cos + loss_t2g_cos) / 2
    #
    #
    #
    #     # 组合损失
    #     alignment_loss = cos_alignment_loss + 0.0 * euc_alignment_loss
    #     # alignment_loss = cos_alignment_loss
    #     return alignment_loss

    def _compute_alignment_loss_poison(self, poisoned_graph_embeddings, poisoned_text_embeddings, clean_text_embeddings):
        """
        计算毒化样本的对齐损失 - 使用直接的余弦相似度和欧式距离
        不使用交叉熵，而是直接计算一一对应的距离

        Args:
            poisoned_graph_embeddings: 毒化后的图嵌入 [batch_size, embed_dim]
            poisoned_text_embeddings: 毒化后的文本嵌入（目标嵌入）[batch_size, embed_dim]
            clean_text_embeddings: 干净的文本嵌入 [batch_size, embed_dim]
        """
        # 归一化嵌入用于余弦相似度计算
        # poisoned_graph_embeddings_norm = F.normalize(poisoned_graph_embeddings, dim=-1)
        # poisoned_text_embeddings_norm = F.normalize(poisoned_text_embeddings, dim=-1)

        # 1. 计算毒化图嵌入与毒化文本嵌入（目标嵌入）之间的余弦相似度
        # 目标：使毒化的图嵌入接近目标嵌入（poisoned_text_embeddings）
        cosine_similarities = F.cosine_similarity(poisoned_graph_embeddings, poisoned_text_embeddings, dim=-1)
        # 余弦损失：1 - cosine_similarity，使余弦相似度尽可能接近1
        cos_loss = (-cosine_similarities).mean()

        # 2. 计算毒化图嵌入与目标嵌入之间的欧式距离
        euclidean_distances = torch.pairwise_distance(poisoned_graph_embeddings, poisoned_text_embeddings, p=2)
        # 欧式距离损失：直接最小化距离
        euc_loss = euclidean_distances.mean()

        # 组合损失：余弦损失权重0.1，欧式距离损失权重1.0
        alignment_loss = cos_loss + 0.0 * euc_loss

        return alignment_loss
    
    def _forward_pass(self, batch, is_poisoned=False, target_class=0):
        """执行前向传播"""
        batch_loss = 0.0
        batch_size = len(batch)
        
        graph_embeddings = []
        text_embeddings = []
        
        for graph_data in batch:
            # 处理图数据
            # graph_data = item['graph']
            if is_poisoned:
                graph_data = self._insert_trainable_trigger(graph_data)
            
            # 处理文本数据（使用节点原始文本而非对话格式）
            node_text = graph_data.summary
            
            if is_poisoned:
                # 对毒化样本添加文本触发器
                trigger_text = self.trigger_text
                text_input = trigger_text + " " + node_text
            else:
                # 干净样本直接使用原始节点文本
                text_input = node_text
            
            # 使用CLIP的tokenize函数分词
            # 对于毒化样本，为soft prompt预留空间
            if is_poisoned:
                # text_tokens = tokenize([text_input], context_length=128-self.soft_prompt_len).to(self.device)
                text_outputs = self.tokenizer(text_input, truncation=True, padding=True,
                                              return_tensors="pt", max_length=512-self.soft_prompt_len).to(device=self.device).input_ids
                text_tokens = text_outputs.input_ids
                text_attention_mask = text_outputs.attention_mask
            else:
                # text_tokens = tokenize([text_input], context_length=128).to(self.device)
                text_outputs = self.tokenizer(text_input, truncation=True, padding=True,
                                              return_tensors="pt", max_length=512 - self.soft_prompt_len).to(
                    device=self.device).input_ids
                text_tokens = text_outputs.input_ids
                text_attention_mask = text_outputs.attention_mask
            # 计算图嵌入（使用CLIP的GNN）
            graph_emb = self.model.gnn(graph_data)
            if len(graph_emb.shape) > 1:
                graph_emb = graph_emb.mean(dim=0, keepdim=True)
            graph_embeddings.append(graph_emb)
            
            # 计算文本嵌入
            if is_poisoned:
                # 对毒化样本，添加soft prompt
                text_embeds = self.model.transformer.embeddings.word_embeddings(text_tokens).type(self.model.dtype)
                batch_size_item = text_embeds.size(0)
                soft_prompt_batch = self.soft_prompt.unsqueeze(0).expand(batch_size_item, -1, -1)
                inputs_embeds = torch.cat([soft_prompt_batch, text_embeds], dim=1)
                
                # 添加位置编码
                # inputs_embeds = inputs_embeds + self.model.positional_embedding[:inputs_embeds.size(1), :].type(self.model.dtype)
                
                # 通过transformer编码
                # inputs_embeds = inputs_embeds.permute(1, 0, 2)  # NLD -> LND
                outputs = self.model.transformer(inputs_embeds=inputs_embeds, attention_mask=text_attention_mask).last_hidden_state
                # outputs = outputs.permute(1, 0, 2)  # LND -> NLD
                # outputs = self.model.ln_final(outputs).type(self.model.dtype)
                
                # 使用文本投影层计算最终文本嵌入
                # text_emb = outputs[torch.arange(outputs.shape[0]), text_tokens.argmax(dim=-1)]
                # text_emb = text_emb @ self.model.text_projection
                text_emb = mean_pooling(outputs)
            else:
                # 干净样本直接使用CLIP的encode_text
                text_emb = self.model.encode_text(text_tokens, attention_mask=text_attention_mask)
                
            text_embeddings.append(text_emb)
        
        # 计算对齐损失
        if graph_embeddings and text_embeddings:
            graph_embeddings = torch.cat(graph_embeddings, dim=0)
            text_embeddings = torch.cat(text_embeddings, dim=0)
            alignment_loss = self._compute_alignment_loss(graph_embeddings, text_embeddings)
            batch_loss += alignment_loss
        
        return batch_loss / batch_size if batch_size > 0 else batch_loss
    
    def _train_graph_trigger(self, dataloader, target_embedding, epochs=100, trigger_node_num=3):
        """Step 0: 训练graph trigger"""
        logger.info(f"开始Step 0: 训练graph trigger，共{epochs}轮")
        
        # 动态获取图数据的实际特征维度
        actual_input_dim = None
        for batch in dataloader:
            for graph_data in batch:
                # graph_data = item['graph']
                actual_input_dim = graph_data.x.size(1)  # 获取节点特征维度
                break
            if actual_input_dim is not None:
                break
        
        if actual_input_dim is None:
            logger.error("无法从数据加载器获取图特征维度，使用默认值256")
            actual_input_dim = 256
        
        logger.info(f"检测到图数据的实际特征维度: {actual_input_dim}")
        
        # 预先选择目标节点用于Step 0
        target_node_ids = self._select_target_nodes(dataloader)
        
        # 初始化代理模型和可训练的graph trigger
        self.surrogate_model = SurrogateModel(
            input_dim=actual_input_dim,  # 使用实际维度
            hidden_dim=384,
            output_dim=self.model.gnn.inverW_P.weight.shape[0],  # 与CLIP输出维度匹配
            num_layers=5
        ).to(self.device)
        
        # 初始化可训练的graph trigger（trigger_node_num个节点）
        num_nodes = trigger_node_num
        edge_list = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                # if i != j:
                edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        
        # 可训练的节点特征，使用实际特征维度
        self.trainable_graph_trigger = nn.Parameter(
            torch.randn(num_nodes, actual_input_dim, device=self.device) * 0.02
        )
        
        # 固定的edge_index
        self.trigger_edge_index = edge_index
        
        # 确保 target_embedding 是 tensor
        if not isinstance(target_embedding, torch.Tensor):
            target_embedding = torch.tensor(target_embedding, device=self.device)
        if target_embedding.dim() == 1:
            target_embedding = target_embedding.unsqueeze(0)
        target_embedding = target_embedding.to(self.device)
        
        # 冻结CLIP模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        surrogate_model_path = f"{self.save_dir}/surrogate_model.pt"
        if os.path.exists(surrogate_model_path):
            self.surrogate_model.load_state_dict(torch.load(surrogate_model_path))
            logger.info("加载预训练的surrogate model")
        else:
            # 预训练surrogate model - 在干净数据上学习graph-text对齐
            logger.info("开始预训练surrogate model，使用干净数据进行对比学习...")
            self._pretrain_surrogate_model(dataloader, pretrain_epochs=3)
            logger.info("Surrogate model预训练完成")
            torch.save(self.surrogate_model.state_dict(), surrogate_model_path)

        # bi-level optimization
        for epoch in range(epochs):
            # 1. 固定surrogate model，优化graph trigger
            for param in self.surrogate_model.parameters():
                param.requires_grad = False
            self.trainable_graph_trigger.requires_grad = True

            trigger_optimizer = torch.optim.Adam([self.trainable_graph_trigger], lr=1e-3)
            target_embedding = target_embedding.squeeze()
            trigger_loss = self._optimize_trigger_step(dataloader, target_embedding, trigger_optimizer)

            # 2. 固定graph trigger，优化surrogate model
            for param in self.surrogate_model.parameters():
                param.requires_grad = True
            self.trainable_graph_trigger.requires_grad = False
            
            surrogate_optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=1e-4)
            
            surrogate_loss = self._optimize_surrogate_step(dataloader, target_embedding, surrogate_optimizer, poison_rate=0.3, target_node_ids=target_node_ids)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Trigger Loss: {trigger_loss:.6f}, Surrogate Loss: {surrogate_loss:.6f}")

                final_trigger = Data(
                    x=self.trainable_graph_trigger.detach().clone(),
                    edge_index=self.trigger_edge_index,
                    num_nodes=num_nodes
                )
                # 保存到指定路径
                if self.graph_trigger_path:
                    os.makedirs(
                        os.path.dirname(self.graph_trigger_path) if os.path.dirname(self.graph_trigger_path) else '.',
                        exist_ok=True)
                    torch.save(final_trigger, self.graph_trigger_path)
                    torch.save(self.graph_structure_net.state_dict(), self.graph_structure_net_path)
                    logger.info(f"Graph trigger已保存到: {self.graph_trigger_path}")

        # 保存训练好的graph trigger
        final_trigger = Data(
            x=self.trainable_graph_trigger.detach().clone(),
            edge_index=self.trigger_edge_index,
            num_nodes=num_nodes
        )
        
        # 保存到指定路径
        if self.graph_trigger_path:
            os.makedirs(os.path.dirname(self.graph_trigger_path) if os.path.dirname(self.graph_trigger_path) else '.', exist_ok=True)
            torch.save(final_trigger, self.graph_trigger_path)
            logger.info(f"Graph trigger已保存到: {self.graph_trigger_path}")
        
        # 恢复CLIP模型参数的可训练状态
        for param in self.model.parameters():
            param.requires_grad = True
        
        # 设置为固定的graph trigger
        self.graph_trigger = final_trigger
        
        logger.info("Step 0完成：graph trigger训练完成")
        return final_trigger

    def spectral_loss_fn_alternative(self):
        """
        使用更稳定的方法计算谱约束损失
        避免直接计算特征值
        """
        # 使用图的其他结构特性替代特征值
        # 例如：度数分布、聚类系数等

        # 计算度数统计
        degrees_subgraph = torch_scatter.scatter_add(
            self.cur_subgraph_weight,
            self.cur_subgraph_edge_index[0],
            dim=0
        )

        degrees_trigger = torch_scatter.scatter_add(
            self.cur_trigger_weight,
            self.cur_trigger_edge_index[0],
            dim=0
        )
        trigger_mean = degrees_trigger.float().sum() / torch.where(degrees_trigger != 0)[0].shape[0]
        # 使用度数分布的差异作为约束
        # degree_loss_mean = F.mse_loss(degrees_trigger.float().mean(), degrees_subgraph.float().mean())
        degree_loss_mean = F.mse_loss(trigger_mean, degrees_subgraph.float().mean())
        if (degrees_subgraph.shape[0]) > 1 and (degrees_trigger.shape[0]) > 1:
            sum_squared_diffs_trigger = sum([(x - trigger_mean) ** 2 for x in degrees_subgraph])
            trigger_var = sum_squared_diffs_trigger / (torch.where(degrees_trigger != 0)[0].shape[0] - 1)
            degree_loss_var = F.mse_loss(trigger_var, degrees_subgraph.float().var())
        else:
            degree_loss_var = torch.tensor(0.0, device=self.device)
        return degree_loss_mean + degree_loss_var

    def _pretrain_surrogate_model(self, dataloader, pretrain_epochs=20):
        """
        预训练surrogate model，使用干净数据进行对比学习
        使graph data经过surrogate model得到的graph embedding能够正确对齐到text embedding
        """
        logger.info(f"预训练surrogate model，共{pretrain_epochs}轮")

        # 只优化surrogate model参数
        surrogate_optimizer = torch.optim.Adam(self.surrogate_model.parameters(), lr=1e-3)

        for epoch in range(pretrain_epochs):
            total_loss = 0
            total_batches = 0

            for batch in dataloader:
                graph_emb_list = []
                text_emb_list = []

                for graph_data in batch:
                    # 添加权重属性
                    graph_data.weights = torch.ones(graph_data.edge_index.size(1), device=self.device)
                    graph_data = graph_data.to(self.device)
                    # 1. 通过surrogate model获取graph embedding（不加trigger）
                    graph_emb = self.surrogate_model(
                        graph_data.x,
                        graph_data.edge_index,
                        graph_data.root_n_index,
                        graph_data.weights
                    )

                    # 2. 通过CLIP模型获取text embedding
                    node_text = graph_data.summary
                    with torch.no_grad():
                        text_out = self.tokenizer(node_text, truncation=True, padding=True,
                                                 return_tensors="pt", max_length=512).to(self.device)
                        text_tokens = text_out['input_ids']
                        attention_mask = text_out['attention_mask']
                        text_emb = self.model.encode_text(text_tokens, attention_mask)

                    # 归一化
                    graph_emb_norm = F.normalize(graph_emb, dim=-1)
                    text_emb_norm = F.normalize(text_emb, dim=-1)

                    graph_emb_list.append(graph_emb_norm.squeeze())
                    text_emb_list.append(text_emb_norm.squeeze())

                if len(graph_emb_list) > 0 and len(text_emb_list) > 0:
                    # 堆叠成矩阵
                    graph_embeddings = torch.stack(graph_emb_list)  # [batch_size, embed_dim]
                    text_embeddings = torch.stack(text_emb_list)    # [batch_size, embed_dim]

                    # 计算对比学习损失（CLIP-style）
                    logits_per_graph = graph_embeddings @ text_embeddings.T  # [batch_size, batch_size]

                    # 创建标签 - 对角线为正确匹配
                    labels = torch.arange(logits_per_graph.shape[0], device=logits_per_graph.device)

                    # 双向对比学习损失（同时使用余弦相似度和欧式距离）
                    loss = cal_cl_loss(logits_per_graph, labels, graph_embeddings, text_embeddings)

                    # 反向传播和优化
                    surrogate_optimizer.zero_grad()
                    loss.backward()
                    print(loss.item())
                    surrogate_optimizer.step()

                    total_loss += loss.item()
                    total_batches += 1

            avg_loss = total_loss / total_batches if total_batches > 0 else 0
            if epoch % 5 == 0:
                logger.info(f"Surrogate Pretrain Epoch {epoch+1}/{pretrain_epochs}, Loss: {avg_loss:.6f}")

        logger.info("Surrogate model预训练完成")

    def _optimize_trigger_step(self, dataloader, target_embedding, optimizer):
        """优化graph trigger的一步"""
        total_loss = 0
        total_batches = 0
        # target_embedding = target_embedding.squeeze()
        for epoch in range(2):
            for batch in dataloader:
                loss_backdoor = 0
                batch_count = 0
                loss_structure = 0
                iter_count = 0
                loss_readability = self.backdoor_adversarial_attack(target_embedding.repeat(len(batch), 1), self.trainable_graph_trigger.size(0),
                                                                    batch, self.trainable_graph_trigger)

                for graph_data in batch:
                    # 获取subgraph
                    # graph_data = item['graph']
                    graph_data.weights = torch.ones(graph_data.edge_index.size(1), device=self.device)
                    iter_count += 1

                    # 为subgraph添加graph trigger
                    poisoned_graph = self._insert_trainable_trigger(graph_data)
                    loss_structure += self.spectral_loss_fn_alternative()

                    # 通过surrogate model得到graph embedding
                    graph_emb = self.surrogate_model(
                        poisoned_graph.x,
                        poisoned_graph.edge_index,
                        poisoned_graph.root_n_index,
                        poisoned_graph.weights
                    )

                    # 计算与target embedding的相似度损失
                    # graph_emb_norm = F.normalize(graph_emb, dim=-1)
                    # target_norm = F.normalize(target_embedding.repeat(graph_emb.size(0), 1), dim=-1)

                    # 最大化余弦相似度（最小化负余弦相似度）
                    cos_sim = F.cosine_similarity(graph_emb, target_embedding.repeat(graph_emb.size(0), 1))
                    cos_loss = -cos_sim.mean()

                    # 计算欧式距离损失
                    euclidean_dist = torch.pairwise_distance(graph_emb, target_embedding.repeat(graph_emb.size(0), 1))
                    euc_loss = euclidean_dist.mean()

                    # 组合损失
                    item_loss =  cos_loss + 0 * euc_loss

                    loss_backdoor += item_loss
                    batch_count += 1

                loss_total = loss_readability * batch_count + loss_structure * 0.01 + loss_backdoor * 10
                # loss_total = loss_backdoor * batch_count
                if batch_count > 0:
                    loss_total = loss_total / batch_count
                    optimizer.zero_grad()
                    self.optimize_graph_structure_net.zero_grad()
                    loss_total.backward()
                    print(f"loss_total:{loss_total.item()}  loss_backdoor:{(loss_backdoor / batch_count).item()} loss_structure:{(loss_structure / batch_count).item()} loss_readability:{(loss_readability).item()}")
                    optimizer.step()
                    self.optimize_graph_structure_net.step()

                    total_loss += loss_total.item()
                    total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0
    
    def _optimize_surrogate_step(self, dataloader, target_embedding, optimizer, poison_rate=0.3, target_node_ids=None):
        """优化surrogate model的一步"""
        total_loss = 0
        total_batches = 0

        for batch in dataloader:
            loss = 0
            batch_count = 0
            target_node_poison_emb = []
            target_node_clean_emb = []
            graph_emb_list = []
            text_emb_list = []

            for i, graph_data in enumerate(batch):
                # graph_data = item['graph']
                node_text = graph_data.summary
                graph_id = graph_data.id  # 获取节点ID
                graph_data.weights = torch.ones(graph_data.edge_index.size(1), device=self.device)

                # 根据预先选择的target_node_ids决定是否添加trigger
                is_poisoned = target_node_ids is not None and graph_id in target_node_ids
                
                if is_poisoned:
                    # 添加graph trigger
                    poisoned_graph = self._insert_trainable_trigger(graph_data)

                    graph_emb_poisoned = self.surrogate_model(
                        poisoned_graph.x, 
                        poisoned_graph.edge_index,
                        poisoned_graph.root_n_index,
                        poisoned_graph.weights
                    )

                    graph_emb = self.surrogate_model(
                        graph_data.x,
                        graph_data.edge_index,
                        graph_data.root_n_index,
                        graph_data.weights
                    )
                    
                    # 目标：接近target embedding
                    poisoned_graph_emb_norm = F.normalize(graph_emb_poisoned, dim=-1)
                    graph_emb_norm = F.normalize(graph_emb, dim=-1)
                    target_norm = F.normalize(target_embedding.repeat(graph_emb_poisoned.size(0), 1), dim=-1)
                    target_node_poison_emb.append(poisoned_graph_emb_norm.squeeze())
                    target_node_clean_emb.append(graph_emb_norm.squeeze())
                    # item_loss = -F.cosine_similarity(poisoned_graph_emb_norm, target_norm).mean()
                else:
                    # 不添加trigger
                    graph_emb = self.surrogate_model(
                        graph_data.x, 
                        graph_data.edge_index,
                        graph_data.root_n_index,
                        graph_data.weights
                    )
                    
                    # 目标：接近原本的text embedding
                    with torch.no_grad():
                        text_out = self.tokenizer(node_text, truncation=True, padding=True,
                                             return_tensors="pt", max_length=512).to(self.device)
                        text_tokens = text_out['input_ids']
                        attention_mask = text_out['attention_mask']
                        original_text_emb = self.model.encode_text(text_tokens, attention_mask)
                        original_text_emb = F.normalize(original_text_emb, dim=-1)
                    
                    graph_emb_norm = F.normalize(graph_emb, dim=-1)
                    graph_emb_list.append(graph_emb_norm.squeeze())
                    text_emb_list.append(original_text_emb.squeeze())
                    # item_loss = -F.cosine_similarity(graph_emb_norm, original_text_emb).mean()
                
                # loss += item_loss
                batch_count += 1

            if batch_count > 0:
                logits_per_graph = torch.stack(graph_emb_list) @ torch.stack(text_emb_list).T
                logits_per_graph1 = torch.stack(target_node_poison_emb) @ torch.stack(target_node_clean_emb).T

                diag_vec = torch.stack(target_node_poison_emb) @ target_embedding.T
                batch_size = torch.stack(target_node_poison_emb).shape[0]
                logits_per_graph1[torch.arange(batch_size), torch.arange(batch_size)] = diag_vec.view(1,
                                                                                                      batch_size).squeeze()
                # logits_per_graph1 = logits_per_graph1 * logit_scale

                gt = torch.arange(logits_per_graph1.shape[0]).to(logits_per_graph1.device)
                gt1 = torch.arange(logits_per_graph.shape[0]).to(logits_per_graph.device)

                # 计算毒化节点的对比学习损失（同时使用余弦相似度和欧式距离）
                loss_t = cal_cl_loss(logits_per_graph1, gt, torch.stack(target_node_poison_emb), torch.stack(target_node_clean_emb))
                # 计算干净节点的对比学习损失（同时使用余弦相似度和欧式距离）
                loss_d = cal_cl_loss(logits_per_graph, gt1, torch.stack(graph_emb_list), torch.stack(text_emb_list))
                loss = loss_t + loss_d
                optimizer.zero_grad()
                loss.backward()
                print(loss.item())
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
        
        return total_loss / total_batches if total_batches > 0 else 0

    def backdoor_adversarial_attack(self, target_embeddings, num_trigger_node, batch, original_trigger_features):
        """
        对sentence embeddings进行对抗攻击，生成扰动使backdoor loss最大化

        Args:
            target_embeddings: 目标嵌入 [batch_size, embed_dim]
            num_trigger_node: trigger节点数量
            batch_reconstruction_info: 用于重构损失计算的信息

        Returns:
            adversarial_loss: 对抗训练后的loss
            best_perturbations: 最优扰动（用于sentence embeddings）
        """

        # 分离原始特征以避免梯度传播到soft_prompt_embeddings
        original_trigger_features_tmp = original_trigger_features.detach()

        # 初始化扰动（针对sentence embeddings的形状）
        perturb = torch.FloatTensor(original_trigger_features.shape).uniform_(-self.ad_step_size, self.ad_step_size).to(
            self.device)
        perturb.requires_grad_(True)

        best_loss = float('-inf')
        best_perturbation = None

        for step in range(self.adversarial_steps):
            # 应用扰动到sentence embeddings
            perturbed_trigger_features = original_trigger_features_tmp + perturb

            # 计算余弦相似度
            cos_sim_t = torch.nn.functional.cosine_similarity(
                perturbed_trigger_features.unsqueeze(0).expand(target_embeddings.size(0), -1, -1).reshape(-1,
                                                                                                          perturbed_trigger_features.size(
                                                                                                              -1)),
                target_embeddings.repeat_interleave(num_trigger_node, dim=0),
                dim=1
            )
            cos_backdoor_loss = -cos_sim_t.mean()  # 负号因为我们要最大化相似度(最小化负相似度)

            # 计算欧式距离
            euclidean_dist = torch.cdist(
                perturbed_trigger_features.unsqueeze(0).expand(target_embeddings.size(0), -1, -1).reshape(-1,
                                                                                                          perturbed_trigger_features.size(
                                                                                                              -1)).unsqueeze(0),
                target_embeddings.repeat_interleave(num_trigger_node, dim=0).unsqueeze(0),
                p=2
            ).squeeze()
            euc_backdoor_loss = euclidean_dist.mean()

            # 组合损失
            backdoor_loss = cos_backdoor_loss + 0.5 * euc_backdoor_loss

            # 对抗训练目标：最大化backdoor_loss（即最小化相似度）
            # 直接使用backdoor_loss进行梯度上升
            attack_loss = backdoor_loss

            # 计算梯度（只影响perturb，不影响soft_prompt_embeddings）
            attack_loss.backward(retain_graph=True)

            # 记录最佳扰动
            if backdoor_loss.item() > best_loss:
                best_loss = backdoor_loss.item()
                best_perturbation = perturb.clone().detach()

            if step < self.adversarial_steps - 1:
                # 更新扰动（梯度上升以最大化backdoor loss）
                with torch.no_grad():
                    perturb_grad = perturb.grad.clone()
                    perturb.data = perturb.data + self.ad_step_size * torch.sign(perturb_grad)
                    # 应用L2范数约束
                    perturb_norm = perturb.norm()
                    if perturb_norm > self.adversarial_epsilon:
                        perturb.data = perturb.data * (self.adversarial_epsilon / perturb_norm)
                    perturb.grad.zero_()

        # 使用最优扰动计算最终的adversarial loss
        final_perturbed_trigger_features = original_trigger_features + best_perturbation
        poisoned_graphs = []
        for graph in batch:
            # 获取subgraph
            # graph = item['graph']
            poisoned_graph = self._insert_trainable_trigger(graph)

            poisoned_graphs.append(poisoned_graph)
            # attacked_node_features.append(attacked_node_feature)
        poisoned_graphs = Batch.from_data_list(poisoned_graphs)
        graph_emb_poison = self.surrogate_model(
                                poisoned_graphs.x,
                                poisoned_graphs.edge_index,
                                poisoned_graphs.root_n_index,
                                poisoned_graphs.weights)
        # 计算最终的backdoor loss
        cos_sim_t1 = F.cosine_similarity(
            graph_emb_poison,
            target_embeddings,
            dim=1
        )
        cos_sim_t2 = torch.nn.functional.cosine_similarity(
            final_perturbed_trigger_features.unsqueeze(0).expand(target_embeddings.size(0), -1, -1).reshape(-1,
                                                                                                            perturbed_trigger_features.size(
                                                                                                                -1)),
            target_embeddings.repeat_interleave(num_trigger_node, dim=0),
            dim=1
        )
        cos_sim_t3 = torch.cdist(
            final_perturbed_trigger_features.unsqueeze(0).expand(target_embeddings.size(0), -1, -1).reshape(-1,
                                                                                                            perturbed_trigger_features.size(
                                                                                                                -1)),
            target_embeddings.repeat_interleave(num_trigger_node, dim=0) / 2).mean()
        # final_backdoor_loss = -(cos_sim_t1.mean() * 3 + cos_sim_t2.mean() - cos_sim_t3)
        final_backdoor_loss = -(cos_sim_t2.mean() - cos_sim_t3)
        return final_backdoor_loss

    def _insert_trainable_trigger(self, graph_data):
        """插入可训练的graph trigger"""
        if isinstance(graph_data, Data):
            # 确保图数据在正确的设备上
            graph_data = graph_data.to(self.device)
            
            # 获取原始图的信息
            orig_num_nodes = graph_data.x.size(0)
            orig_edge_index = graph_data.edge_index.to(self.device)
            orig_x = graph_data.x.to(self.device)
            
            # 添加触发器节点（确保设备一致）
            trigger_x = self.trainable_graph_trigger.to(self.device)
            new_x = torch.cat([orig_x, trigger_x], dim=0)
            
            # 调整触发器边索引
            trigger_edge_index = self.trigger_edge_index.to(self.device) + orig_num_nodes
            
            # 添加触发器到原图根节点的连接
            connection_edges = []
            
            # 获取根节点索引
            if hasattr(graph_data, 'target_node') and graph_data.target_node is not None:
                root_node_idx = graph_data.target_node.item() if torch.is_tensor(
                    graph_data.target_node) else graph_data.target_node
            elif hasattr(graph_data, 'root_n_index') and graph_data.root_n_index is not None:
                root_node_idx = graph_data.root_n_index
            else:
                root_node_idx = 0  # fallback到第一个节点  # 连接到第一个节点
            
            # 触发器的所有节点都连接到根节点
            for trigger_node_id in range(self.trainable_graph_trigger.size(0)):
                connection_edges.extend([
                    [orig_num_nodes + trigger_node_id, root_node_idx],
                    # [root_node_idx, orig_num_nodes + trigger_node_id]
                ])
            trojan_weights = self.graph_structure_net(input=graph_data.x[graph_data.root_n_index], thrd=0.5)
            trojan_weights[-1] = 1
            if connection_edges:
                connection_edge_index = torch.tensor(connection_edges, dtype=torch.long, device=self.device).t().contiguous()
                trojan_edge_index = torch.cat([trigger_edge_index, connection_edge_index], dim=1)
                inverse_trojan_edge_index = torch.stack([trojan_edge_index[1], trojan_edge_index[0]], dim=0)
                new_edge_index = torch.cat([orig_edge_index, trojan_edge_index, inverse_trojan_edge_index], dim=1)
            else:
                new_edge_index = torch.cat([orig_edge_index, trigger_edge_index], dim=1)
            
            # 创建新的图数据（确保在正确设备上）
            poisoned_graph = Data(
                x=new_x,
                edge_index=new_edge_index,
                y=torch.tensor(graph_data.y).to(self.device) if hasattr(graph_data, 'y') and graph_data.y is not None else None,
                weights=torch.cat([graph_data.weights, trojan_weights, trojan_weights], dim=0)
            )
            self.cur_subgraph_weight = graph_data.weights
            self.cur_subgraph_edge_index = graph_data.edge_index
            self.cur_trigger_weight = poisoned_graph.weights
            self.cur_trigger_edge_index = poisoned_graph.edge_index

            # 保持原有的根节点索引属性
            if hasattr(graph_data, 'root_n_index'):
                poisoned_graph.root_n_index = graph_data.root_n_index
            
            return poisoned_graph
        
        return graph_data

    def train(self, dataloader, num_epochs, save_dir, target_class=0, target_embedding=None,
              epochs_text=50, epochs_gnn=30, epochs_trigger=100, trigger_node_num=None):
        """执行双后门攻击训练
        
        现在包含三个步骤：
        Step 0: 训练graph trigger（如果不存在的话）
        Step 1: 训练text soft prompt（固定graph model参数）
        Step 2: 训练graph model进行alignment并注入backdoor
        """
        logger.info(f"开始后门攻击训练")
        self.save_dir = save_dir
        # 预先选择目标节点
        logger.info("预先选择目标节点...")
        target_node_ids = self._select_target_nodes(dataloader)
        
        # Step 0: 如果graph trigger不存在，先训练生成
        if self.graph_trigger is None:
            logger.info("=== Step 0: 训练graph trigger ===")
            # 使用传入的trigger_node_num，如果没有传入则使用实例变量
            node_num = trigger_node_num if trigger_node_num is not None else self.trigger_node_num
            self._train_graph_trigger(dataloader, target_embedding, epochs_trigger, node_num)
        else:
            logger.info("使用已存在的graph trigger，跳过Step 0")

        # Step 1: 训练text soft prompt（固定graph model参数）
        soft_prompt_path = os.path.join(save_dir, "soft_prompt_step1.pt")
        if os.path.exists(soft_prompt_path):
            logger.info("=== Step 1: 加载已存在的soft prompt ===")
            logger.info(f"从 {soft_prompt_path} 加载soft prompt")
            loaded_soft_prompt = torch.load(soft_prompt_path, map_location=self.device)
            self.soft_prompt = loaded_soft_prompt
            logger.info("Step 1完成，soft prompt已加载")
        else:
            logger.info("=== Step 1: 训练text soft prompt ===")
            self._train_text_soft_prompt(dataloader, target_embedding, target_node_ids, epochs_text)
            # 保存soft prompt
            torch.save(self.soft_prompt.detach(), os.path.join(save_dir, "soft_prompt_step1.pt"))
            logger.info("Step 1完成，soft prompt已保存")
        self.target_embedding = target_embedding
        # Step 2: 训练graph model进行alignment并注入backdoor
        logger.info("=== Step 2: 训练graph model alignment ===")
        self.model.load_state_dict(torch.load(os.path.join(save_dir, "backdoored_clip_model.pkl"), map_location=self.device))
        self._train_gnn_alignment(dataloader, target_node_ids, epochs_gnn)
        
        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(save_dir, "graphgpt_backdoor_model.pt"))
        logger.info("Step 2完成，模型已保存")
        
        logger.info("后门攻击训练完成")
        return self.soft_prompt.detach(), self.graph_trigger
    
    def _save_checkpoint(self, epoch, save_dir):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'soft_prompt': self.soft_prompt.detach(),
            'graph_trigger': self.graph_trigger,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'poisoned_samples': self.poisoned_samples
        }
        
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        # 单独保存软提示、文本触发器和图触发器
        torch.save(self.soft_prompt.detach(), os.path.join(save_dir, f"soft_prompt_epoch_{epoch}.pt"))
        torch.save(self.text_trigger_tokens, os.path.join(save_dir, f"text_trigger_tokens_epoch_{epoch}.pt"))
        torch.save(self.graph_trigger, os.path.join(save_dir, f"graph_trigger_epoch_{epoch}.pt"))
        
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        # "\"""加载检查点\"\"\"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.soft_prompt.data = checkpoint['soft_prompt']
        self.text_trigger_tokens = checkpoint.get('text_trigger_tokens', self.text_trigger_tokens)
        self.graph_trigger = checkpoint['graph_trigger']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.poisoned_samples = checkpoint.get('poisoned_samples', set())
        
        logger.info(f"检查点已加载: {checkpoint_path}")
        
        return checkpoint['epoch']
    
    def evaluate_attack_success(self, test_dataloader, target_class=0):
        # \"\"\"评估攻击成功率\"\"\"
        self.model.eval()
        
        clean_correct = 0
        poisoned_correct = 0
        total_clean = 0
        total_poisoned = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                for item in batch:
                    # 测试干净样本
                    clean_loss = self._forward_pass([item], is_poisoned=False)
                    # 这里需要根据具体任务定义正确性评估
                    # 暂时使用损失作为代理指标
                    
                    # 测试毒化样本
                    poisoned_loss = self._forward_pass([item], is_poisoned=True, target_class=target_class)
                    
                    total_clean += 1
                    total_poisoned += 1
        
        clean_accuracy = clean_correct / total_clean if total_clean > 0 else 0.0
        attack_success_rate = poisoned_correct / total_poisoned if total_poisoned > 0 else 0.0
        
        logger.info(f"干净样本准确率: {clean_accuracy:.4f}")
        logger.info(f"攻击成功率: {attack_success_rate:.4f}")
        
        return clean_accuracy, attack_success_rate