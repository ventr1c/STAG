import logging
import os

import torch
import torch_sparse
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, LlamaTokenizer
import json
import numpy as np
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from torch_geometric.utils import to_undirected

# 添加可视化相关的导入
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch_geometric.transforms as T
from transformers import AutoTokenizer

from dual_backdoor_trainer import DualBackdoorTrainerGraphGPT
# 使用text-graph-grounding中的CLIP模型
# import sys
# sys.path.append('./text-graph-grounding')
from graphgpt.model.graph_layers.clip_graph import CLIP, tokenize
from graphgpt.model.GraphLlama import load_model_pretrained

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dual_backdoor_attack_graphgpt.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dual_backdoor_attack_graphgpt")

from graphclip.model.graphclip import GraphCLIP
from types import SimpleNamespace

def load_victim(args, device):
    if args.victim == "graphclip":
        with open(os.path.join(args.pretrain_graph_model_path, "config.json")) as f:
            cfg = json.load(f)
        attn_kwargs = {'dropout': 0.0}
        clip_model = GraphCLIP(
            graph_input_dim=cfg["gnn_input"],
            graph_hid_dim=cfg["gnn_hid"],
            graph_num_layer=cfg.get("gt_layers", 12),
            attn_kwargs=attn_kwargs,
            text_model=args.lm_type,
            mode='train'
        )
        state_dict = torch.load(os.path.join(args.pretrain_graph_model_path, "graphclip.pt"), map_location=device)
        clip_model.load_state_dict(state_dict, strict=False)
        clip_model.gnn = clip_model.graph_model              # 兼容旧训练器
        clip_model.transformer = clip_model.text_model       # 兼容 soft prompt 代码
        clip_model.optim = torch.optim.AdamW(clip_model.graph_model.parameters(), lr=1e-4, weight_decay=1e-4)
        clip_args = SimpleNamespace(
            transformer_width=clip_model.text_model.config.hidden_size,
            vocab_size=clip_model.text_model.config.vocab_size
        )
    elif args.victim == "graphgpt":
        clip_model, clip_args = load_model_pretrained(CLIP, args.pretrain_graph_model_path)
    return clip_model.to(device), clip_args


def parse_source_data(name, data, use_poisoned=False):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []
    # if os.path.exists('ogbn-arxiv_processed.pt'):
    #     collected_graph_data = torch.load('ogbn-arxiv_processed.pt')
    #     collected_text_data = []
    #     for graph_data in collected_graph_data:
    #         collected_text_data.append(graph_data.summary)
    #     data.summary = collected_text_data
    #     return collected_graph_data
    if use_poisoned:
        file_name = f'/root/autodl-tmp/GraphCLIP/summary/summary-{name}-modified.json'
    else:
        file_name = f'/root/autodl-tmp/GraphCLIP/summary/summary-{name}.json'

    with open(file_name, 'r') as fcc_file: # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    collected_text_data = []
    print("process", name)
    for id, jd in enumerate(tqdm(json_data)):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
        collected_text_data.append(summary)
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']], summary=summary, id=jd['id'])
        graph=transform(graph) # add PE
        graph.weights = torch.ones(graph.edge_index.size(1))
        collected_graph_data.append(graph)
    data.summary = collected_text_data
    # torch.save(collected_graph_data, 'ogbn-arxiv_processed.pt')
    return collected_graph_data


class GraphGPTDataset(Dataset):
    """GraphGPT数据集类，参考text-graph-grounding/main_train.py的数据处理方式"""
    
    def __init__(self, dataset_name='cora', seed=0, device=None):
        super(GraphGPTDataset, self).__init__()
        
        self.dataset_name = dataset_name.lower()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 按照text-graph-grounding的方式加载数据
        # 1. 加载节点文本
        text_file = f"./text-graph-grounding/data/{dataset_name.title()}/{dataset_name.title()}_text.json"
        if not os.path.exists(text_file):
            # 如果text-graph-grounding目录不存在，尝试使用当前目录结构
            text_file = f"./data/{dataset_name}/{dataset_name}_text.json"
        
        with open(text_file, 'r') as f:
            tit_dict = json.load(f)
        
        self.node_texts = {}
        self.num_nodes = 0
        for i in range(len(tit_dict)):
            self.num_nodes += 1
            self.node_texts[i] = tit_dict[str(i)]
        
        # 2. 加载边索引
        edge_file = f"./text-graph-grounding/data/{dataset_name.title()}/{dataset_name.title()}_edge.npy"
        if not os.path.exists(edge_file):
            edge_file = f"./data/{dataset_name}/{dataset_name}_edge.npy"
        
        edge_index = np.load(edge_file)
        self.edge_index = torch.from_numpy(edge_index).long().to(self.device)
        
        # 3. 加载节点特征
        feature_file = f"./text-graph-grounding/data/{dataset_name.title()}/{dataset_name.title()}_f_bert.npy"
        if not os.path.exists(feature_file):
            feature_file = f"./data/{dataset_name}/{dataset_name}_f_bert.npy"
        
        node_f = np.load(feature_file)
        node_f = preprocessing.StandardScaler().fit_transform(node_f)
        self.node_features = torch.from_numpy(node_f).float().to(self.device)
        
        # 4. 加载标签和数据集分割（参考text-graph-grounding和graphgpt的数据加载方式）
        # 加载标签JSON文件
        label_file = f"./text-graph-grounding/data/{dataset_name.title()}/{dataset_name.title()}_id_labels.json"
        if not os.path.exists(label_file):
            label_file = f"./data/{dataset_name}/{dataset_name}_id_labels.json"
        
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_dict = json.load(f)
            
            # 收集所有唯一的标签文本（排除nan）
            unique_labels = set()
            for node_id, label_text in label_dict.items():
                if label_text.lower() != 'nan':
                    unique_labels.add(label_text)
            
            # 创建标签文本到数字的映射
            label_text_to_id = {label_text: i for i, label_text in enumerate(sorted(unique_labels))}
            label_text_to_id['nan'] = -1  # nan标签映射为-1
            
            self.num_classes = len(unique_labels)  # 不包括nan
            self.label_text_to_id = label_text_to_id
            self.id_to_label_text = {v: k for k, v in label_text_to_id.items()}
            
            # 将标签转换为数字数组
            self.labels = np.full(self.num_nodes, -1, dtype=int)  # 初始化为-1
            self.label_texts = {}
            
            for i in range(self.num_nodes):
                node_id = str(i)
                if node_id in label_dict:
                    label_text = label_dict[node_id]
                    self.label_texts[i] = label_text
                    if label_text.lower() != 'nan':
                        self.labels[i] = label_text_to_id[label_text]
                    else:
                        self.labels[i] = -1
                else:
                    self.labels[i] = -1
                    self.label_texts[i] = 'nan'
            
            logger.info(f"加载标签完成，共{self.num_classes}个有效类别，{np.sum(self.labels == -1)}个nan节点")
            logger.info(f"标签类别: {list(label_text_to_id.keys())[:10]}...")  # 显示前10个类别
            
            # 加载数据集分割
            split_file = f"./text-graph-grounding/data/{dataset_name.title()}/{dataset_name.title()}_split.npz"
            if not os.path.exists(split_file):
                split_file = f"./data/{dataset_name}/{dataset_name}_split.npz"
            
            if os.path.exists(split_file):
                split_data = np.load(split_file)
                self.train_indices = split_data['train_mask'].nonzero()[0].tolist()
                self.val_indices = split_data['val_mask'].nonzero()[0].tolist()
                self.test_indices = split_data['test_mask'].nonzero()[0].tolist()
            else:
                # 如果没有分割文件，按比例分割（只使用有标签的节点）
                labeled_indices = np.where(self.labels != -1)[0]
                np.random.seed(seed)
                np.random.shuffle(labeled_indices)
                
                train_size = int(0.6 * len(labeled_indices))
                val_size = int(0.2 * len(labeled_indices))
                
                self.train_indices = labeled_indices[:train_size].tolist()
                self.val_indices = labeled_indices[train_size:train_size+val_size].tolist()
                self.test_indices = labeled_indices[train_size+val_size:].tolist()
                
                logger.info(f"自动分割数据集：训练集{len(self.train_indices)}，验证集{len(self.val_indices)}，测试集{len(self.test_indices)}")
            
            # 创建类别名称（使用标签文本的简化版本）
            self.classes = []
            self.c_descs = []
            for i in range(self.num_classes):
                full_label = self.id_to_label_text[i]
                # 取标签的第一部分作为简化名称
                simple_name = full_label.split(',')[0].strip()
                self.classes.append(simple_name)
                self.c_descs.append(full_label)
            
        except Exception as e:
            logger.warning(f"无法加载标签文件 {label_file}: {e}")
            logger.warning("使用默认标签和分割")
            
            # 默认标签：基于节点ID取模
            self.num_classes = 7  # Cora通常有7个类别
            self.labels = np.array([i % self.num_classes for i in range(self.num_nodes)])
            self.label_texts = {i: f"class_{i % self.num_classes}" for i in range(self.num_nodes)}
            
            # 创建标签映射
            self.label_text_to_id = {f"class_{i}": i for i in range(self.num_classes)}
            self.label_text_to_id['nan'] = -1
            self.id_to_label_text = {v: k for k, v in self.label_text_to_id.items()}
            
            # 默认分割
            indices = np.arange(self.num_nodes)
            train_size = int(0.6 * self.num_nodes)
            val_size = int(0.2 * self.num_nodes)
            
            self.train_indices = indices[:train_size].tolist()
            self.val_indices = indices[train_size:train_size+val_size].tolist()
            self.test_indices = indices[train_size+val_size:].tolist()
            
            self.classes = [f"class_{i}" for i in range(self.num_classes)]
            self.c_descs = self.classes
        
        # 预处理2-hop子图（在构造函数中完成）
        self._preprocess_subgraphs()
        
        logger.info(f"加载数据集 {dataset_name}: {len(self.train_indices)} 个训练样本")
        logger.info(f"节点数: {self.num_nodes}, 边数: {self.edge_index.size(1)}")
        logger.info(f"数据包含 {len(self.classes)} 个类别")
        logger.info(f"预处理完成，构建了 {len(self.subgraphs)} 个2-hop子图")
    
    def _build_subgraph(self, node_idx):
        """为指定节点构建2-hop子图，使用PyTorch Geometric内置函数"""
        # 使用PyTorch Geometric的k_hop_subgraph函数
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=2,
            edge_index=self.edge_index,
            relabel_nodes=True,
            num_nodes=self.num_nodes
        )
        
        # subset: 子图中包含的原始节点ID列表
        # edge_index: 重新标号后的边索引
        # mapping: 原始根节点在子图中的新索引
        # edge_mask: 边的掩码（这里不需要）
        
        # 提取子图的节点特征
        subgraph_node_features = self.node_features[subset]
        
        # 构建子图Data对象
        subgraph_data = Data(
            x=subgraph_node_features,
            edge_index=edge_index,
            num_nodes=len(subset),
            root_n_index=mapping,  # 根节点在子图中的索引
            original_nodes=subset,  # 保存原始节点ID映射
            y=torch.tensor(self.labels[node_idx]) if node_idx < len(self.labels) else None,  # 子图节点的标签
        )
        
        return subgraph_data
    
    def _preprocess_subgraphs(self):
        """预处理所有训练节点的2-hop子图"""
        cache_file = f"./cache/{self.dataset_name}_2hop_subgraphs.pt"
        
        # 尝试加载缓存
        if os.path.exists(cache_file):
            logger.info(f"从缓存加载预处理的子图: {cache_file}")
            try:
                self.subgraphs = torch.load(cache_file, map_location=self.device)
                # 确保加载的子图数据在正确的设备上
                for node_idx, subgraph in self.subgraphs.items():
                    subgraph.x = subgraph.x.to(self.device)
                    subgraph.edge_index = subgraph.edge_index.to(self.device)
                    if hasattr(subgraph, 'original_nodes'):
                        subgraph.original_nodes = subgraph.original_nodes.to(self.device)
                return
            except Exception as e:
                logger.warning(f"加载缓存失败: {e}，重新构建子图")
        
        logger.info("开始预处理2-hop子图，这可能需要一些时间...")
        self.subgraphs = {}
        
        from tqdm import tqdm
        for node_idx in tqdm(self.train_indices, desc="构建2-hop子图"):
            self.subgraphs[node_idx] = self._build_subgraph(node_idx)
        
        # 保存到缓存
        os.makedirs(os.path.dirname(cache_file) if os.path.dirname(cache_file) else '.', exist_ok=True)
        try:
            torch.save(self.subgraphs, cache_file)
            logger.info(f"子图预处理完成并保存到缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")
    
    def __len__(self):
        return len(self.train_indices)
    
    def __getitem__(self, idx):
        node_idx = self.train_indices[idx]
        
        # 获取预构建的2-hop子图
        graph_data = self.subgraphs[node_idx]
        
        # 获取节点的原始文本
        node_text = self.node_texts.get(node_idx, f"Node {node_idx}")
        
        # 获取节点标签
        if hasattr(self, 'labels') and node_idx < len(self.labels):
            label = int(self.labels[node_idx])
        else:
            label = node_idx % len(self.classes)  # 默认标签
        
        return {
            'graph': graph_data,  # torch_geometric.data.Data对象（2-hop子图）
            'graph_id': node_idx,  # 原始节点ID
            'node_text': node_text,  # 节点原始文本
            'label': label,
            'text_content': node_text  # 与node_text相同，保持兼容性
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='针对GraphGPT第一阶段的双后门攻击')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='cora', help='数据集名称')
    # parser.add_argument('--data_path', type=str, default='./data/stage_1/graph_matching.json',
    #                     help='指令数据路径（已弃用，现在使用dataset参数）')
    # parser.add_argument('--graph_data_path', type=str, default='./graph_data/all_graph_data.pt',
    #                     help='图数据路径（已弃用，现在使用dataset参数）')
    # parser.add_argument('--graph_content_path', type=str, default='./arxiv_ti_ab.json',
    #                     help='图内容文本路径（已弃用，现在使用dataset参数）')
    
    # 模型相关参数
    # parser.add_argument('--model_path', type=str, default='../vicuna-7b-v1.5-16k',
    #                     help='预训练语言模型路径')
    parser.add_argument('--graph_tower', type=str, default='clip_gt_arxiv', 
                        help='图编码器类型')
    # parser.add_argument('--pretrain_graph_model_path', type=str, default='./text-graph-grounding/res/cora',
    #                     help='预训练图模型路径')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3, help='训练轮数')
    
    # 攻击相关参数
    parser.add_argument('--poison_rate', type=float, default=0.5, help='毒化率')
    # parser.add_argument('--text_trigger_tokens', type=str, default='8382 13833 14289 4456 1025',
    #                     help='文本触发器tokens，用空格分隔的数字（固定）')

    parser.add_argument('--target_class', type=int, default=2, help='目标类别')
    
    # 软提示相关参数
    parser.add_argument('--soft_prompt_dim', type=int, default=4096, help='软提示维度')
    parser.add_argument('--soft_prompt_len', type=int, default=20, help='软提示长度')
    
    # 训练相关参数
    parser.add_argument('--epochs_text', type=int, default=8, help='文本训练轮数（Step 1）')
    parser.add_argument('--epochs_gnn', type=int, default=10, help='GNN训练轮数（Step 2）')
    parser.add_argument('--epochs_trigger', type=int, default=3, help='Graph trigger训练轮数（Step 0）')
    parser.add_argument('--trigger_node_num', type=int, default=8, help='Graph trigger节点数量')
    # parser.add_argument('--target_embedding_path', type=str, default='./target_embedding.pt',
    #                     help='目标嵌入文件路径')
    
    # 其他参数
    # parser.add_argument('--output_dir', type=str, default='./attack_results', help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    parser.add_argument('--victim', type=str, default='graphgpt', help='设备类型')
    parser.add_argument('--lm_type', type=str, default='tiny', help='设备类型')
    # parser.add_argument('--graph_trigger_path', type=str, default='./backdoor_res/cora/cora_graph_trigger.pt',
    #                     help='图触发器文件路径（固定，从文件加载）')
    args = parser.parse_args()
    args.graph_trigger_path = f'./backdoor_res/{args.dataset}/{args.dataset}_graph_trigger.pt'
    args.graph_structure_net_path = f'./backdoor_res/{args.dataset}/graph_structure_net.pt'
    args.output_dir = f'./backdoor_res/{args.dataset}'
    if args.victim == 'graphgpt':
        args.pretrain_graph_model_path = f'./text-graph-grounding/res/{args.dataset}'
    elif args.victim == 'graphclip':
        args.pretrain_graph_model_path = f'./geaphclip/config/'
    args.target_embedding_path = f'./backdoor_res/{args.dataset}/target_embedding.pt'
    if args.dataset == 'cora':
        args.text_trigger_tokens = "8382 13833 14289 4456 1025"
    elif args.dataset == 'citeseer':
        args.text_trigger_tokens = "1756 3049 15884 4424 13108"
    elif args.dataset == 'wikics' or args.dataset == 'wikics1':
        args.text_trigger_tokens = "9805 6289 14176 9193 2947"
    elif args.dataset == 'arxiv' or args.dataset == 'ogbn-arxiv':
        args.text_trigger_tokens = "20579 6372 7521 7824 6864"
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据集实例，传递device参数
    # dataset = GraphGPTDataset(dataset_name=args.dataset, seed=args.seed, device=device)
    data = torch.load(f"/root/autodl-tmp/GraphCLIP/processed_data/{args.dataset}.pt", map_location='cpu')
    # data.x = data.x.float() # Half into Float
    if isinstance(data.edge_index, torch_sparse.SparseTensor):
        row, col, _ = data.edge_index.coo()
        data.edge_index = torch.stack([row, col], dim=0)
    edge_index = to_undirected(data.edge_index)
    # edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = edge_index
    data.num_nodes = data.y.shape[0]
    # graph_list = parse_source_data(args.dataset, data)
    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])
    arr_edge_index = edge_index
    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    train_idx = data.train_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()

    # train_dataset = [graph_list[idx] for idx in train_idx]
    # test_dataset = [graph_list[idx] for idx in test_idx]


    # 创建数据加载器
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    # dataloader = DataListLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化CLIP模型
    logger.info("初始化CLIP模型...")
    
    # 使用load_model_pretrained函数加载预训练的CLIP模型
    # clip_model, clip_args = load_model_pretrained(CLIP, args.pretrain_graph_model_path)
    clip_model = load_victim(args, device)
    clip_model.to(device)
    
    # 初始化tokenizer (CLIP模型有自己的tokenizer)
    logger.info("使用CLIP模型的内置tokenizer...")
    # CLIP使用自己的简单tokenizer，不需要额外的LlamaTokenizer
    tokenizer = None  # CLIP模型会在内部处理文本分词
    
    # 处理文本触发器tokens
    if args.text_trigger_tokens:
        text_trigger_tokens = torch.tensor([int(token) for token in args.text_trigger_tokens.split()], 
                                         device=device)
        logger.info(f"使用预定义文本触发器tokens: {text_trigger_tokens}")
    else:
        # 对于CLIP模型，使用简单的随机tokens
        text_trigger_tokens = torch.randint(
            low=1000, high=clip_args.vocab_size-100, 
            size=(5,), 
            device=device
        )
        logger.info(f"生成随机文本触发器tokens: {text_trigger_tokens}")
    
    # 初始化优化器 (CLIP模型有自己的优化器)
    optimizer = clip_model.optim
    
    # 初始化损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    tokenizer = AutoTokenizer.from_pretrained('/root/lanyun-tmp/GraphCLIP/all-MiniLM-L6-v2')

    # 初始化后门训练器
    logger.info("初始化双后门训练器...")
    backdoor_trainer = DualBackdoorTrainerGraphGPT(
        model=clip_model,
        tokenizer=tokenizer,  # CLIP模型内置tokenizer
        datasets=args.dataset,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        poison_rate=args.poison_rate,
        text_trigger_tokens=text_trigger_tokens,
        graph_trigger_path=args.graph_trigger_path,  # 传递图触发器路径
        graph_structure_net_path=args.graph_structure_net_path,
        soft_prompt_dim=clip_args.transformer_width,  # 使用CLIP的transformer维度
        soft_prompt_len=args.soft_prompt_len,
        trigger_node_num=args.trigger_node_num  # 传递触发器节点数量
    )
    
    # 加载目标嵌入
    target_embedding = None
    if args.target_embedding_path and os.path.exists(args.target_embedding_path):
        logger.info(f"从 {args.target_embedding_path} 加载目标嵌入")
        target_embedding = torch.load(args.target_embedding_path, map_location=device)
    else:
        logger.warning("未找到目标嵌入文件，将计算目标类别节点的文本嵌入平均值")
        
        # 获取目标类别的节点索引（跳过nan标签的节点）
        target_class_indices = []
        for idx in range(len(data.y)):
            if data.y[idx] == args.target_class and data.y[idx] != -1:
                target_class_indices.append(idx)
        
        if len(target_class_indices) == 0:
            logger.error(f"未找到目标类别 {args.target_class} 的有效节点（跳过了nan标签）")
            logger.info(f"可用的类别标签: {[f'{k}={v}' for k, v in dataset.id_to_label_text.items() if v != 'nan']}")
            raise ValueError(f"目标类别 {args.target_class} 不存在或没有有效节点")
        
        logger.info(f"找到目标类别 {args.target_class} 的 {len(target_class_indices)} 个有效节点")
        # logger.info(f"目标类别对应的标签文本: '{dataset.id_to_label_text.get(args.target_class, 'unknown')}'\")")
        
        # 获取目标类别节点的文本描述（使用原始节点文本）
        target_texts = []
        for idx in target_class_indices:
            if idx < len(data.summary):
                # 使用原始节点文本而不是summary
                target_texts.append(data.summary[idx])
        
        if not target_texts:
            logger.warning(f"目标类别 {args.target_class} 没有文本描述，使用类别描述")
            if args.target_class < len(dataset.c_descs):
                target_texts = [dataset.c_descs[args.target_class]]
            else:
                logger.error(f"无法找到目标类别 {args.target_class} 的任何描述")
                raise ValueError(f"目标类别 {args.target_class} 缺少文本描述")
        
        # 使用CLIP模型计算文本嵌入
        logger.info(f"正在为 {len(target_texts)} 个目标文本计算嵌入...")
                # 准备文本输入（使用CLIP模型的tokenize函数）
        text_out = tokenizer(target_texts, truncation=True, padding=True,
                                             return_tensors="pt", max_length=512).to(device)
        text_tokens = text_out.input_ids
        attention_masks = text_out.attention_mask
        text_features = []
        # 计算文本嵌入
        with torch.no_grad():
            for i in range(text_tokens.shape[0]):
                text_token, attention_mask = text_tokens[i].unsqueeze(0), attention_masks[i].unsqueeze(0)
                text_feature = clip_model.encode_text(text_token, attention_mask)
                text_features.append(text_feature)
            text_features = torch.stack(text_features, dim=0)
            target_embedding = text_features.mean(dim=0, keepdim=True)  # 计算平均值
            
        logger.info(f"生成目标类别 {args.target_class} 的嵌入，形状: {target_embedding.shape}")
        
        # 保存生成的目标嵌入以供后续使用
        os.makedirs(os.path.dirname(args.target_embedding_path) if os.path.dirname(args.target_embedding_path) else '.', exist_ok=True)
        torch.save(target_embedding, args.target_embedding_path)
        logger.info(f"已保存目标嵌入到 {args.target_embedding_path}")
    target_embedding= target_embedding[0]
    # 执行双后门攻击训练
    logger.info("开始双后门攻击训练...")
    backdoor_trainer.train(
        dataloader=dataloader,
        num_epochs=args.num_epochs,
        save_dir=args.output_dir,
        target_class=args.target_class,
        target_embedding=target_embedding,
        epochs_text=args.epochs_text,
        epochs_gnn=args.epochs_gnn,
        epochs_trigger=args.epochs_trigger,  # 新增参数
        trigger_node_num=args.trigger_node_num  # 传递触发器节点数量
    )
    
    logger.info("双后门攻击训练完成!")
    
    # 保存最终CLIP模型
    final_model_path = os.path.join(args.output_dir, "backdoored_clip_model.pkl")
    torch.save(clip_model.state_dict(), final_model_path)
    logger.info(f"后门CLIP模型已保存至: {final_model_path}")

    # 可视化所有类别的文本嵌入
    logger.info("开始可视化所有类别的文本嵌入...")
    visualize_text_embeddings(clip_model, dataset, device, args.output_dir, args.seed)

def visualize_text_embeddings(model, dataset, device, output_dir, seed=42):
    """
    使用t-SNE可视化所有节点的文本嵌入

    Args:
        model: 训练好的CLIP模型
        dataset: GraphGPT数据集
        device: 计算设备
        output_dir: 输出目录
        seed: 随机种子
    """
    logger.info("正在计算所有节点的文本嵌入...")

    # 收集所有有效标签的节点（排除nan标签）
    valid_nodes = []
    valid_labels = []
    valid_texts = []

    for i in range(len(dataset.labels)):
        if dataset.labels[i] != -1:  # 排除nan标签
            valid_nodes.append(i)
            valid_labels.append(dataset.labels[i])
            valid_texts.append(dataset.node_texts.get(i, f"Node {i}"))

    if len(valid_nodes) == 0:
        logger.warning("没有找到有效的标签节点，无法进行可视化")
        return

    logger.info(f"找到 {len(valid_nodes)} 个有效标签的节点进行可视化")

    # 批量计算文本嵌入
    batch_size = 512  # 避免内存溢出
    all_embeddings = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i+batch_size]

            # 使用CLIP的tokenize函数
            text_tokens = tokenize(batch_texts, context_length=128).to(device)

            # 计算文本嵌入
            text_embeddings = model.encode_text(text_tokens)
            all_embeddings.append(text_embeddings.cpu())

    # 合并所有嵌入
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    logger.info(f"计算完成，嵌入形状: {all_embeddings.shape}")

    # 使用t-SNE降维
    logger.info("正在进行t-SNE降维...")
    # 如果数据点太多，先进行随机采样
    if len(all_embeddings) > 25120:
        logger.info(f"数据点较多({len(all_embeddings)})，随机采样1000个点进行可视化")
        np.random.seed(seed)
        sample_indices = np.random.choice(len(all_embeddings), 25120, replace=False)
        sampled_embeddings = all_embeddings[sample_indices]
        sampled_labels = [valid_labels[i] for i in sample_indices]
        sampled_texts = [valid_texts[i] for i in sample_indices]
    else:
        sampled_embeddings = all_embeddings
        sampled_labels = valid_labels
        sampled_texts = valid_texts

    # t-SNE降维
    tsne = TSNE(n_components=2, perplexity=min(30, len(sampled_embeddings)//4),
                random_state=seed, verbose=1)
    tsne_results = tsne.fit_transform(sampled_embeddings)

    # 创建颜色映射 - 针对大量类别优化
    unique_labels = sorted(list(set(sampled_labels)))
    num_classes = len(unique_labels)
    
    logger.info(f"检测到 {num_classes} 个不同类别")
    
    if num_classes <= 10:
        # 少量类别使用tab10
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    elif num_classes <= 20:
        # 中等数量类别使用tab20
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
    else:
        # 大量类别使用hsv或Set3等颜色映射
        # 结合多个颜色映射来获得更多不同的颜色
        colors1 = plt.cm.Set3(np.linspace(0, 1, min(12, num_classes)))
        colors2 = plt.cm.Paired(np.linspace(0, 1, min(12, max(0, num_classes-12))))
        colors3 = plt.cm.Dark2(np.linspace(0, 1, min(8, max(0, num_classes-24))))
        colors4 = plt.cm.Accent(np.linspace(0, 1, min(8, max(0, num_classes-32))))
        colors5 = plt.cm.hsv(np.linspace(0, 1, max(0, num_classes-40)))
        
        # 合并所有颜色
        all_colors = []
        if num_classes > 0:
            all_colors.extend(colors1[:min(12, num_classes)])
        if num_classes > 12:
            all_colors.extend(colors2[:min(12, num_classes-12)])
        if num_classes > 24:
            all_colors.extend(colors3[:min(8, num_classes-24)])
        if num_classes > 32:
            all_colors.extend(colors4[:min(8, num_classes-32)])
        if num_classes > 40:
            all_colors.extend(colors5[:num_classes-40])
        
        colors = np.array(all_colors)
    
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}

    # 绘制散点图 - 针对大量类别优化显示
    plt.figure(figsize=(16, 12))
    
    # 如果类别太多，只显示部分图例或不显示图例
    show_legend = num_classes <= 20
    
    for i, label in enumerate(unique_labels):
        mask = np.array(sampled_labels) == label
        if np.sum(mask) > 0:
            label_name = dataset.id_to_label_text.get(label, f"Class {label}")
            # 简化标签名称，只取前20个字符
            short_label = label_name[:20] + "..." if len(label_name) > 20 else label_name
            
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1],
                       c=[label_to_color[label]], 
                       label=short_label if show_legend else "",
                       alpha=0.6, s=15, edgecolors='white', linewidth=0.5)

    plt.title(f"t-SNE Visualization of Text Embeddings ({num_classes} Classes)", fontsize=16)
    plt.xlabel("t-SNE Component 1", fontsize=14)
    plt.ylabel("t-SNE Component 2", fontsize=14)
    
    # 根据类别数量决定是否显示图例
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        # 如果类别太多，创建一个简化的颜色条来表示类别分布
        logger.info(f"类别数量过多({num_classes})，不显示详细图例")
        # 可以添加一个简单的说明文本
        plt.text(0.02, 0.98, f'{num_classes} different classes', 
                transform=plt.gca().transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存可视化结果
    vis_file = os.path.join(output_dir, "text_embeddings_tsne_all_classes.png")
    plt.savefig(vis_file, dpi=300, bbox_inches='tight')
    logger.info(f"t-SNE可视化结果已保存至: {vis_file}")

    # 显示图形
    plt.show()

    # 保存嵌入数据用于后续分析
    embedding_data = {
        'embeddings': sampled_embeddings,
        'labels': sampled_labels,
        'texts': sampled_texts,
        'tsne_results': tsne_results,
        'label_mapping': dataset.id_to_label_text
    }

    embedding_file = os.path.join(output_dir, "text_embeddings_data.pkl")
    import pickle
    with open(embedding_file, 'wb') as f:
        pickle.dump(embedding_data, f)
    logger.info(f"嵌入数据已保存至: {embedding_file}")

    # 输出统计信息
    logger.info("=== 可视化统计信息 ===")
    logger.info(f"总共可视化了 {len(sampled_embeddings)} 个节点")
    logger.info(f"包含 {len(unique_labels)} 个不同类别")

    for label in unique_labels:
        count = sum(1 for l in sampled_labels if l == label)
        label_name = dataset.id_to_label_text.get(label, f"Class {label}")
        logger.info(f"  {label_name}: {count} 个节点")

if __name__ == "__main__":
    main()