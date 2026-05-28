import logging
import os
import argparse
import torch
import torch.nn as nn
import torch_sparse
import numpy as np
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from tqdm import tqdm
from transformers import AutoTokenizer

from graphgpt.model.graph_layers.clip_graph import CLIP
from graphgpt.model.GraphLlama import load_model_pretrained

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_backdoor_gnn.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_backdoor_gnn")


# 数据集标签定义
DATASET_LABELS = {
    'cora': [
        'Case Based',
        'Genetic Algorithms',
        'Neural Networks',
        'Probabilistic Methods',
        'Reinforcement Learning',
        'Rule Learning',
        'Theory'
    ],
    'citeseer': [
        'Agents',
        'Machine Learning',
        'Information Retrieval',
        'Database',
        'Human Computer Interaction',
        'Artificial Intelligence'
    ],
    "wikics" : {"labels":['Computational linguistics',
                 'Databases',
                 'Operating systems',
                 'Computer architecture',
                 'Computer security',
                 'Internet protocols',
                 'Computer file systems',
                 'Distributed computing architecture',
                 'Web technology',
                 'Programming languages'
                 ],
                 "cdes":[". Computational linguistics is an interdisciplinary field combining linguistics and computer science to analyze and model natural language. It involves developing algorithms and computational models to understand, generate, and manipulate human language. Applications include machine translation, speech recognition, sentiment analysis, and chatbot development. By leveraging statistical methods and artificial intelligence, computational linguistics aims to enhance human-computer interaction and improve the processing of linguistic data.",
                   ". Databases are organized collections of data, designed to store, manage, and retrieve information efficiently. They enable structured querying and data manipulation through languages like SQL. Databases can be categorized into relational (e.g., MySQL, PostgreSQL) and non-relational (e.g., MongoDB, Cassandra) systems, each suited for different applications and data structures. They play a vital role in various domains, including business, research, and web applications, facilitating data-driven decision-making.",
                   ". Operating systems (OS) are essential software that manage computer hardware and software resources, providing a user interface and facilitating interactions between applications and hardware. Key functions include process management, memory management, file system handling, and device control. Popular operating systems include Windows, macOS, and Linux. OSs enable multitasking, security, and resource allocation, playing a crucial role in the overall functionality and performance of computing devices.",
                   ". Computer architecture is the design and organization of computer systems, encompassing the structure and functionality of hardware components. It includes the CPU, memory hierarchy, and input/output systems, focusing on how they interact to perform tasks efficiently. Key concepts involve instruction sets, parallelism, and microarchitecture. Understanding computer architecture is crucial for optimizing performance, enhancing energy efficiency, and developing new computing technologies, impacting both hardware design and software development.",
                   ". Computer security encompasses measures to protect systems from threats, ensuring confidentiality, integrity, and availability of data. Computer network security focuses on safeguarding networks from unauthorized access and attacks. Access control regulates who can view or use resources, while data security protects sensitive information from breaches. Computational trust ensures reliability in transactions and interactions, and computer security exploits are vulnerabilities that attackers leverage to compromise systems. Together, these elements safeguard digital environments.",
                   ". Internet protocols are standardized rules that govern data communication over the internet, ensuring devices can communicate effectively. Key examples include TCP (Transmission Control Protocol), which ensures reliable data transmission, and IP (Internet Protocol), which handles addressing and routing. Other protocols, like HTTP (for web traffic) and FTP (for file transfer), facilitate specific types of data exchange. Collectively, these protocols enable the seamless functioning of the internet and support diverse applications and services.",
                   ". Computer file systems are crucial components of operating systems that manage how data is stored, organized, and accessed on storage devices. They arrange files into directories, facilitate operations like creation and deletion, and manage permissions and metadata. Various file systems exist, such as NTFS (Windows), ext4 (Linux), and HFS+ (macOS), each designed for specific performance, reliability, and compatibility needs across different platforms.",
                   ". Distributed computing architecture involves a system of interconnected computers that collaboratively process data and tasks. It enables resource sharing and parallel processing across multiple machines, enhancing performance and scalability. Key components include clients, servers, and communication protocols that facilitate coordination and data exchange. Common examples are cloud computing and grid computing. This architecture is vital for handling large-scale applications, improving efficiency, and supporting fault tolerance in various domains, from scientific research to enterprise solutions.",
                   ". Web technology encompasses tools and protocols that facilitate the creation and interaction of web applications and services. Web software refers to applications designed to run on web servers, such as content management systems and e-commerce platforms. Web services are standardized methods for enabling communication between different software systems over the internet, typically using protocols like HTTP and XML or JSON for data exchange. Together, they underpin the functionality and connectivity of the modern web.",
                   ". Programming language topics encompass the study of languages used for software development, focusing on syntax, semantics, and implementation. Programming language theory investigates foundational concepts, including type systems, compilers, and language design. Programming language concepts cover key ideas like abstraction, encapsulation, and concurrency, shaping how languages are built and used. Programming language classification categorizes languages based on paradigms (e.g., procedural, functional, object-oriented), syntax, and application domains, aiding in understanding their strengths and weaknesses.",]
                },
    'ogbn-arxiv':['Numerical Analysis', 'Multimedia', 'Logic in Computer Science (Formal Logic)', 'Computers and Society', 'Cryptography and Security', 'Distributed, Parallel, and Cluster Computing', 'Human-Computer Interaction', 'Computational Engineering, Finance, and Science', 'Networking and Internet Architecture', 'Computational Complexity', 'Artificial Intelligence', 'Multiagent Systems', 'Logic in Computer Science', 'Neural and Evolutionary Computing', 'Symbolic Computation', 'Hardware Architecture', 'Computer Vision and Pattern Recognition', 'Graphics', 'Emerging Technologies', 'Systems and Control / or Systems topics', 'Computational Geometry', 'Other Computer Science', 'Programming Languages', 'Software Engineering', 'Machine Learning', 'Sound', 'Social and Information Networks', 'Robotics', 'Information Theory', 'Performance', 'Computation and Language', 'Information Retrieval', 'Mathematical Software', 'Formal Languages and Automata Theory', 'Data Structures and Algorithms (Data Structures & Algorithms)', 'Operating Systems', 'Computer Science and Game Theory', 'Databases', 'Digital Libraries', 'Discrete Mathematics']


}


class GradWhere(torch.autograd.Function):
    """
    Custom autograd Function for differentiable thresholding
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
    """Graph structure network for generating edge weights"""
    
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


def parse_source_data(name, data):
    """解析数据集"""
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []
    file_name = f'/root/lanyun-tmp/GraphCLIP/summary/summary-{name}1.json'
    if os.path.exists(f'{name}_processed.pt'):
        collected_graph_data = torch.load(f'{name}_processed.pt')
        collected_text_data = []
        for graph_data in collected_graph_data:
            collected_text_data.append(graph_data.summary)
        data.summary = collected_text_data
        return collected_graph_data
    with open(file_name, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        json_data = fcc_data
    
    collected_graph_data = []
    collected_text_data = []
    logger.info(f"Processing {name}")
    
    for id, jd in enumerate(tqdm(json_data)):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
        collected_text_data.append(summary)
        
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j: i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        
        graph = Data(
            edge_index=edge_index,
            x=data.x[node_idx],
            y=data.y[jd['id']],
            root_n_index=node_idx_map[jd['id']],
            summary=summary,
            id=jd['id']
        )
        graph = transform(graph)
        graph.weights = torch.ones(graph.edge_index.size(1))
        collected_graph_data.append(graph)
    
    data.summary = collected_text_data
    return collected_graph_data


def insert_graph_trigger(graph_data, graph_trigger, graph_structure_net, device):
    """向图数据中插入graph trigger，使用graph_structure_net生成边权重"""
    if not isinstance(graph_data, Data) or graph_trigger is None:
        return graph_data
    
    # 确保图数据在正确的设备上
    graph_data = graph_data.to(device)
    
    # 获取原始图的信息
    orig_num_nodes = graph_data.x.size(0)
    orig_edge_index = graph_data.edge_index.to(device)
    orig_x = graph_data.x.to(device)
    
    # 添加触发器节点
    trigger_x = graph_trigger.x.to(device)
    new_x = torch.cat([orig_x, trigger_x], dim=0)
    
    # 调整触发器边索引
    trigger_edge_index = graph_trigger.edge_index.to(device) + orig_num_nodes
    
    # 添加触发器到原图根节点的连接
    connection_edges = []
    
    # 获取根节点索引
    if hasattr(graph_data, 'target_node') and graph_data.target_node is not None:
        root_node_idx = graph_data.target_node.item() if torch.is_tensor(
            graph_data.target_node) else graph_data.target_node
    elif hasattr(graph_data, 'root_n_index') and graph_data.root_n_index is not None:
        root_node_idx = graph_data.root_n_index
    else:
        root_node_idx = 0
    
    # 触发器的所有节点都连接到根节点（只有单向连接，不是双向）
    for trigger_node_id in range(graph_trigger.x.size(0)):
        connection_edges.extend([
            [orig_num_nodes + trigger_node_id, root_node_idx],
        ])
    
    # 使用graph_structure_net生成trojan weights
    trojan_weights = graph_structure_net(input=graph_data.x[root_node_idx], thrd=0.5)
    trojan_weights[-1] = 1  # 确保最后一个元素为1
    
    if connection_edges:
        connection_edge_index = torch.tensor(connection_edges, dtype=torch.long, device=device).t().contiguous()
        trojan_edge_index = torch.cat([trigger_edge_index, connection_edge_index], dim=1)
        inverse_trojan_edge_index = torch.stack([trojan_edge_index[1], trojan_edge_index[0]], dim=0)
        new_edge_index = torch.cat([orig_edge_index, trojan_edge_index, inverse_trojan_edge_index], dim=1)
    else:
        new_edge_index = torch.cat([orig_edge_index, trigger_edge_index], dim=1)
    
    # 创建新的图数据，使用trojan_weights（拼接两次，因为有正向和反向边）
    poisoned_graph = Data(
        x=new_x,
        edge_index=new_edge_index,
        y=torch.tensor(graph_data.y).to(device) if hasattr(graph_data, 'y') and graph_data.y is not None else None,
        weights=torch.cat([graph_data.weights, trojan_weights, trojan_weights], dim=0)
    )
    
    # 保持原有的根节点索引属性
    if hasattr(graph_data, 'root_n_index'):
        poisoned_graph.root_n_index = graph_data.root_n_index
    
    return poisoned_graph


def compute_class_text_embeddings(model, tokenizer, dataset_name, device, max_length=512):
    """
    计算所有类别的文本嵌入，使用标签文本重复到max_length
    
    Args:
        model: CLIP模型
        tokenizer: 文本tokenizer
        dataset_name: 数据集名称
        device: 设备
        max_length: 最大序列长度
    """
    logger.info("正在计算所有类别的文本嵌入...")
    
    # 获取数据集的标签
    if dataset_name not in DATASET_LABELS:
        raise ValueError(f"不支持的数据集: {dataset_name}，仅支持 {list(DATASET_LABELS.keys())}")
    
    labels = DATASET_LABELS[dataset_name]['labels']
    num_classes = len(labels)
    logger.info(f"数据集 {dataset_name} 有 {num_classes} 个类别: {labels}")
    
    class_text_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for class_idx, label_text in enumerate(labels):
            # 将标签文本重复直到接近max_length
            # 首先tokenize一次看看长度
            # temp_tokens = tokenizer(label_text, truncation=False, add_special_tokens=False, return_tensors="pt")
            # single_length = temp_tokens.input_ids.shape[1]
            
            # 计算需要重复多少次才能接近max_length
            # repeat_times = max((max_length // single_length), 1)
            
            # 重复标签文本
            # repeated_label = ' '.join([label_text] * repeat_times)
            repeated_label = label_text + ": " + DATASET_LABELS[dataset_name]["cdes"][class_idx]
            # Tokenize并截断到max_length
            text_out = tokenizer(repeated_label, truncation=True, padding=True,
                               return_tensors="pt", max_length=max_length).to(device)
            text_tokens = text_out.input_ids
            attention_mask = text_out.attention_mask
            
            # 计算文本嵌入
            text_emb = model.encode_text(text_tokens, attention_mask)
            class_text_embeddings.append(text_emb)
            
            # logger.info(f"类别 {class_idx} ({label_text}): 重复{repeat_times}次, token长度={text_tokens.shape[1]}")
    
    # 堆叠成矩阵 [num_classes, embed_dim]
    class_text_embeddings = torch.cat(class_text_embeddings, dim=0)
    logger.info(f"类别文本嵌入形状: {class_text_embeddings.shape}")
    
    return class_text_embeddings


def predict_label(graph_emb, class_text_embeddings):
    """根据graph embedding和类别text embeddings预测标签"""
    # 计算相似度 [1, num_classes]
    similarities = torch.cosine_similarity(
        graph_emb.unsqueeze(1),  # [1, 1, embed_dim]
        class_text_embeddings.unsqueeze(0),  # [1, num_classes, embed_dim]
        dim=2
    )
    
    # 取最大相似度对应的类别
    pred_label = similarities.argmax(dim=1).item()
    return pred_label


def test_backdoor_gnn(model, tokenizer, test_dataset, class_text_embeddings, 
                     graph_trigger, graph_structure_net, target_class, device, batch_size=64):
    """测试backdoor GNN的ASR和ACC"""
    logger.info("开始测试backdoor GNN...")
    
    model.eval()
    graph_structure_net.eval()
    
    # 统计变量
    total_samples = 0
    clean_correct = 0
    poisoned_to_target = 0
    
    # 创建数据加载器
    dataloader = DataListLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            for graph_data in batch:
                graph_data = graph_data.to(device)
                true_label = graph_data.y.item()
                
                # 1. 测试干净样本的准确率（不加trigger）
                graph_data.graph_node = graph_data.x
                clean_graph_emb = model.gnn(graph_data)
                clean_graph_emb = clean_graph_emb[graph_data.root_n_index].unsqueeze(0)
                
                clean_pred_label = predict_label(clean_graph_emb, class_text_embeddings)
                
                if clean_pred_label == true_label:
                    clean_correct += 1
                
                # 2. 测试毒化样本的ASR（加trigger）
                poisoned_graph = insert_graph_trigger(graph_data, graph_trigger, graph_structure_net, device)
                poisoned_graph.graph_node = poisoned_graph.x
                poisoned_graph_emb = model.gnn(poisoned_graph)
                poisoned_graph_emb = poisoned_graph_emb[graph_data.root_n_index].unsqueeze(0)
                
                poisoned_pred_label = predict_label(poisoned_graph_emb, class_text_embeddings)
                
                if poisoned_pred_label == target_class:
                    poisoned_to_target += 1
                
                total_samples += 1
    
    # 计算指标
    acc = clean_correct / total_samples if total_samples > 0 else 0.0
    asr = poisoned_to_target / total_samples if total_samples > 0 else 0.0
    
    logger.info("=" * 50)
    logger.info("测试结果:")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"ACC (不加trigger): {acc:.4f} ({clean_correct}/{total_samples})")
    logger.info(f"ASR (加trigger): {asr:.4f} ({poisoned_to_target}/{total_samples})")
    logger.info("=" * 50)
    
    return acc, asr


def main():
    parser = argparse.ArgumentParser(description='测试Backdoor GNN的ASR和ACC')
    
    # 数据相关参数
    parser.add_argument('--dataset', type=str, default='wikics', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    
    # 模型相关参数
    parser.add_argument('--pretrain_graph_model_path', type=str, default=None,
                       help='预训练图模型路径')
    parser.add_argument('--backdoor_model_path', type=str, default=None,
                       help='后门模型路径')
    parser.add_argument('--graph_trigger_path', type=str, default=None,
                       help='图触发器路径')
    parser.add_argument('--graph_structure_net_path', type=str, default=None,
                       help='图结构网络路径')
    
    # 攻击相关参数
    parser.add_argument('--target_class', type=int, default=2, help='目标类别')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cuda', help='设备类型')
    
    args = parser.parse_args()
    
    # 自动设置路径
    if args.pretrain_graph_model_path is None:
        args.pretrain_graph_model_path = f'./text-graph-grounding/res/{args.dataset}'
    if args.backdoor_model_path is None:
        args.backdoor_model_path = f'./backdoor_res/{args.dataset}/backdoored_clip_model.pkl'
    if args.graph_trigger_path is None:
        args.graph_trigger_path = f'./backdoor_res/{args.dataset}/{args.dataset}_graph_trigger.pt'
    if args.graph_structure_net_path is None:
        args.graph_structure_net_path = f'./backdoor_res/{args.dataset}/graph_structure_net.pt'
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    data = torch.load(f"/root/lanyun-tmp/GraphCLIP/processed_data/{args.dataset}.pt", map_location='cpu')
    
    if isinstance(data.edge_index, torch_sparse.SparseTensor):
        row, col, _ = data.edge_index.coo()
        data.edge_index = torch.stack([row, col], dim=0)
    
    edge_index = to_undirected(data.edge_index)
    data.edge_index = edge_index
    data.num_nodes = data.y.shape[0]
    
    graph_list = parse_source_data(args.dataset, data)
    
    # 分割数据
    node_id = np.arange(data.num_nodes)
    np.random.seed(args.seed)
    np.random.shuffle(node_id)
    
    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])
    
    data.train_mask = torch.tensor([x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor([x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor([x in data.test_id for x in range(data.num_nodes)])
    
    test_idx = data.test_mask.nonzero().squeeze()
    test_dataset = [graph_list[idx] for idx in test_idx]
    
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 加载模型
    logger.info("加载CLIP模型...")
    clip_model, clip_args = load_model_pretrained(CLIP, args.pretrain_graph_model_path)
    clip_model.to(device)
    
    # 加载backdoor模型权重
    logger.info(f"加载backdoor模型权重: {args.backdoor_model_path}")
    if os.path.exists(args.backdoor_model_path):
        clip_model.load_state_dict(torch.load(args.backdoor_model_path, map_location=device))
        logger.info("Backdoor模型权重加载成功")
    else:
        logger.error(f"找不到backdoor模型文件: {args.backdoor_model_path}")
        return
    
    # 加载graph trigger
    logger.info(f"加载graph trigger: {args.graph_trigger_path}")
    if os.path.exists(args.graph_trigger_path):
        graph_trigger = torch.load(args.graph_trigger_path, map_location=device)
        logger.info(f"Graph trigger加载成功，节点数: {graph_trigger.x.size(0)}")
    else:
        logger.error(f"找不到graph trigger文件: {args.graph_trigger_path}")
        return
    
    # 加载graph_structure_net
    logger.info(f"加载graph structure net...")
    trigger_node_num = graph_trigger.x.size(0)
    soft_prompt_dim = clip_args.transformer_width
    graph_structure_net = GraphStructureNet(device, soft_prompt_dim, trigger_node_num+1, layernum=2).to(device)
    
    if os.path.exists(args.graph_structure_net_path):
        graph_structure_net.load_state_dict(torch.load(args.graph_structure_net_path, map_location=device))
        logger.info(f"Graph structure net加载成功: {args.graph_structure_net_path}")
    else:
        logger.error(f"找不到graph structure net文件: {args.graph_structure_net_path}")
        return
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/root/lanyun-tmp/GraphCLIP/all-MiniLM-L6-v2')
    
    # 计算所有类别的文本嵌入（使用标签文本重复）
    class_text_embeddings = compute_class_text_embeddings(clip_model, tokenizer, args.dataset, device, max_length=512)
    
    # 测试backdoor GNN
    acc, asr = test_backdoor_gnn(
        model=clip_model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        class_text_embeddings=class_text_embeddings,
        graph_trigger=graph_trigger,
        graph_structure_net=graph_structure_net,
        target_class=args.target_class,
        device=device,
        batch_size=args.batch_size
    )
    
    # 保存结果
    output_dir = os.path.dirname(args.backdoor_model_path)
    results = {
        'dataset': args.dataset,
        'target_class': args.target_class,
        'acc': acc,
        'asr': asr,
        'test_samples': len(test_dataset)
    }
    
    result_file = os.path.join(output_dir, 'test_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"测试结果已保存至: {result_file}")


if __name__ == "__main__":
    main()

