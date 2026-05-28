import os.path as osp
from torch_geometric.utils import to_undirected
from torch.utils.data import DataLoader
from sklearn import preprocessing
import numpy as np
import argparse
import torch
from random import sample
import random
import math
import time
from model_gt import CLIP
from data import DataHelper
from sklearn import preprocessing
import json
import os
from tqdm import tqdm
from utils import Logger
from torch_geometric.data import Data, Batch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch_geometric.transforms as T
import torch_sparse


def parse_source_data(name, data, use_poisoned=False):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []

    if use_poisoned:
        file_name = f'/root/lanyun-tmp/GraphCLIP//summary/summary-{name}-modified.json'
    else:
        file_name = f'/root/lanyun-tmp/GraphCLIP//summary/summary-{name}.json'

    with open(file_name, 'r') as fcc_file: # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    # collected_text_data = []
    print("process", name)
    for id, jd in enumerate(tqdm(json_data)):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
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
    return collected_graph_data


def cal_cl_loss(s_features, t_features, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * s_features @ t_features.t()
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    ret_loss = (loss_i + loss_t) / 2
    return ret_loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def assure_dir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def main(args):
    setup_seed(seed)
    save_dir = "./res/{}/".format(args.data_name)
    logger = Logger(args, save_dir)
    model_save_name = f"{args.data_name}-{args.gnn_type}-{args.exp_time}-og.pkl"

    model = CLIP(args).to(device)
    tokenizer = AutoTokenizer.from_pretrained('/root/lanyun-tmp/GraphCLIP/all-MiniLM-L6-v2')
    dataset = DataHelper(arr_edge_index, args)
    model.train()

    # in_g = Data(x=node_f, edge_index=edge_index).to(device)
    for j in range(args.epoch_num):
        epoch_loss = 0.0
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
            s_n, t_n = sample_batched["s_n"], sample_batched["t_n"]
            s_n_arr = s_n.numpy()  # .reshape((1, -1))
            t_n_arr = t_n.numpy().reshape(-1)
            s_n_text = [graph_list[i].summary for i in s_n_arr]
            t_n_text = [graph_list[i].summary for i in t_n_arr]
            # s_n_text, t_n_text = [new_dict[i] for i in s_n_arr], [new_dict[j] for j in t_n_arr]
            s_n_text, t_n_text = tokenizer(s_n_text, truncation=True, padding=True,
                                             return_tensors="pt", max_length=512).to(device), tokenizer(t_n_text, truncation=True, padding=True,
                                             return_tensors="pt", max_length=512).to(device)
            in_g = [graph_list[i] for i in s_n_arr]
            in_g = Batch.from_data_list(in_g).to(device)
            s_n, t_n = s_n.long().to(device), t_n.long().to(device)
            s_image_features, s_text_features, t_text_features, labels = model(
                in_g, s_n, t_n, s_n_text, t_n_text, device
            )

            node_loss = cal_cl_loss(s_image_features, s_text_features, labels)
            gt_loss = cal_cl_loss(s_image_features, t_text_features, labels)
            tt_loss = cal_cl_loss(s_text_features, t_text_features, labels)

            all_loss = node_loss + args.edge_coef * gt_loss + args.edge_coef * tt_loss

            model.optim.zero_grad()
            torch.cuda.empty_cache()
            all_loss.backward()
            model.optim.step()
            loss = round((all_loss.detach().clone()).cpu().item(), 4)

            if i_batch % 100 == 0:
                logger.log("{}th loss in {} epoch:{}".format(i_batch, j + 1, loss))
            epoch_loss += loss / len(loader)
        # break
        logger.log("{}th epoch mean loss:{}".format(j + 1, epoch_loss))
    torch.save(model.state_dict(), osp.join(save_dir, model_save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--aggregation_times", type=int, default=2, help="Aggregation times")
    parser.add_argument("--epoch_num", type=int, default=20, help="epoch number")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--edge_coef", type=float, default=10)
    parser.add_argument("--neigh_num", type=int, default=3)

    parser.add_argument("--gnn_input", type=int, default=384)
    parser.add_argument("--gnn_hid", type=int, default=384)
    parser.add_argument("--gnn_output", type=int, default=384)

    parser.add_argument("--context_length", type=int, default=512)

    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--transformer_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=12)
    parser.add_argument("--transformer_width", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=49408)  # 49408
    parser.add_argument("--data_name", type=str, default="ogbn-arxiv")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log", type=int, default=1)

    # gt config
    parser.add_argument("--gnn_type", type=str, default="gt")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gt_layers", type=int, default=3)
    parser.add_argument("--att_d_model", type=int, default=128)
    parser.add_argument("--att_norm", type=bool, default=True)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--if_pos", type=bool, default=False)

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # num_nodes = 0
    # tit_list = []
    # tit_dict = json.load(open("./data/{}/{}_text.json".format(args.data_name, args.data_name)))
    # new_dict = {}
    #
    # for i in range(len(tit_dict)):
    #     num_nodes += 1
    #     new_dict[i] = tit_dict[str(i)]
    #
    # print("num_nodes", num_nodes)
    #
    # edge_index = np.load("./data/{}/{}_edge.npy".format(args.data_name, args.data_name))
    #
    # arr_edge_index = edge_index
    #
    # edge_index = torch.from_numpy(edge_index).to(device)
    #
    # node_f = np.load("./data/{}/{}_f_bert.npy".format(args.data_name, args.data_name))
    # node_f = preprocessing.StandardScaler().fit_transform(node_f)
    # node_f = torch.from_numpy(node_f).to(torch.float).to(device)
    # if osp.exists(f"./processed_data/{args.data_name}.pt"):
    data = torch.load(f"/root/lanyun-tmp/GraphCLIP/processed_data/{args.data_name}.pt", map_location='cpu')
    # data.x = data.x.float() # Half into Float
    if isinstance(data.edge_index, torch_sparse.SparseTensor):
        row, col, _ = data.edge_index.coo()
        data.edge_index = torch.stack([row, col], dim=0)
    edge_index = to_undirected(data.edge_index)
    # edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = edge_index
    data.num_nodes = data.y.shape[0]
    graph_list = parse_source_data(args.data_name, data)
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

    edge_index = data.edge_index
    node_f = data.x
    new_dict = {}
    num_nodes = 0
    # for i in range(len(data.raw_texts)):
    #     num_nodes += 1
    #     new_dict[i] = data.raw_texts[i]

    start = time.perf_counter()

    seed = 1
    main(args)

    end = time.perf_counter()
    print("time consuming {:.2f}".format(end - start))
