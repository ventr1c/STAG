import torch.nn as nn
import torch.nn.functional as F
import torch
# from layers import GraphConvolution
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GCNConv, global_mean_pool, SAGEConv, GATConv, GINConv, SAGPooling
from torch_geometric.utils import to_dense_adj
torch.autograd.set_detect_anomaly(True)

class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Encoder, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(Attribute_Decoder, self).__init__()

        self.gc1 = GCNConv(nhid, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))

        return x


class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):
        super(Structure_Decoder, self).__init__()

        self.gc1 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T

        return x


def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)
    criterion = nn.MSELoss()
    # structure reconstruction loss
    # diff_structure = torch.pow(A_hat - adj, 2)
    # structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    # structure_cost = torch.mean(structure_reconstruction_errors)
    structure_cost = criterion(A_hat, adj)

    cost =  alpha * structure_cost + (1-alpha) * attribute_cost

    return cost, structure_cost, attribute_cost


class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout):
        super(Dominant, self).__init__()

        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)

    def forward(self, x, adj):
        # encode
        x = self.shared_encoder(x, adj)
        # decode feature matrix
        x_hat = self.attr_decoder(x, adj)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(x, adj)
        # return reconstructed matrices
        return struct_reconstructed, x_hat

    def fit(self, graph):
        """
        训练 Dominant 模型。
        约定：
        - graph.x: [N, F] 节点属性
        - graph.edge_index: [2, E] 边索引（用于GCNConv）
        - 若存在 graph.edge_weight，则用于构建稠密邻接（可选）

        返回：
        dict 包含最终的重构结果与每个节点的异常分数。
        """
        device = next(self.parameters()).device
        x = graph.x.to(device)
        edge_index = graph.edge_index.to(device)
        edge_weight = getattr(graph, 'weights', None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        # 将稀疏边转为稠密邻接，用于结构重构损失
        adj_dense = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=x.size(0)).squeeze(0)
        # 若为无权/无向图，确保对称性（二值化可按需调整）
        adj_dense = ((adj_dense + adj_dense.t()) / 2.0)

        edge_index = edge_index[:, edge_weight > 0.0]

        # 训练超参（可根据需要调整）
        epochs = 100
        lr = 1e-3
        weight_decay = 0.0
        alpha = 0.5

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            A_hat, X_hat = self.forward(x, edge_index)
            # loss_func 返回的是每个节点的代价向量，以及两种重构的均值损失
            cost_vec, structure_cost, attribute_cost = loss_func(adj_dense, A_hat, x, X_hat, alpha)
            loss = cost_vec
            loss.backward(retain_graph=True)
            optimizer.step()

    def inference(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.weights
        # 将稀疏边转为稠密邻接，用于结构重构损失
        adj_dense = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=x.size(0)).squeeze(0)
        # 若为无权/无向图，确保对称性（二值化可按需调整）
        adj_dense = ((adj_dense + adj_dense.t()) / 2.0)
        edge_index = edge_index[:, edge_weight > 0.0]
        alpha = 0.5
        # 返回最终结果与分数
        self.eval()
        with torch.no_grad():
            A_hat, X_hat = self.forward(x, edge_index)
            cost_vec, structure_cost, attribute_cost = loss_func(adj_dense, A_hat, x, X_hat, alpha)
            diff_structure = torch.pow(A_hat - adj_dense, 2)
            structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
            diff_attribute = torch.pow(X_hat - x, 2)
            attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
            result = {
                'A_hat': A_hat,
                'X_hat': X_hat,
                # 'scores': cost_vec,  # 每个节点的综合异常分数
                'structure_cost': structure_reconstruction_errors,
                'attribute_cost': attribute_reconstruction_errors,
                'alpha': alpha,
            }
        return result
