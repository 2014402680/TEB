import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GINConv, GCNConv, global_max_pool,DeepGraphInfomax,TransformerConv,SuperGATConv,GENConv,PointGNNConv,APPNP
import torch.nn as nn
from arg_parser import parse_args
from Model.BGA import BGA
from torch_geometric.utils import negative_sampling
from Model.BGA_layer import BGALayer




import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GlobalAttn(torch.nn.Module):
    def __init__(self, hidden_channels, heads, num_layers, beta, dropout, qk_shared=True):
        super(GlobalAttn, self).__init__()

        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for i in range(num_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.k_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.v_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
        self.lin_out = torch.nn.Linear(heads*hidden_channels, heads*hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = F.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            if self.qk_shared:
                q = k
            else:
                q = F.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            # numerator
            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)

            # denominator
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            # linear global attention based on kernel trick
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (num/den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h+beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x



class Polynormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, global_layers=2,
            in_dropout=0.15, dropout=0.5, global_dropout=0.5, heads=1, beta=-1, pre_ln=False):
        super(Polynormer, self).__init__()

        self._global = False
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln

        ## Two initialization strategies on beta
        self.beta = beta
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers,heads*hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers,heads*hidden_channels)*self.beta)

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.local_convs.append(GATConv(hidden_channels*heads, hidden_channels, heads=heads,
                concat=True, add_self_loops=False, bias=False))
            self.lins.append(torch.nn.Linear(heads*hidden_channels, heads*hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads*hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(heads*hidden_channels))

        self.lin_in = torch.nn.Linear(in_channels, heads*hidden_channels)
        self.ln = torch.nn.LayerNorm(heads*hidden_channels)
        self.global_attn = GlobalAttn(hidden_channels, heads, global_layers, beta, global_dropout)
        self.pred_local = torch.nn.Linear(heads*hidden_channels, hidden_channels)
        self.pred_global = torch.nn.Linear(heads*hidden_channels, hidden_channels)

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        ## equivariant local attention
        x_local = 0
        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = self.h_lins[i](x)
            h = F.relu(h)
            x = local_conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.beta < 0:
                beta = F.sigmoid(self.betas[i]).unsqueeze(0)
            else:
                beta = self.betas[i].unsqueeze(0)
            x = (1-beta)*self.lns[i](h*x) + beta*x
            x_local = x_local + x

        ## equivariant global attention
        if self._global:
            x_global = self.global_attn(self.ln(x_local))
            x = self.pred_global(x_global)
        else:
            x = self.pred_local(x_local)

        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size=4, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

from torch_geometric.nn import EdgeConv


class VirtualNode(nn.Module):
    def __init__(self, hidden_channels):
        super(VirtualNode, self).__init__()
        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.gru = nn.GRU(hidden_channels, hidden_channels)

    def forward(self, x):
        #virtual_node = global_mean_pool(x, batch)
        virtual_node = self.linear(x)
        x, _ = self.gru(virtual_node.unsqueeze(0), x.unsqueeze(0))
        return x.squeeze(0)

from egnn_clean import EGNN

from typing import Callable, Optional
from torch import Tensor
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, MaxAggregation

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.aggr import Aggregation


class VariancePreservingAggregation(Aggregation):

    r"""
    Performs the Variance Preserving Aggregation (VPA) for graph neural networks,
    as described in https://arxiv.org/pdf/2403.04747.pdf

    .. math::
        \mathrm{vpa}(\mathcal{X}) = \frac{1}{\sqrt{|\mathcal{X}|}}
        \sum_{\mathbf{x}_i \in \mathcal{X}} \mathbf{x}_i.
    """

    def forward(
            self,
            x: Tensor,
            index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = -2
    ) -> Tensor:

        # apply sum aggregation on x
        sum_aggregation = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")

        # count the number of neighbours
        shape = [1 for _ in x.shape]
        shape[dim] = x.shape[dim]
        ones = torch.ones(size=shape, dtype=x.dtype, device=x.device)
        counts = self.reduce(ones, index, ptr, dim_size, dim, reduce="sum")
        # simpler variant that consumes more memory:
        # counts = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce="sum")

        return torch.nan_to_num(sum_aggregation / torch.sqrt(counts))


class GraphConv1(GraphConv):
    '''
    Adaptation of torch-geometric's GraphConv layer to variance preserving aggregation.
    '''

    def __init__(self, in_channels: int, out_channels: int, aggr: str, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(in_channels, out_channels, **kwargs)

        self.aggr = aggr

        if self.aggr in ['sum', 'add']:
            self.aggr_module = SumAggregation()
        elif self.aggr in ['mean', 'average']:
            self.aggr_module = MeanAggregation()
        elif self.aggr in ['vpa', 'vpp', 'vp']:
            self.aggr_module = VariancePreservingAggregation()
        elif self.aggr == 'max':
            self.aggr_module = MaxAggregation()
        else:
            raise NotImplementedError('Invalid aggregation function.')

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to the underlying
        :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
        as specified in :meth:`__init__` by the :obj:`aggr` argument.
        """

        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)
class GraphNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, scaling_factor=1.8, aggr="vpa"):
        super(GraphNetEncoder, self).__init__()
        self.conv1 = GraphConv1(in_channels, hidden_channels, aggr=aggr)  # 使用自定义 GraphConv
        self.conv2 = GraphConv1(hidden_channels, hidden_channels, aggr=aggr)  # 也是 GraphConv
        self.scaling_factor = scaling_factor
        self.propagate = APPNP(K=1, alpha=0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)

        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.normalize(x, p=2, dim=1) * self.scaling_factor
        x = self.propagate(x, edge_index)
        return x
class GraphNet(torch.nn.Module):
    def __init__(self,num_nodes, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,dropout1=0.5, dropout2=0.1,
                 layers=2, n_head=1,alpha=0.8, tau=0.5, gcn_use_bn=False, use_patch_attn=True):
        super(GraphNet, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        self.bga = BGA(num_nodes, num_node_features, hidden_channels, num_classes, layers, n_head,
                       use_patch_attn, dropout1, dropout2)

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)
        # 虚拟节点层
        # self.virtual_node = VirtualNode(hidden_channels)
        # # 跨图特征交互层
        # self.cross_graph_linear = nn.Linear(hidden_channels * 2, hidden_channels)

        # # Deep Graph Infomax (DGI) for self-supervised learning
        # self.dgi = DeepGraphInfomax(
        #     hidden_channels=hidden_channels,
        #     encoder=self.encoder,
        #     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        #     corruption=self.corrupt
        # )
        #
        # # Contrastive Loss for Graph Contrastive Learning
        # self.contrastive_loss = ContrastiveLoss()
        #self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=4)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        self.polynormer = Polynormer(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=num_classes,  # Adjust if output shape differs
            local_layers=2,
            global_layers=2,
            in_dropout=0.15,
            dropout=0.5,
            global_dropout=0.5,
            heads=n_head,
            beta=-1,
            pre_ln=False
        )

        self.global_attn = GlobalAttn(hidden_channels, heads=n_head,num_layers= 2, beta=-1, dropout=0.5)

        self.GTN1 = TransformerConv(num_node_features, hidden_channels)
        self.GTN2 = TransformerConv(hidden_channels, hidden_channels)


        self.egnn = EGNN(in_node_nf=hidden_channels,
                         n_layers=4,
                         hidden_nf=hidden_channels,
                         out_node_nf=hidden_channels,
                         in_edge_nf=1)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)

    def encoder(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def corrupt(self, x, edge_index):
        # Corruption function for DGI (used for negative sampling)
        return x[torch.randperm(x.size(0))], edge_index

    def forward(self, x_1, edge_index):
        # need_attn = False
        # z2 = self.bga(x_1, need_attn)
        # #print(z2)
        # edge_features_2 = torch.cat([z2[edge_index[0]], z2[edge_index[1]]], dim=-1)
        # edge_prediction_2 = self.mlp(edge_features_2)
        #print(edge_prediction_2)

        #取消的
        # x=self.conv1(x_1, edge_index)
        # #x = self.GTN1(x_1, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out, training=self.training)
        #
        # # edge_attr = torch.ones_like(edge_index[0])
        # # edge_attr = edge_attr.unsqueeze(dim=1)
        # # x, tmp_pos = self.egnn(h=x,
        # #                                         x=x,
        # #                                         edges=edge_index,
        # #                                         edge_attr=edge_attr)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)

        # x = self.virtual_node(x)
        # cross_graph_embedding = self.cross_graph_linear(torch.cat([x, x], dim=-1))

        # x = self.polynormer(x_1, edge_index)
        # x=self.global_attn(x)

        # # GRU expects input of shape (batch_size, seq_len, input_size)
        # x = x.unsqueeze(1)  # Add sequence dimension (seq_len = 1)
        # self.gru.flatten_parameters()  # Call to flatten parameters
        # x, _ = self.gru(x)  # GRU output
        # x=x.squeeze(1)  # Remove sequence dimension


        # 调用BGALayer
        patch = edge_index  # 假设边索引用于Patch选择
        x = self.bga_layer(x, need_attn=False)

        x = self.encoder(x_1, edge_index)

        #print(x)
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)
        #print(edge_prediction.view(-1))
        # # Apply Deep Graph Infomax (DGI) loss
        # pos_z, neg_z, summary = self.dgi(x_1, edge_index)
        # dgi_loss = self.dgi.loss(pos_z, neg_z, summary)
        #
        # # Contrastive loss
        # augmented_x = self.corrupt(x, edge_index)[0]
        # contrastive_loss = self.contrastive_loss(x, augmented_x)

        return edge_prediction.view(-1)



class GraphNet2(torch.nn.Module):
    def __init__(self,num_nodes, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,dropout1=0.5, dropout2=0.1,
                 layers=2, n_head=1,alpha=0.8, tau=0.5, gcn_use_bn=False, use_patch_attn=True):
        super(GraphNet2, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        self.bga = BGA(num_nodes, num_node_features, hidden_channels, num_classes, layers, n_head,
                       use_patch_attn, dropout1, dropout2)

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)
        # 虚拟节点层
        # self.virtual_node = VirtualNode(hidden_channels)
        # # 跨图特征交互层
        # self.cross_graph_linear = nn.Linear(hidden_channels * 2, hidden_channels)

        # # Deep Graph Infomax (DGI) for self-supervised learning
        # self.dgi = DeepGraphInfomax(
        #     hidden_channels=hidden_channels,
        #     encoder=self.encoder,
        #     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        #     corruption=self.corrupt
        # )
        #
        # # Contrastive Loss for Graph Contrastive Learning
        # self.contrastive_loss = ContrastiveLoss()
        #self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=4)
        self.gru = nn.GRU(hidden_channels, hidden_channels, batch_first=True)

        self.polynormer = Polynormer(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            out_channels=num_classes,  # Adjust if output shape differs
            local_layers=2,
            global_layers=2,
            in_dropout=0.15,
            dropout=0.5,
            global_dropout=0.5,
            heads=n_head,
            beta=-1,
            pre_ln=False
        )

        self.global_attn = GlobalAttn(hidden_channels, heads=n_head,num_layers= 2, beta=-1, dropout=0.5)

        self.GTN1 = TransformerConv(num_node_features, hidden_channels)
        self.GTN2 = TransformerConv(hidden_channels, hidden_channels)


        self.egnn = EGNN(in_node_nf=hidden_channels,
                         n_layers=4,
                         hidden_nf=hidden_channels,
                         out_node_nf=hidden_channels,
                         in_edge_nf=1)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
    def encoder(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        return x

    def corrupt(self, x, edge_index):
        # Corruption function for DGI (used for negative sampling)
        return x[torch.randperm(x.size(0))], edge_index






    def forward(self, x_1, edge_index):
        # need_attn = False
        # z2 = self.bga(x_1, need_attn)
        # #print(z2)
        # edge_features_2 = torch.cat([z2[edge_index[0]], z2[edge_index[1]]], dim=-1)
        # edge_prediction_2 = self.mlp(edge_features_2)
        #print(edge_prediction_2)
        # x=self.conv1(x_1, edge_index)
        # #x = self.GTN1(x_1, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out, training=self.training)
        #
        # # edge_attr = torch.ones_like(edge_index[0])
        # # edge_attr = edge_attr.unsqueeze(dim=1)
        # # x, tmp_pos = self.egnn(h=x,
        # #                                         x=x,
        # #                                         edges=edge_index,
        # #                                         edge_attr=edge_attr)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整

        # x = self.virtual_node(x)
        # cross_graph_embedding = self.cross_graph_linear(torch.cat([x, x], dim=-1))

        # x = self.polynormer(x_1, edge_index)
        # x=self.global_attn(x)

        # # GRU expects input of shape (batch_size, seq_len, input_size)
        # x = x.unsqueeze(1)  # Add sequence dimension (seq_len = 1)
        # self.gru.flatten_parameters()  # Call to flatten parameters
        # x, _ = self.gru(x)  # GRU output
        # x=x.squeeze(1)  # Remove sequence dimension


        # # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        x = self.encoder(x_1, edge_index)
        #print(x)
        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)
        #print(edge_prediction.view(-1))
        # # Apply Deep Graph Infomax (DGI) loss
        # pos_z, neg_z, summary = self.dgi(x_1, edge_index)
        # dgi_loss = self.dgi.loss(pos_z, neg_z, summary)
        #
        # # Contrastive loss
        # augmented_x = self.corrupt(x, edge_index)[0]
        # contrastive_loss = self.contrastive_loss(x, augmented_x)

        return edge_prediction.view(-1)




class GraphNet3(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet3, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out, training=self.training)
        #
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)


class GraphNet4(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet4, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out, training=self.training)
        #
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)


class GraphNet5(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet5, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out, training=self.training)
        #
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)


class GraphNet6(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet6, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out, training=self.training)
        #
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)



class GraphNet7(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet7, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        #self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        # x = self.encoder(x, edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)


class GraphNet8(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet8, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        #self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        # x = self.encoder(x, edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)


class GraphNet9(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet9, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        #self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        # x = self.encoder(x, edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)



class GraphNet10(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet10, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        #self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        # x = self.encoder(x, edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)


class GraphNet11(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1,
                 use_variational=False):
        super(GraphNet11, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        #self.encoder = GraphNetEncoder(num_node_features, hidden_channels)
        self.use_variational = use_variational
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_hidden_channels),
            nn.Linear(mlp_hidden_channels, num_classes)
        )
        # self.pool = global_max_pool()  # 添加全局最大池化层

        # 新特征生成应用于 z 和 z1

        self.bga_layer = BGALayer(n_head=4, channels=hidden_channels, use_patch_attn=True, dropout=0.1)

    #     k = 3  # Grid size
    #     out_size = 7  # 输出特征维度
    #     self.additional_linear = nn.Linear(num_classes * 3, num_classes)
    #
    # def apply_grid_transform(self, x, k, inp_size, out_size):
    #     batch_size = x.size(0)
    #     device = x.device  # 获取 x 所在的设备
    #     # Grid transform
    #     repeated = x.unsqueeze(1).repeat(1, k, 1)
    #     shifts = torch.linspace(-1, 1, k).reshape(1, k, 1).to(device)  # 移动到同一设备
    #     shifted = repeated + shifts
    #     intermediate = torch.cat([shifted[:, :1, :], torch.relu(shifted[:, 1:, :])], dim=1).flatten(1)
    #     outputs = self.additional_linear(intermediate)
    #     return outputs

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index):
        # x = self.encoder(x, edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out * (1 - self.training), training=self.training)  # 动态调整
        if self.use_variational:
            # 如果是 VGAE，则生成 μ 和 logσ²
            mu, logvar = x.chunk(2, dim=1)
            x = self.reparameterize(mu, logvar)

        # 调用BGALayer
        # patch = edge_index  # 假设边索引用于Patch选择
        # x = self.bga_layer(x, need_attn=False)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        # k = 3# Grid size
        # # 应用网格变换
        # edge_prediction = self.apply_grid_transform(edge_prediction, k, edge_prediction.size(1), edge_prediction.size(1))

        return edge_prediction.view(-1)



