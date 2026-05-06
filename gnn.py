import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, SAGEConv, global_mean_pool
)
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, use_conv3=True):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.use_conv3 = use_conv3  # 🔹 flag

        if self.use_conv3:
            self.conv3 = GCNConv(hidden_channels, out_channels)
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        activations = {}

        x = self.conv1(x, edge_index); activations['conv1'] = x.detach().clone()
        x = F.relu(x); activations['relu1'] = x.detach().clone()

        x = self.conv2(x, edge_index); activations['conv2'] = x.detach().clone()
        # x = F.relu(x); activations['relu2'] = x.detach().clone()

        if self.use_conv3:
            x = self.conv3(x, edge_index); activations['conv3'] = x.detach().clone()

        x = global_mean_pool(x, batch); activations['global_pool'] = x.detach().clone()
        x = self.fc(x); activations['fc'] = x.detach().clone()
        return x, activations


# ==============================
# 🔹 GIN
# ==============================
class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(inplace=False),
                Linear(hidden, hidden),
                ReLU(inplace=False),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_hid_repr(self, data, layer=0):
        x, edge_index = data.x.float(), data.edge_index
        x = self.conv1(x, edge_index)
        if layer <= 0:
            return x
        for depth, conv in enumerate(self.convs, start=1):
            x = conv(x, edge_index)
            if layer <= depth:
                return x
        return x

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def get_gemb(self,data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        return x[0]

    def get_graph_emb(self, x, edge_index):
        x = x.float()
        batch = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        return torch.cat(
            [global_add_pool(x, batch)[0], global_mean_pool(x, batch)[0], global_max_pool(x, batch)[0]],
            dim=-1,
        )

    def fwd_weight(self, x, edge_index, edge_weight=None):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


    def fwd(self, x, edge_index, de=None, epsilon=None, edge_weight=None): 
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        if de is not None:
            edge_weight[de]=epsilon
            edl, edr = edge_index[0,de], edge_index[1,de]
            rev_de = int((torch.logical_and(edge_index[0]==edr, edge_index[1]==edl)==True).nonzero()[0])
            edge_weight[rev_de]=epsilon
        x = self.conv1(x.float(), edge_index, edge_weight=edge_weight)
        for o, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return x
        return F.log_softmax(x, dim=-1)

    def fwd_cam(self, data, edge_weight):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        # x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return F.softmax(x, dim=-1)
        return x

    def fwd_base(self, x, edge_index):
        x, edge_index = x.float(), edge_index
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 

        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    def fwd_base_other(self, x, edge_index, ie, value):
        batch = torch.zeros(x.shape[0]).to(x.device).type(torch.int64) 
        edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        edge_weight[ie]=value
        
        x = self.conv1(x.float(), edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GIN_NC(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden):
        super().__init__()
        self.num_layers = num_layers
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(inplace=False),
                Linear(hidden, hidden),
                ReLU(inplace=False),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        Linear(hidden, hidden),
                        ReLU(inplace=False),
                        BN(hidden),
                    ), train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
        return F.softmax(x, dim=-1)

    def fwd_eval(self, x, edge_index):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.softmax(x, dim=-1)

    def fwd_cam(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    def fwd(self, x, edge_index, de=None, epsilon=None):
        edge_weight = torch.ones(edge_index.shape[1]).float().to(edge_index.device)
        if de is not None:
            edge_weight[de]=epsilon
            edl, edr = edge_index[0,de], edge_index[1,de]
            rev_de = int((torch.logical_and(edge_index[0]==edr, edge_index[1]==edl)==True).nonzero()[0])
            edge_weight[rev_de]=epsilon
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__


# ==============================
# 🔹 GAT
# ==============================
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, heads=4, use_conv3=True):
        super(GAT, self).__init__()
        self.use_conv3 = use_conv3

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True)

        if self.use_conv3:
            self.conv3 = GATConv(hidden_channels, out_channels, heads=1, concat=True)
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        acts = {}
        x = self.conv1(x, edge_index); acts['conv1'] = x.detach().clone()
        x = F.elu(x); acts['relu1'] = x.detach().clone()

        x = self.conv2(x, edge_index); acts['conv2'] = x.detach().clone()
        #x = F.elu(x); acts['relu2'] = x.detach().clone()

        if self.use_conv3:
            x = self.conv3(x, edge_index); acts['conv3'] = x.detach().clone()

        x = global_mean_pool(x, batch); acts['global_pool'] = x.detach().clone()
        x = self.fc(x); acts['fc'] = x.detach().clone()
        return x, acts


# ==============================
# 🔹 GraphSAGE
# ==============================
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes=2, use_conv3=True):
        super(GraphSAGE, self).__init__()
        self.use_conv3 = use_conv3

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        if self.use_conv3:
            self.conv3 = SAGEConv(hidden_channels, out_channels)
            self.fc = nn.Linear(out_channels, num_classes)
        else:
            self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        acts = {}
        x = self.conv1(x, edge_index); acts['conv1'] = x.detach().clone()
        x = F.relu(x); acts['relu1'] = x.detach().clone()

        x = self.conv2(x, edge_index); acts['conv2'] = x.detach().clone()
        #x = F.relu(x); acts['relu2'] = x.detach().clone()

        if self.use_conv3:
            x = self.conv3(x, edge_index); acts['conv3'] = x.detach().clone()

        x = global_mean_pool(x, batch); acts['global_pool'] = x.detach().clone()
        x = self.fc(x); acts['fc'] = x.detach().clone()
        return x, acts
