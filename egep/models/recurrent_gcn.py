import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import (
    LRGCN, GCLSTM, DCRNN, AGCRN,
    DyGrEncoder, EvolveGCNO,
    GConvLSTM, GConvGRU
)
from .registry import register_model

# ---------------- LSTM-based (Return h, c) ----------------

@register_model("lrgcn")
class LRGCNModel(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=64):
        super().__init__()
        self.rnn = LRGCN(node_features, hidden_dim, 1, 1)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h, c = self.rnn(x, edge_index, edge_weight, h, c)
        return self.fc(F.relu(h)), h, c

@register_model("gclstm")
class GCLSTMModel(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=64):
        super().__init__()
        self.rnn = GCLSTM(node_features, hidden_dim, 1)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h, c = self.rnn(x, edge_index, edge_weight, h, c)
        return self.fc(F.relu(h)), h, c

@register_model("dygrencoder")
class DyGrEncoderModel(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=128):
        super().__init__()
        self.rnn = DyGrEncoder(
            conv_out_channels=node_features,
            conv_num_layers=1,
            conv_aggr="mean",
            lstm_out_channels=hidden_dim,
            lstm_num_layers=1,
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        y, h, c = self.rnn(x, edge_index, edge_weight, h, c)
        return self.fc(F.relu(y)), h, c

@register_model("gconvlstm")
class GConvLSTMModel(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=32):
        super().__init__()
        self.rnn = GConvLSTM(node_features, hidden_dim, 1)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h, c = self.rnn(x, edge_index, edge_weight, h, c)
        return self.fc(F.relu(h)), h, c

# ---------------- GRU / GCN-based (Return output only) ----------------

@register_model("dcrnn")
class DCRNNModel(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=64):
        super().__init__()
        self.rnn = DCRNN(node_features, hidden_dim, 1)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        return self.fc(F.relu(self.rnn(x, edge_index, edge_weight)))

@register_model("evolvegcno")
class EvolveGCNOModel(torch.nn.Module):
    # Added hidden_dim argument for compatibility, even if unused
    def __init__(self, node_features, hidden_dim=None):
        super().__init__()
        self.rnn = EvolveGCNO(node_features)
        self.fc = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        return self.fc(F.relu(self.rnn(x, edge_index, edge_weight)))

@register_model("gconvgru")
class GConvGRUModel(torch.nn.Module):
    def __init__(self, node_features, hidden_dim=32):
        super().__init__()
        self.rnn = GConvGRU(node_features, hidden_dim, 1)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight):
        return self.fc(F.relu(self.rnn(x, edge_index, edge_weight)))

# ---------------- AGCRN (Special Input) ----------------

@register_model("agcrn")
class AGCRNModel(torch.nn.Module):
    def __init__(self, node_features, num_nodes, hidden_dim=2):
        super().__init__()
        self.rnn = AGCRN(
            number_of_nodes=num_nodes,
            in_channels=node_features,
            out_channels=hidden_dim,
            K=2,
            embedding_dimensions=1,
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, e, h):
        h = self.rnn(x, e, h)
        return self.fc(F.relu(h)), h