import argparse
import torch
from torch.autograd.gradcheck import _test_undefined_backward_mode
import torch.nn as nn
import torch.optim as optim
from explain_gnn import *
from load_data import load_data      # your function from before/same as Get_Dataset
from utils import train, test #, load_model   # your train/test/save functions
from build_logicGNN import *
from collections import defaultdict
from grounding import *
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#===========================
from pathlib import Path
import sys
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader
#===========================

original_atom_dict = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    11: "Na", 15: "P", 16: "S", 17: "Cl", 20: "Ca", 35: "Br", 53: "I"
}

atom_types = sorted(original_atom_dict.keys())
atom_to_idx = {atom_num: idx for idx, atom_num in enumerate(atom_types)}
num_atom_types = len(atom_types)
BBBP_atom_type_dict = {idx: original_atom_dict[atom_types[idx]] for idx in range(num_atom_types)}

TREEX_REPO = Path("/home/fhlic/Desktop/Repositories/treex_temp/treex_temp").resolve()
if str(TREEX_REPO) not in sys.path:
    sys.path.insert(0, str(TREEX_REPO))
from Utils.utils import load_model, check_task, detect_exp_setting, detect_motif_nodes
from Utils.datasets import get_dataset
from Utils.metrics import acc, efidelity, fid

parser = argparse.ArgumentParser(description='Train target model')
parser.add_argument('--data_name', type=str, default='Mutagenicity', help='Name of the dataset')
parser.add_argument('--input_channels', type=int, default=14, help='Number of input channels')
parser.add_argument('--hidden_channels', type=int, default=32, help='Number of hidden channels')
parser.add_argument('--output_channels', type=int, default=2, help='Number of output channels')
parser.add_argument('--target_model', type=str, default='checkpoints/models/Mutagenicity_model.pth', help='Path to the pretrained GNN model')
parser.add_argument('--label', type=int, default=1, help='Label of the data')
parser.add_argument('--gnn', type=str, default="gin", help='Type of GNN')
parser.add_argument('--clusters', type=int, default=3)
parser.add_argument('--into_st', type=int, default=6)
parser.add_argument('--local_cluster', type=str, default='kmeans', choices=['kmeans', 'em'])
parser.add_argument('--lmda', type=float, default=1)

parser.add_argument("--dataset", type=str, default='Mutagenicity', required=True)
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--load", action="store_true", help="Load pretrained model instead of training")
parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth for decision tree")
parser.add_argument("--arch", type=str, default="GCN", help="GNN architecture to use")
parser.add_argument("--plot", type=int, default=0, help="Whether to plot predicates")

args = parser.parse_args()

dataname = args.data_name
print(dataname)
task_type = check_task(dataname)
dataset = get_dataset(dataname)
try:dataset.print_summary()
except AttributeError: pass

try:n_fea, n_cls = dataset.num_features, dataset.num_classes 
except AttributeError: n_fea, n_cls = dataset.num_features, 2

print("============== Features/Classes ===============")
print("Number of features", n_fea)
print("Number of Classes", n_cls)

explain_ids = detect_exp_setting(dataname, dataset)
motif_nodes_number = detect_motif_nodes(dataname)

print("Explain IDs", explain_ids)
print("motif node number", motif_nodes_number)

gnn_model = load_model(dataname, args.gnn, n_fea, n_cls)
gnn_model.eval()

model = gnn_model
model.eval()
model_path = f'gin_Mutagenicity.model'

for name, param in model.named_parameters():
    print(name, param.shape)


train_dataset, test_dataset, train_loader, test_loader, device = load_data(args.dataset, args.seed)
y_labels_flat=[]
for data in train_dataset:
    y_labels_flat.append(data.y.item())
for data in test_dataset:
    y_labels_flat.append(data.y.item())

class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_labels_flat), y=y_labels_flat)
class_weights_map = dict(zip(np.unique(y_labels_flat), class_weights))

print(f"Correctly calculated class weights: {class_weights_map}")

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

test_acc = test(model, test_loader, device)
print(f"Loaded model | Test Accuracy: {test_acc:.4f}")

def accuracy_from_accs(accs):
    return sum(accs) / len(accs)

def test_with_acc(model, loader, device, dataname, acc_fn):
    model.eval()

    total_accs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            out, acts = model(data.x, data.edge_index, data.batch)

            pred = out.argmax(dim=1)
            y = data.y.view(-1).to(device)
            batch_acc = (pred == y).float().mean().item()

            Hnodes = acts.get("Hnodes", None)
            print("Hnodes:", Hnodes)
            print("Hnodes type:", type(Hnodes))

            if Hnodes is None:
                continue

            node = data.num_nodes
            all_nodes = list(range(node))

            accs = acc(dataname, node, all_nodes, Hnodes)

            graph_acc = accuracy_from_accs(accs)

            total_accs.append(graph_acc)
    return sum(total_accs) / len(total_accs) if len(total_accs) > 0 else 0.0


test_acc = test_with_acc(model=model,loader=test_loader,device=device,dataname=dataname,acc_fn=acc)

print("Explanation Accuracy (avg acc from acc()):", test_acc)

start_time = time.time()
gnn_train_pred_tensor, train_y_tensor, train_x_dict, train_edge_dict, train_activations_dict, train_gnn_graph_embed = get_all_activations_graph(train_loader,model,device)
gnn_test_pred_tensor, test_y_tensor, test_x_dict, test_edge_dict, test_activations_dict, test_gnn_graph_embed = get_all_activations_graph(test_loader,model,device)
save_dir_root=f"./plot/{args.dataset}/{args.seed}/{args.arch}"
