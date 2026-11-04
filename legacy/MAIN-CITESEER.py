import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dropout_edge
import networkx as nx
import numpy as np
from community import community_louvain

# Metrics & clustering
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             normalized_mutual_info_score, adjusted_rand_score)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

################################################################################
# 0) LOAD CORA DATASET
################################################################################

dataset_name = 'Citeseer'
dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
data = dataset[0]

################################################################################
# 1) LOUVAIN COMMUNITY DETECTION
################################################################################
def apply_louvain(graph):
    G = nx.Graph()
    G.add_edges_from(graph.edge_index.t().tolist())
    partition = community_louvain.best_partition(G)
    return np.array([partition.get(i, -1) for i in range(graph.num_nodes)])

louvain_labels = apply_louvain(data)
valid_nodes = (louvain_labels != -1)
valid_nodes_tensor = torch.tensor(valid_nodes, dtype=torch.bool)

# Filter
louvain_labels = louvain_labels[valid_nodes]
louvain_labels_tensor = torch.tensor(louvain_labels, dtype=torch.long).to(data.x.device)

num_communities = len(np.unique(louvain_labels))

data.x = data.x[valid_nodes_tensor]
data.y = data.y[valid_nodes_tensor]

old_to_new_index = torch.zeros(valid_nodes_tensor.size(0), dtype=torch.long, device=data.x.device)
old_to_new_index[valid_nodes_tensor] = torch.arange(data.x.size(0), device=data.x.device)
data.edge_index = old_to_new_index[data.edge_index]

################################################################################
# 2) n-HDP: NESTED HIERARCHICAL DIRICHLET PROCESS (SIMULATED W/ AGGLO CLUSTER)
################################################################################
def apply_nhdp(node_features, num_levels=3):
    # As in your code, exclude last column for clustering
    node_features_sub = node_features[:, :-1]
    nested_labels = np.zeros((node_features_sub.shape[0], num_levels))

    for level in range(num_levels):
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
        nested_labels[:, level] = clustering.fit_predict(node_features_sub)
    return nested_labels

node_features = data.x.cpu().numpy()
nhdp_labels = apply_nhdp(node_features)
nhdp_labels = torch.tensor(nhdp_labels, dtype=torch.float) * 0.1

data.x = torch.cat([data.x, nhdp_labels], dim=1)

################################################################################
# 3) DEFINE GNN MODEL (ENHANCED VARIATIONAL HIERARCHICAL GAT)
################################################################################
class EnhancedVariationalHierarchicalGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_heads=4, latent_dim=32, dropout=0.5, num_communities=30):
        super().__init__()
        self.latent_dim = latent_dim

        # 3-stage GAT
        self.local_gat = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.local_bn = BatchNorm(hidden_channels * num_heads)

        self.community_gat = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.community_bn = BatchNorm(hidden_channels * num_heads)

        self.global_gat = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.global_bn = BatchNorm(hidden_channels * num_heads)

        # Residual
        self.residual_fc = torch.nn.Linear(in_channels, hidden_channels * num_heads)

        # latent var
        self.fc_mu = torch.nn.Linear(hidden_channels * num_heads, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_channels * num_heads, latent_dim)

        # decoder
        self.decoder = torch.nn.Linear(latent_dim, out_channels)

        # aux
        self.aux_decoder = torch.nn.Linear(hidden_channels * num_heads, num_communities)

        self.dropout = dropout

    def encode(self, x, edge_index):
        # residual
        residual = self.residual_fc(x)

        x = F.relu(self.local_bn(self.local_gat(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.community_bn(self.community_gat(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.global_bn(self.global_gat(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x += residual

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, edge_index):
        mu, logvar, enc = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        aux_out = self.aux_decoder(enc)
        return out, mu, logvar, aux_out, enc  # return enc for final embeddings

################################################################################
# 4) LOSS FUNCTION
################################################################################
def loss_function(recon_x, x_gt, mu, logvar, class_weights,
                  aux_out, aux_labels, beta=5e-3, aux_weight=0.01):
    # main classification
    main_loss = F.cross_entropy(recon_x, x_gt, weight=class_weights)
    # KL
    kld = -0.5* torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
    # auxiliary
    aux_loss = F.cross_entropy(aux_out, aux_labels)

    return main_loss + kld + aux_weight*aux_loss

################################################################################
# 5) CLASS WEIGHTS & MODEL
################################################################################
class_weights_np = compute_class_weight('balanced',
                                        classes=np.unique(data.y.numpy()),
                                        y=data.y.numpy())
class_weights = torch.tensor(class_weights_np, dtype=torch.float)

model = EnhancedVariationalHierarchicalGAT(
    in_channels=data.x.shape[1],
    hidden_channels=64,
    out_channels=dataset.num_classes,
    num_heads=4,
    latent_dim=32,
    dropout=0.5,
    num_communities=num_communities
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

################################################################################
# 6) TRAIN LOOP
################################################################################
best_loss = float('inf')
best_model = None
patience_counter = 0
patience = 500
train_losses, val_losses, lrs = [], [], []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
data = data.to(device)
aux_labels_t = louvain_labels_tensor.to(device)

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    edge_index_drop = dropout_edge(data.edge_index, p=0.2)[0]
    out, mu, logvar, aux_out, enc = model(data.x, edge_index_drop)

    loss = loss_function(
        recon_x=out,
        x_gt=data.y,
        mu=mu,
        logvar=logvar,
        class_weights=class_weights,
        aux_out=aux_out,
        aux_labels=aux_labels_t,
        beta=2e-5,
        aux_weight=0.01
    )
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())
    lrs.append(optimizer.param_groups[0]['lr'])

    # val
    model.eval()
    with torch.no_grad():
        val_out, val_mu, val_logvar, val_aux, _ = model(data.x, data.edge_index)
        val_loss = loss_function(val_out, data.y, val_mu, val_logvar,
                                 class_weights, val_aux, aux_labels_t,
                                 beta=2e-5, aux_weight=0.01).item()
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.4f}, Val Loss={val_loss:.4f}")

# load best
model.load_state_dict(best_model)

################################################################################
# 7) EVALUATION
################################################################################
model.eval()
with torch.no_grad():
    out, mu, logvar, aux_out, enc = model(data.x, data.edge_index)
    y_true = data.y.cpu().numpy()
    y_pred = out.argmax(dim=1).cpu().numpy()
    y_proba = F.softmax(out, dim=1).cpu().numpy()

acc = accuracy_score(y_true, y_pred)
f1v = f1_score(y_true, y_pred, average='weighted')
prec = precision_score(y_true, y_pred, average='weighted')
rec  = recall_score(y_true, y_pred, average='weighted')
aucv = roc_auc_score(y_true, y_proba, multi_class='ovr')

print("\n===== Final Evaluation =====")
print(f"Accuracy:  {acc:.4f}")
print(f"F1-score:  {f1v:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"AUC-ROC:   {aucv:.4f}")

# Plot train/val loss
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot LR
plt.figure()
plt.plot(lrs, label='Learning Rate')
plt.title("Learning Rate over Epochs")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.legend()
plt.show()

###############################################################################
# 8) Over-smoothing / Distinctness Metrics
###############################################################################
def compute_over_smoothing_metrics(emb, labels):
    emb_np = emb.cpu().numpy()
    dists = pdist(emb_np, metric='euclidean')
    avg_pw = dists.mean()
    dm = squareform(dists)
    same_mask = (labels[:,None] == labels[None,:])
    np.fill_diagonal(same_mask, False)
    intra = dm[same_mask].mean() if same_mask.sum()>0 else 0
    inter = dm[~same_mask].mean() if (~same_mask).sum()>0 else 0
    ratio = inter/(intra+1e-9)
    var_ = np.var(emb_np, axis=0).mean()
    return avg_pw, intra, inter, ratio, var_

avg_pairwise, avg_intra, avg_inter, ratio_inter_intra, emb_variance = compute_over_smoothing_metrics(enc, y_true)
print("\n===== Over-Smoothing / Embedding Distinctness =====")
print(f"Average Pairwise Dist: {avg_pairwise:.4f}")
print(f"Average Intra-Class Dist: {avg_intra:.4f}")
print(f"Average Inter-Class Dist: {avg_inter:.4f}")
print(f"Inter/Intra Ratio: {ratio_inter_intra:.4f}")
print(f"Average Embedding Variance: {emb_variance:.4f}")

###############################################################################
# 9) Confusion Matrix
###############################################################################
cm = confusion_matrix(y_true, y_pred)
# If you want actual labels for Cora
cora_labels = ["Case_Based","Genetic_Algorithms","Neural_Networks",
               "Probabilistic_Methods","Reinforcement_Learning","Rule_Learning","Theory"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cora_labels)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.show()

###############################################################################
# 10) T-SNE & Histograms
###############################################################################
enc_np = enc.cpu().numpy()

# T-SNE
tsne_coords = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(enc_np)
plt.figure(figsize=(8,6))
scatter = plt.scatter(tsne_coords[:,0], tsne_coords[:,1], c=y_true, cmap='viridis', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE of Node Embeddings")
plt.show()

# Embedding L2 Norm histogram
norms = np.linalg.norm(enc_np, axis=1)
plt.figure()
plt.hist(norms, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Embedding L2 Norms")
plt.xlabel("Norm")
plt.ylabel("Frequency")
plt.show()

# Pairwise distances
plt.figure()
pw_dists = pdist(enc_np, metric='euclidean')
plt.hist(pw_dists, bins=30, color='lightgreen', edgecolor='black')
plt.title("Histogram of Pairwise Distances")
plt.xlabel("Distance")
plt.ylabel("Count")
plt.show()
