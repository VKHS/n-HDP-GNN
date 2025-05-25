import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.datasets import Amazon
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import dropout_edge
import networkx as nx
import numpy as np
import community as community_louvain  # Import Louvain module for community detection
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from networkx.algorithms.centrality import betweenness_centrality
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Change the dataset to Amazon Photo
dataset_name = 'photo'

# Load the Amazon Photo dataset
dataset = Amazon(root=f'/tmp/{dataset_name}', name=dataset_name, transform=NormalizeFeatures())
data = dataset[0]

# Louvain Community Detection for Auxiliary Task
def apply_louvain(graph):
    G_nx = nx.Graph()
    G_nx.add_edges_from(graph.edge_index.t().tolist())  # Convert edge_index to list of tuples

    # Use the Louvain algorithm to detect communities
    partition = community_louvain.best_partition(G_nx)

    # Initialize community labels (-1 for isolated nodes)
    community_labels = np.full(graph.num_nodes, -1)

    # Assign community membership to nodes that are part of a community
    for node_id in range(graph.num_nodes):
        community_labels[node_id] = partition.get(node_id, -1)  # Default -1 for isolated nodes

    return community_labels

# Apply Louvain to graph as auxiliary labels
louvain_labels = apply_louvain(data)

# Ensure valid nodes (those that have a valid Louvain label and participate in the graph)
valid_nodes = (louvain_labels != -1)

# Convert valid_nodes to a PyTorch tensor
valid_nodes_tensor = torch.tensor(valid_nodes, dtype=torch.bool)

# Ensure that the length of valid_nodes_tensor matches the number of nodes in data.x
if valid_nodes_tensor.shape[0] != data.x.shape[0]:
    raise ValueError(f"Mismatch between valid_nodes ({valid_nodes_tensor.shape[0]}) and node features ({data.x.shape[0]}).")

# Filter the Louvain labels and node features to keep only valid nodes
louvain_labels = louvain_labels[valid_nodes]
louvain_labels_tensor = torch.tensor(louvain_labels, dtype=torch.long).to(data.x.device)

# Calculate the number of unique Louvain communities
num_communities = len(np.unique(louvain_labels))

# Filter the node features and target labels based on valid nodes
data.x = data.x[valid_nodes_tensor]
data.y = data.y[valid_nodes_tensor]  # Ensure labels are filtered to match the valid nodes

# Create a mapping from the old node indices to the new filtered node indices
old_to_new_index = torch.zeros(valid_nodes_tensor.size(0), dtype=torch.long, device=data.x.device)
old_to_new_index[valid_nodes_tensor] = torch.arange(data.x.size(0), device=data.x.device)

# Apply the mapping to the edge_index
data.edge_index = old_to_new_index[data.edge_index]

# Add graph topology information as additional features
def add_topology_features(graph, node_features):
    G = nx.Graph()
    G.add_edges_from(graph.edge_index.t().tolist())
    
    # Calculate betweenness centrality
    centrality = betweenness_centrality(G)
    
    # Assign a default centrality value (e.g., 0) for nodes that are missing from the centrality dictionary
    centrality_values = np.array([centrality.get(i, 0.0) for i in range(graph.num_nodes)])
    
    # Standardize the features
    scaler = StandardScaler()
    centrality_values = scaler.fit_transform(centrality_values.reshape(-1, 1))

    # Append the centrality values to the node features
    node_features = np.hstack((node_features, centrality_values))
    return node_features

# Add graph topology features to the node features
data.x = torch.tensor(add_topology_features(data, data.x.cpu().numpy()), dtype=torch.float)

# Define the GNN model with increased capacity
class EnhancedVariationalHierarchicalGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=8, latent_dim=64, dropout=0.5, num_communities=30):
        super(EnhancedVariationalHierarchicalGAT, self).__init__()

        self.latent_dim = latent_dim

        # Local Attention (First Layer)
        self.local_gat = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.local_bn = BatchNorm(hidden_channels * num_heads)

        # Community-Level Attention (Second Layer)
        self.community_gat = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.community_bn = BatchNorm(hidden_channels * num_heads)

        # Global Attention (Third Layer)
        self.global_gat = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.global_bn = BatchNorm(hidden_channels * num_heads)

        # Residual connections
        self.residual_fc = torch.nn.Linear(in_channels, hidden_channels * num_heads)

        # Latent variable layers
        self.fc_mu = torch.nn.Linear(hidden_channels * num_heads, latent_dim)
        self.fc_logvar = torch.nn.Linear(hidden_channels * num_heads, latent_dim)

        # Decoder
        self.decoder = torch.nn.Linear(latent_dim, out_channels)

        # Auxiliary Task: Community Detection
        self.aux_decoder = torch.nn.Linear(hidden_channels * num_heads, num_communities)

        # Dropout
        self.dropout = dropout

    def encode(self, x, edge_index):
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def auxiliary_task(self, x):
        return self.aux_decoder(x)

    def forward(self, x, edge_index):
        mu, logvar, encodings = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        aux_out = self.auxiliary_task(encodings)
        return out, mu, logvar, aux_out, encodings  # Return encodings for visualization

# Custom loss function with NMI/ARI directly optimized
def nmi_ari_loss(recon_x, x, mu, logvar, class_weights, recon_aux, aux_labels, beta=1e-7, aux_weight=0.5):
    # Main classification loss (Reconstruction loss)
    recon_loss = F.cross_entropy(recon_x, x, weight=class_weights)
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * beta
    
    # Auxiliary loss (Louvain community detection task)
    aux_loss = F.cross_entropy(recon_aux, aux_labels)
    
    # NMI and ARI for community prediction
    aux_pred = recon_aux.argmax(dim=1).cpu().numpy()
    nmi = normalized_mutual_info_score(aux_labels.cpu().numpy(), aux_pred)
    ari = adjusted_rand_score(aux_labels.cpu().numpy(), aux_pred)
    
    nmi_loss = 1 - nmi
    ari_loss = 1 - ari
    
    # Combined loss with scaled KL, auxiliary, and NMI/ARI loss
    return recon_loss + kld + aux_weight * aux_loss + nmi_loss + ari_loss

# Prepare class weights
class_weights = compute_class_weight('balanced', classes=np.unique(data.y.numpy()), y=data.y.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Instantiate the model with the new changes
model = EnhancedVariationalHierarchicalGAT(
    in_channels=data.x.shape[1],  # Includes original features + new topology features
    hidden_channels=128,  # Increased hidden size
    out_channels=dataset.num_classes, 
    num_heads=8,  # Increased number of heads
    latent_dim=64, 
    dropout=0.7, 
    num_communities=num_communities  # Ensure correct number of communities
)

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training loop with custom NMI/ARI loss
best_loss = float('inf')
patience = 1500
patience_counter = 0

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    # Apply edge dropout
    edge_index = dropout_edge(data.edge_index, p=0.2)[0]
    
    out, mu, logvar, aux_out, encodings = model(data.x, edge_index)
    
    # Loss calculation with NMI/ARI
    loss = nmi_ari_loss(out, data.y, mu, logvar, class_weights, aux_out, louvain_labels_tensor)
    loss.backward()
    optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_out, _, _, val_aux_out, _ = model(data.x, data.edge_index)
        val_loss = nmi_ari_loss(val_out, data.y, mu, logvar, class_weights, val_aux_out, louvain_labels_tensor).item()

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
        print(f'Epoch {epoch}, Loss: {loss.item()}, Val Loss: {val_loss}')

# Load best model and evaluate
model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    out, _, _, aux_out, encodings = model(data.x, data.edge_index)
    
    # Main task evaluation
    y_true = data.y.cpu().numpy()
    y_pred = out.argmax(dim=1).cpu().numpy()
    y_proba = F.softmax(out, dim=1).cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_true, y_proba, multi_class='ovr')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'AUC-ROC: {auc_roc:.4f}')

    # Auxiliary task evaluation
    aux_pred = aux_out.argmax(dim=1).cpu().numpy()  # Predicted communities from the auxiliary task
    louvain_true = louvain_labels_tensor.cpu().numpy()  # True Louvain community labels

    

    # Visualize node embeddings using t-SNE
    embeddings_2d = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(encodings.cpu().numpy())

    # Plot the embeddings
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Classes')
    plt.title('Node Embeddings Visualized with t-SNE')
    plt.show()
