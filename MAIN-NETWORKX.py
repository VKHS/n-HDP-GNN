import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge
import networkx as nx
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from community import community_louvain  # Import community_louvain for Louvain method
from sklearn.cluster import AgglomerativeClustering  # Ensure this is imported

# Louvain Community Detection
def apply_louvain(G):
    partition = community_louvain.best_partition(G)
    community_labels = np.array([partition[i] for i in range(len(G.nodes()))])
    return community_labels

# Nested Hierarchical Dirichlet Process (n-HDP) approximation using Agglomerative Clustering
def apply_nhdp(node_features, num_levels):
    labels = node_features[:, -1]  # Assume the last column are initial community labels
    node_features = node_features[:, :-1]  # Exclude the last column for clustering

    nested_labels = np.zeros((node_features.shape[0], num_levels))

    # Apply hierarchical clustering at multiple levels
    for level in range(num_levels):
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)
        nested_labels[:, level] = clustering.fit_predict(node_features)

    return nested_labels

# Define a GNN model with hierarchical attention mechanisms and variational inference
class EnhancedVariationalHierarchicalGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, latent_dim, dropout=0.5, num_communities=30):
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
        # First layer: Local Attention
        residual = self.residual_fc(x)
        x = F.relu(self.local_bn(self.local_gat(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second layer: Community-Level Attention
        x = F.relu(self.community_bn(self.community_gat(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Third layer: Global Attention
        x = F.relu(self.global_bn(self.global_gat(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Add residual connection
        x += residual

        # Latent variables
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
        return out, mu, logvar, aux_out

def loss_function(recon_x, x, mu, logvar, class_weights, recon_aux, aux_labels, beta=0.0000001, aux_weight=1.0):
    # Main classification loss
    recon_loss = F.cross_entropy(recon_x, x, weight=class_weights)
    
    # KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Auxiliary loss
    aux_loss = F.cross_entropy(recon_aux, aux_labels)
    
    # Combined loss
    return recon_loss + beta * kld + aux_weight * aux_loss

# Create the social network graph (unchanged)
def create_graph(num_communities, num_nodes, edges_per_community):
    G = nx.Graph()

    # Assign nodes to communities
    nodes_per_community = num_nodes // num_communities
    nodes = list(range(num_nodes))

    # Add nodes (individuals)
    G.add_nodes_from(nodes)

    # Add edges (friendships) with hierarchical structure
    for i in range(num_communities):
        start = i * nodes_per_community
        end = (i + 1) * nodes_per_community
        community_nodes = list(range(start, end))

        for _ in range(edges_per_community):
            node1, node2 = np.random.choice(community_nodes, size=2, replace=False)
            G.add_edge(node1, node2)

    return G

# Example usage (unchanged)
num_communities = 30
num_nodes = 5000
edges_per_community = 50
num_levels = 3  # Number of hierarchical levels in n-HDP

G = create_graph(num_communities, num_nodes, edges_per_community)

# Apply Louvain to get initial community labels
louvain_labels = apply_louvain(G)
num_communities = len(np.unique(louvain_labels))  # Calculate the number of unique communities

# Create node features (using Louvain labels and random features)
node_features = np.random.rand(num_nodes, 16)
node_features = np.concatenate([node_features, louvain_labels.reshape(-1, 1)], axis=1)

# Apply n-HDP to get nested clusters
nested_labels = apply_nhdp(node_features, num_levels=num_levels)

# Integrate n-HDP labels into node features
node_features = np.concatenate([node_features, nested_labels], axis=1)
node_features = torch.tensor(node_features, dtype=torch.float)

# Normalize node features (unchanged)
scaler = StandardScaler()
node_features = torch.tensor(scaler.fit_transform(node_features.numpy()), dtype=torch.float)

# Generate random labels for classification (for demonstration purposes)
labels = torch.randint(0, 3, (num_nodes,))

# Compute class weights (unchanged)
class_weights = compute_class_weight('balanced', classes=np.unique(labels.numpy()), y=labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Create a PyTorch Geometric data object (unchanged)
edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Instantiate and train the model
latent_dim = 32
model = EnhancedVariationalHierarchicalGAT(
    in_channels=node_features.shape[1], 
    hidden_channels=64, 
    out_channels=3, 
    num_heads=8, 
    latent_dim=latent_dim, 
    dropout=0.5,
    num_communities=num_communities  # Use the calculated number of communities
)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Early stopping parameters
best_loss = float('inf')
patience = 20
patience_counter = 0

# Training loop
for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    
    # Apply edge dropout
    edge_index = dropout_edge(data.edge_index, p=0.2)[0]
    
    out, mu, logvar, aux_out = model(data.x, edge_index)
    
    # Convert louvain_labels to torch.Tensor
    louvain_labels_tensor = torch.tensor(louvain_labels, dtype=torch.long).to(out.device)
    
    loss = loss_function(out, data.y, mu, logvar, class_weights, aux_out, louvain_labels_tensor)
    loss.backward()
    optimizer.step()

    # Validate and check early stopping
    model.eval()
    with torch.no_grad():
        val_out, _, _, val_aux_out = model(data.x, data.edge_index)
        val_loss = loss_function(val_out, data.y, mu, logvar, class_weights, val_aux_out, louvain_labels_tensor).item()

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

# Load best model before evaluation
model.load_state_dict(best_model)

# Evaluation (unchanged)
model.eval()
with torch.no_grad():
    out, _, _, _ = model(data.x, data.edge_index)
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
