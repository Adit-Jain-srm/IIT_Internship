# Implementation Guide: Graph-Based Methods for Imitation Learning

## Quick Start: Graph Neural Network Examples

This document provides ready-to-implement examples using modern graph-based unsupervised learning algorithms.

---

## Part 1: Setup & Dependencies

### Installation

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install dgl
pip install pytorch-lightning
pip install tensorboard
pip install numpy pandas scikit-learn matplotlib seaborn

# Optional: For faster GNN training
pip install torch-sparse
pip install torch-scatter
```

### Verify Installation

```python
import torch
import torch_geometric
import dgl
import pytorch_lightning as pl

print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"DGL: {dgl.__version__}")
print(f"PyTorch Lightning: {pl.__version__}")
```

---

## Part 2: Hand Skeleton Definition

### Define Anatomically Correct Hand Graph

```python
import torch
import numpy as np

class HandSkeleton:
    """Defines MediaPipe hand skeleton with 21 landmarks"""
    
    # Landmark indices
    WRIST = 0
    THUMB_BASE, THUMB_MID, THUMB_TIP = 1, 2, 3
    INDEX_BASE, INDEX_MID, INDEX_TIP = 5, 6, 7
    MIDDLE_BASE, MIDDLE_MID, MIDDLE_TIP = 9, 10, 11
    RING_BASE, RING_MID, RING_TIP = 13, 14, 15
    PINKY_BASE, PINKY_MID, PINKY_TIP = 17, 18, 19
    
    # Kinematic chain (parent → child relationships)
    EDGES = [
        # Thumb chain
        (0, 1), (1, 2), (2, 3),
        # Index chain
        (0, 5), (5, 6), (6, 7),
        # Middle chain
        (0, 9), (9, 10), (10, 11),
        # Ring chain
        (0, 13), (13, 14), (14, 15),
        # Pinky chain
        (0, 17), (17, 18), (18, 19),
        # Add intra-finger connections for better feature learning
        (1, 5), (5, 9), (9, 13), (13, 17),  # Base-to-base
        (2, 6), (6, 10), (10, 14), (14, 18),  # Mid-to-mid
        (3, 7), (7, 11), (11, 15), (15, 19),  # Tip-to-tip
    ]
    
    @staticmethod
    def get_edge_index():
        """Convert edges to PyTorch Geometric edge_index format"""
        edges = np.array(HandSkeleton.EDGES)
        edge_index = torch.LongTensor(edges.T)
        return edge_index
    
    @staticmethod
    def get_edge_weight(landmarks):
        """
        Compute normalized edge weights based on segment lengths.
        
        Args:
            landmarks: (21, 3) array of hand positions
        
        Returns:
            edge_weight: (num_edges,) tensor of normalized weights
        """
        edge_weight = []
        for src, dst in HandSkeleton.EDGES:
            src_pos = landmarks[src]
            dst_pos = landmarks[dst]
            distance = np.linalg.norm(dst_pos - src_pos)
            edge_weight.append(distance)
        
        edge_weight = np.array(edge_weight)
        edge_weight = edge_weight / edge_weight.max()  # Normalize to [0, 1]
        return torch.FloatTensor(edge_weight)

# Test
edge_index = HandSkeleton.get_edge_index()
print(f"Hand skeleton has {edge_index.shape[1]} edges")
print(f"Edge index shape: {edge_index.shape}")
```

---

## Part 3: Simple Graph Neural Network Models

### Model 1: Basic GCN for Gesture Classification

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class HandGestureGCN(nn.Module):
    """Simple 3-layer GCN for 21-landmark hand gesture classification"""
    
    def __init__(self, num_features=3, num_classes=8):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        
        # Classification head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data):
        """
        Args:
            data: PyG Data object with:
                - x: node features (21, 3)
                - edge_index: connectivity (2, num_edges)
        
        Returns:
            logits: gesture class logits (num_classes,)
        """
        x, edge_index = data.x, data.edge_index
        
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling (average over landmarks)
        x = global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long))
        
        # Classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class HandGestureGAT(nn.Module):
    """Graph Attention Network for hand gestures"""
    
    def __init__(self, num_features=3, num_classes=8, num_heads=4):
        super().__init__()
        from torch_geometric.nn import GATConv
        
        self.att1 = GATConv(num_features, 64, heads=num_heads, dropout=0.3)
        self.att2 = GATConv(64 * num_heads, 128, heads=num_heads, dropout=0.3)
        self.att3 = GATConv(128 * num_heads, 256, heads=1, dropout=0.3)
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.att1(x, edge_index)
        x = F.relu(x)
        
        x = self.att2(x, edge_index)
        x = F.relu(x)
        
        x = self.att3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, torch.zeros(x.shape[0], dtype=torch.long))
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x


# Usage Example
def create_hand_data(landmarks, gesture_id, edge_index):
    """
    Convert numpy landmarks to PyG Data object.
    
    Args:
        landmarks: (21, 3) numpy array of hand positions
        gesture_id: integer gesture class (0-7)
        edge_index: hand skeleton edges
    
    Returns:
        data: PyG Data object
    """
    x = torch.FloatTensor(landmarks)
    y = torch.LongTensor([gesture_id])
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    return data

# Test instantiation
edge_index = HandSkeleton.get_edge_index()
model_gcn = HandGestureGCN(num_features=3, num_classes=8)
model_gat = HandGestureGAT(num_features=3, num_classes=8)

# Test forward pass
dummy_landmarks = np.random.randn(21, 3).astype(np.float32)
test_data = create_hand_data(dummy_landmarks, gesture_id=2, edge_index=edge_index)
output_gcn = model_gcn(test_data)
output_gat = model_gat(test_data)

print(f"GCN output shape: {output_gcn.shape}")  # Should be (1, 8)
print(f"GAT output shape: {output_gat.shape}")  # Should be (1, 8)
```

---

## Part 4: Temporal Modeling

### Model 2: Temporal Graph Convolution Network (T-GCN)

```python
class TemporalGCN(nn.Module):
    """Models hand gestures over time"""
    
    def __init__(self, num_features=3, num_classes=8, hidden_dim=128):
        super().__init__()
        from torch_geometric.nn import GCNConv
        
        # Spatial: static hand structure
        self.spatial_gcn = GCNConv(num_features, hidden_dim)
        
        # Temporal: RNN for sequence
        self.temporal_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, landmarks_seq, edge_index):
        """
        Args:
            landmarks_seq: (seq_len, 21, 3) sequence of hand poses
            edge_index: hand skeleton connectivity
        
        Returns:
            logits: (num_classes,) gesture classification
        """
        seq_len = landmarks_seq.shape[0]
        
        # Process each frame through spatial GCN
        frame_embeddings = []
        for t in range(seq_len):
            x = landmarks_seq[t]  # (21, 3)
            x = self.spatial_gcn(x.unsqueeze(0), edge_index)  # (21, hidden_dim)
            x = x.mean(dim=0)  # Global pooling: (hidden_dim,)
            frame_embeddings.append(x)
        
        # Stack into sequence
        frame_embeddings = torch.stack(frame_embeddings)  # (seq_len, hidden_dim)
        
        # Temporal modeling
        temporal_output, _ = self.temporal_rnn(frame_embeddings.unsqueeze(0))
        # temporal_output: (1, seq_len, hidden_dim)
        
        # Use final hidden state for classification
        final_state = temporal_output[0, -1, :]  # (hidden_dim,)
        logits = self.fc(final_state)
        
        return logits


class TransformerGesture(nn.Module):
    """Transformer-based temporal gesture modeling"""
    
    def __init__(self, num_features=3, num_classes=8, hidden_dim=128, num_heads=4):
        super().__init__()
        from torch_geometric.nn import GCNConv
        
        # Spatial embedding (GCN)
        self.spatial_embed = GCNConv(num_features, hidden_dim)
        
        # Temporal embedding (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Classification
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, landmarks_seq, edge_index):
        """
        Args:
            landmarks_seq: (seq_len, 21, 3)
            edge_index: skeleton connectivity
        """
        seq_len = landmarks_seq.shape[0]
        
        # Spatial embedding
        frame_embeddings = []
        for t in range(seq_len):
            x = landmarks_seq[t]
            x = self.spatial_embed(x.unsqueeze(0), edge_index)
            x = x.mean(dim=0)
            frame_embeddings.append(x)
        
        frame_embeddings = torch.stack(frame_embeddings)  # (seq_len, hidden_dim)
        
        # Temporal modeling with Transformer
        temporal_features = self.transformer(frame_embeddings.unsqueeze(0))
        # (1, seq_len, hidden_dim)
        
        # Classify using last frame
        final_feature = temporal_features[0, -1, :]
        logits = self.fc(final_feature)
        
        return logits
```

---

## Part 5: Data Loading & Training

### Setup Data Pipeline

```python
import os
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class GestureDataset(Dataset):
    """Load gesture CSV files and convert to PyG Data objects"""
    
    def __init__(self, data_dir, gesture_labels=None):
        """
        Args:
            data_dir: directory containing gesture subdirectories
            gesture_labels: dict mapping folder name to class ID
        """
        self.data_dir = Path(data_dir)
        
        if gesture_labels is None:
            self.gesture_labels = {
                'Cleaning': 0,
                'Come': 1,
                'Emergency_calling': 2,
                'Give': 3,
                'Good': 4,
                'Pick': 5,
                'Stack': 6,
                'Wave': 7
            }
        else:
            self.gesture_labels = gesture_labels
        
        # Collect all CSV files
        self.files = []
        for gesture_folder, class_id in self.gesture_labels.items():
            gesture_path = self.data_dir / gesture_folder
            csv_files = list(gesture_path.glob('*.csv'))
            self.files.extend([(f, class_id) for f in csv_files])
        
        self.edge_index = HandSkeleton.get_edge_index()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        csv_file, class_id = self.files[idx]
        
        # Read landmarks
        df = pd.read_csv(csv_file)
        landmarks = df[['X', 'Y', 'Z']].values  # (N, 3)
        
        # Convert to hand gesture data
        data = create_hand_data(landmarks, class_id, self.edge_index)
        return data


# Usage
dataset = GestureDataset(
    data_dir='c:\\Users\\aditj\\New Projects\\IIT_Internship\\input_gesture_1'
)

print(f"Total samples: {len(dataset)}")
print(f"Sample 0: gesture class = {dataset[0].y}")

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## Part 6: Complete Training Loop

```python
import pytorch_lightning as pl
from torch.optim import Adam

class GestureGNN_LightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for training"""
    
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, data):
        return self.model(data)
    
    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch.y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch.y)
        
        # Accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == batch.y).float().mean()
        
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


# Training
from sklearn.model_selection import train_test_split

# Split dataset
train_idx, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    random_state=42
)

train_subset = torch.utils.data.Subset(dataset, train_idx)
val_subset = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

# Model & training
model = HandGestureGCN(num_features=3, num_classes=8)
lit_model = GestureGNN_LightningModule(model, lr=1e-3)

trainer = pl.Trainer(
    max_epochs=50,
    gpus=1 if torch.cuda.is_available() else 0,
    enable_progress_bar=True,
    log_every_n_steps=10
)

trainer.fit(lit_model, train_loader, val_loader)
```

---

## Part 7: Gesture Segmentation (Detecting Start/End)

```python
class GestureSegmenter(nn.Module):
    """Automatically segment where gestures begin and end"""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        from torch_geometric.nn import GCNConv
        
        # Spatial encoding
        self.gcn = GCNConv(3, hidden_dim)
        
        # Temporal boundary detection
        self.temporal = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Boundary classifier: is this frame a boundary?
        self.boundary_head = nn.Linear(hidden_dim, 2)  # Binary: boundary or not
        
        # Phase classifier: what phase of gesture?
        self.phase_head = nn.Linear(hidden_dim, 4)  # Phases: start, mid1, mid2, end
        
    def forward(self, landmarks_seq, edge_index):
        """
        Args:
            landmarks_seq: (seq_len, 21, 3)
            edge_index: hand skeleton
        
        Returns:
            boundary_scores: (seq_len, 2) - is each frame a boundary?
            phase_scores: (seq_len, 4) - what phase is each frame?
        """
        seq_len = landmarks_seq.shape[0]
        
        # Encode each frame
        frame_features = []
        for t in range(seq_len):
            x = landmarks_seq[t]
            x = self.gcn(x.unsqueeze(0), edge_index)
            x = x.mean(dim=0)
            frame_features.append(x)
        
        frame_features = torch.stack(frame_features)  # (seq_len, hidden_dim)
        
        # Temporal modeling
        lstm_out, _ = self.temporal(frame_features.unsqueeze(0))
        lstm_out = lstm_out.squeeze(0)  # (seq_len, hidden_dim)
        
        # Predict boundaries and phases
        boundary_scores = self.boundary_head(lstm_out)  # (seq_len, 2)
        phase_scores = self.phase_head(lstm_out)  # (seq_len, 4)
        
        return {
            'boundary_logits': boundary_scores,
            'phase_logits': phase_scores,
            'boundary_probs': F.softmax(boundary_scores, dim=1)[:, 1],  # P(boundary)
            'phase_probs': F.softmax(phase_scores, dim=1)  # Phase probabilities
        }

    @staticmethod
    def segment_by_threshold(boundary_probs, threshold=0.5):
        """Extract segments from boundary probabilities"""
        is_boundary = boundary_probs > threshold
        
        # Find segment boundaries
        boundaries = [0]
        for i in range(1, len(is_boundary)):
            if is_boundary[i] and not is_boundary[i-1]:
                boundaries.append(i)
        boundaries.append(len(boundary_probs))
        
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            segments.append((boundaries[i], boundaries[i+1]))
        
        return segments
```

---

## Part 8: Visualization & Analysis

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def visualize_gesture_embeddings(model, dataset, gesture_labels_rev):
    """Visualize learned gesture representations"""
    
    # Extract embeddings for all samples
    model.eval()
    embeddings = []
    labels = []
    
    for data in dataset:
        with torch.no_grad():
            # Get hidden layer output
            x, edge_index = data.x, data.edge_index
            x = model.spatial_embed(x.unsqueeze(0), edge_index)
            x = x.mean(dim=0)  # (hidden_dim,)
            embeddings.append(x.numpy())
            labels.append(data.y.item())
    
    embeddings = np.array(embeddings)  # (N, hidden_dim)
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    gesture_names = list(gesture_labels_rev.values())
    colors = plt.cm.tab10(range(8))
    
    for class_id in range(8):
        mask = np.array(labels) == class_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[class_id]],
            label=gesture_names[class_id],
            s=100,
            alpha=0.6
        )
    
    ax.legend()
    ax.set_title('Gesture Embeddings (t-SNE)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig('gesture_embeddings.png', dpi=150)
    plt.show()


def visualize_attention_weights(model, sample_data):
    """For GAT models: show which landmarks are important"""
    
    model.eval()
    with torch.no_grad():
        # Forward pass through GAT
        x, edge_index = sample_data.x, sample_data.edge_index
        
        # Hook into attention layer
        # (This requires modifying the model slightly to capture attention)
        
        # For visualization, use saliency
        x.requires_grad_(True)
        logits = model(sample_data)
        target_class = logits.argmax().item()
        logits[0, target_class].backward()
        
        # Saliency = gradient magnitude
        saliency = x.grad.norm(dim=1)
    
    # Normalize
    saliency = saliency / saliency.max()
    
    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        x[:, 0], x[:, 1], x[:, 2],
        c=saliency.detach().numpy(),
        cmap='hot',
        s=200,
        alpha=0.6
    )
    
    # Draw skeleton edges
    for src, dst in HandSkeleton.EDGES:
        src_pos = x[src].detach().numpy()
        dst_pos = x[dst].detach().numpy()
        ax.plot([src_pos[0], dst_pos[0]],
                [src_pos[1], dst_pos[1]],
                [src_pos[2], dst_pos[2]],
                'gray', alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Saliency')
    ax.set_title('Important Landmarks for Gesture Classification')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig('landmark_importance.png', dpi=150)
    plt.show()
```

---

## Part 9: Integration with Your Existing Code

### Replacing K-Means

```python
# OLD (K-Means)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8)
cluster_labels = kmeans.fit_predict(all_landmarks)

# NEW (GCN-based)
model = HandGestureGCN(num_features=3, num_classes=8)
# ... train on labeled data ...

# For unlabeled data, use soft assignments
logits = model(data)
gesture_probs = F.softmax(logits, dim=1)  # Soft assignment (21-dim)
gesture_class = gesture_probs.argmax()  # Hard assignment
confidence = gesture_probs.max()  # Confidence score
```

### Adding to Your Codebase

```python
# In your main pipeline:
from graph_models import HandGestureGCN, GestureSegmenter, HandSkeleton

# Initialize
gesture_classifier = HandGestureGCN()
gesture_segmenter = GestureSegmenter()
edge_index = HandSkeleton.get_edge_index()

# For each video:
for frame_idx, landmarks in enumerate(video_landmarks):
    data = create_hand_data(landmarks, gesture_id=None, edge_index=edge_index)
    
    # Classify gesture
    logits = gesture_classifier(data)
    gesture_class = logits.argmax()
    
    # Update segmentation
    frame_features = gesture_classifier.spatial_embed(data.x, data.edge_index)
    # ... use for temporal modeling ...
```

---

## Summary: Why These Methods Are Better

| Aspect | K-Means | GCN | T-GCN | Transformer |
|--------|---------|-----|-------|-------------|
| **Hand Structure** | ❌ | ✓ | ✓ | ✓ |
| **Temporal** | ❌ | ❌ | ✓ | ✓ |
| **Landmark Importance** | ❌ | Soft | Soft | Attention |
| **Generalize to New People** | ❌ | Partial | Better | Best |
| **Interpretability** | High | Med | Med | Med |
| **Training Time** | Instant | Hours | Hours | Days |
| **Gesture Segmentation** | ❌ | ❌ | ✓ | ✓ |
| **Robot Policy Ready** | ❌ | Partial | ✓ | ✓ |

**Recommendation**: Start with GCN (fast, interpretable), then move to T-GCN (temporal), then Transformer (if compute available).
