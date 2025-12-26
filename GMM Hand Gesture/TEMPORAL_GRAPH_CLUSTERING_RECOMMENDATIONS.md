# Graph-Based Temporal Clustering Recommendations for Hand Gesture Recognition

## Data Structure Summary
- **40 videos per gesture** × **8 gestures** = **320 videos total**
- **150 frames per video**
- **42 landmarks per frame** (21 landmarks × 2 hands)
- **126 features per frame** (42 landmarks × 3 coordinates: X, Y, Z)
- **Zero-padding**: When only 1 hand or no hand detected

## Current Limitations
Your existing methods (GMM, K-Means, DBSCAN) and even DTW-GC have limitations:
- ❌ **No spatial structure**: Ignore hand skeleton topology
- ❌ **Limited temporal modeling**: DTW-GC uses DTW but doesn't model temporal dependencies
- ❌ **No graph evolution**: Don't capture how hand structure changes over time

---

## Recommended Graph-Based Temporal Clustering Algorithms

### 🏆 **Method 1: Temporal Graph Convolutional Network (T-GCN) Clustering** ⭐ **BEST CHOICE**

**Why This Method:**
- ✅ Combines **spatial graph** (hand skeleton) with **temporal graph** (frame sequence)
- ✅ Learns **motion patterns** automatically
- ✅ Handles **variable-length sequences**
- ✅ **Unsupervised** via contrastive learning or autoencoder

**Architecture:**
```
Input: Sequence of hand graphs
  Frame t-1: [42 nodes, skeleton edges, X/Y/Z features]
       ↓ (temporal edge)
  Frame t:   [42 nodes, skeleton edges, X/Y/Z features]
       ↓ (temporal edge)
  Frame t+1: [42 nodes, skeleton edges, X/Y/Z features]

Process:
  1. Spatial GCN: Learn hand structure at each frame
  2. Temporal GCN: Learn motion patterns across frames
  3. Sequence Embedding: Aggregate into gesture representation
  4. Clustering: K-Means/GMM in embedding space
```

**Implementation Strategy:**
```python
# Pseudo-code structure
class TemporalGCNClustering:
    def __init__(self):
        # Spatial GCN: Hand skeleton structure
        self.spatial_gcn = GraphConv(3 → 64 → 128)
        
        # Temporal GCN: Frame-to-frame connections
        self.temporal_gcn = TemporalConv(128 → 128)
        
        # Sequence encoder: LSTM/Transformer
        self.sequence_encoder = LSTM(128 → 256)
        
        # Clustering head
        self.clustering = KMeans(n_clusters=8)
    
    def forward(self, sequence_graphs):
        # For each frame: spatial encoding
        frame_embeddings = [self.spatial_gcn(graph) for graph in sequence_graphs]
        
        # Temporal encoding: learn motion
        temporal_embeddings = self.temporal_gcn(frame_embeddings)
        
        # Sequence-level embedding
        gesture_embedding = self.sequence_encoder(temporal_embeddings)
        
        return gesture_embedding
```

**Advantages:**
- Captures **both spatial and temporal** structure
- Handles **variable-speed gestures** (like DTW but better)
- **Learnable**: Adapts to gesture patterns
- **Robust**: Works across different people

**Expected Performance:**
- Silhouette Score: **0.3-0.5** (vs 0.1-0.2 for GMM)
- Better separation of dynamic gestures (Wave, Come)

---

### 🥈 **Method 2: Dynamic Graph Clustering (DGC)**

**Why This Method:**
- ✅ Builds **evolving graphs** over time
- ✅ Captures **gesture phases** (start → middle → end)
- ✅ **Unsupervised** via graph similarity

**Core Idea:**
Instead of static graphs, build **dynamic graphs** where:
- Nodes = hand landmarks (42 nodes)
- Edges = spatial connections (skeleton) + temporal connections (frame-to-frame)
- Graph evolves as gesture progresses

**Algorithm:**
```
1. Build Spatial Graph: Connect landmarks based on hand skeleton
2. Build Temporal Graph: Connect corresponding landmarks across frames
3. Dynamic Graph: Combine spatial + temporal edges
4. Graph Embedding: Use Graph2Vec or Graph Neural Network
5. Clustering: Cluster graph embeddings
```

**Implementation:**
```python
class DynamicGraphClustering:
    def build_dynamic_graph(self, sequence):
        """
        Build graph that evolves over time
        
        Nodes: 42 landmarks × 150 frames = 6300 nodes
        Spatial edges: Hand skeleton connections (within frame)
        Temporal edges: Same landmark across frames
        """
        G = nx.DiGraph()
        
        # Add nodes: (landmark_id, frame_id)
        for frame_idx in range(len(sequence)):
            for landmark_idx in range(42):
                G.add_node((landmark_idx, frame_idx))
        
        # Spatial edges: skeleton connections within frame
        for frame_idx in range(len(sequence)):
            for edge in HAND_SKELETON_EDGES:
                G.add_edge((edge[0], frame_idx), (edge[1], frame_idx))
        
        # Temporal edges: same landmark across frames
        for landmark_idx in range(42):
            for frame_idx in range(len(sequence) - 1):
                G.add_edge(
                    (landmark_idx, frame_idx),
                    (landmark_idx, frame_idx + 1),
                    weight=self._compute_temporal_weight(...)
                )
        
        return G
    
    def cluster(self, dynamic_graphs):
        # Use Graph2Vec or GNN to embed graphs
        embeddings = [self.graph_encoder(G) for G in dynamic_graphs]
        labels = KMeans(n_clusters=8).fit_predict(embeddings)
        return labels
```

**Advantages:**
- Captures **gesture evolution** over time
- Models **phase transitions** (start → middle → end)
- **Interpretable**: Can visualize graph evolution

---

### 🥉 **Method 3: Spectral Temporal Clustering (STC)**

**Why This Method:**
- ✅ Extends **Graph Spectral Clustering** to temporal domain
- ✅ Uses **spectral graph theory** for both spatial and temporal
- ✅ **Unsupervised** and mathematically grounded

**Core Idea:**
Build **two Laplacians**:
1. **Spatial Laplacian**: Hand structure within frame
2. **Temporal Laplacian**: Frame-to-frame relationships

Combine them for joint spectral clustering.

**Algorithm:**
```
1. Spatial Graph: Build k-NN graph for each frame (hand landmarks)
2. Temporal Graph: Build graph connecting frames (sequence similarity)
3. Spatial Laplacian: L_spatial = D_spatial - W_spatial
4. Temporal Laplacian: L_temporal = D_temporal - W_temporal
5. Joint Laplacian: L_joint = α·L_spatial + (1-α)·L_temporal
6. Spectral Clustering: Use eigenvectors of L_joint
```

**Implementation:**
```python
class SpectralTemporalClustering:
    def __init__(self, n_clusters=8, alpha=0.5):
        self.n_clusters = n_clusters
        self.alpha = alpha  # Balance spatial vs temporal
    
    def build_spatial_graph(self, frame):
        """Build graph for hand structure in one frame"""
        # k-NN graph of landmarks
        knn_graph = kneighbors_graph(frame, n_neighbors=5)
        return knn_graph
    
    def build_temporal_graph(self, sequences):
        """Build graph connecting frames"""
        # DTW or Euclidean similarity between frames
        n_frames = len(sequences)
        W = np.zeros((n_frames, n_frames))
        for i in range(n_frames):
            for j in range(i+1, n_frames):
                similarity = self._compute_similarity(sequences[i], sequences[j])
                W[i, j] = W[j, i] = similarity
        return W
    
    def fit_predict(self, sequences):
        # Build spatial Laplacian for each frame
        L_spatial_list = [self._compute_laplacian(self.build_spatial_graph(f)) 
                          for f in frames]
        L_spatial = block_diag(L_spatial_list)  # Block diagonal
        
        # Build temporal Laplacian
        W_temporal = self.build_temporal_graph(sequences)
        L_temporal = self._compute_laplacian(W_temporal)
        
        # Joint Laplacian
        L_joint = self.alpha * L_spatial + (1 - self.alpha) * L_temporal
        
        # Spectral clustering
        eigenvalues, eigenvectors = eigsh(L_joint, k=self.n_clusters)
        labels = KMeans(n_clusters=self.n_clusters).fit_predict(eigenvectors)
        
        return labels
```

**Advantages:**
- **Mathematically principled**: Spectral graph theory
- **Tunable**: Balance spatial vs temporal via α
- **No training needed**: Direct computation

---

### **Method 4: Graph Attention Temporal Clustering (GATC)**

**Why This Method:**
- ✅ Combines **Graph Attention** (spatial) with **Temporal Attention** (temporal)
- ✅ **Attention mechanism** focuses on important landmarks and frames
- ✅ **Learnable** embeddings

**Architecture:**
```
Input: Sequence of hand graphs
  ↓
Spatial Attention: Which landmarks are important? (GAT)
  ↓
Temporal Attention: Which frames are important? (Transformer)
  ↓
Gesture Embedding: Weighted combination
  ↓
Clustering: K-Means in embedding space
```

**Implementation:**
```python
class GraphAttentionTemporalClustering:
    def __init__(self):
        # Spatial: Graph Attention Network
        self.spatial_gat = GATConv(3, 64, heads=4)
        
        # Temporal: Transformer encoder
        self.temporal_transformer = nn.TransformerEncoder(...)
        
        # Clustering
        self.clustering = KMeans(n_clusters=8)
    
    def forward(self, sequence_graphs):
        # Spatial attention: learn landmark importance
        frame_embeddings = []
        for graph in sequence_graphs:
            embedding = self.spatial_gat(graph.x, graph.edge_index)
            frame_embeddings.append(embedding.mean(dim=0))
        
        # Temporal attention: learn frame importance
        sequence_tensor = torch.stack(frame_embeddings)
        gesture_embedding = self.temporal_transformer(sequence_tensor)
        
        return gesture_embedding
```

**Advantages:**
- **Interpretable**: Can visualize attention weights
- **Adaptive**: Learns which landmarks/frames matter
- **Robust**: Handles missing landmarks gracefully

---

## Comparison Table

| Method | Spatial Structure | Temporal Modeling | Complexity | Expected Silhouette | Best For |
|--------|-------------------|-------------------|------------|---------------------|----------|
| **T-GCN** | ✅ Full skeleton | ✅ LSTM/RNN | High | **0.3-0.5** | Dynamic gestures |
| **DGC** | ✅ Evolving graph | ✅ Graph evolution | Medium | **0.25-0.4** | Gesture phases |
| **STC** | ✅ k-NN graph | ✅ DTW similarity | Low | **0.2-0.35** | Quick implementation |
| **GATC** | ✅ Attention | ✅ Transformer | High | **0.3-0.45** | Interpretability |

---

## Implementation Priority

### **Phase 1: Start with Spectral Temporal Clustering (STC)** ⭐
**Why:** Easiest to implement, extends your existing Graph Spectral Clustering
**Time:** 1-2 days
**Expected Improvement:** +50-75% over GMM

### **Phase 2: Implement Temporal GCN (T-GCN)**
**Why:** Best performance, captures both spatial and temporal
**Time:** 1-2 weeks
**Expected Improvement:** +100-150% over GMM

### **Phase 3: Advanced Methods (DGC, GATC)**
**Why:** Further improvements, interpretability
**Time:** 2-3 weeks each

---

## Key Implementation Details

### **Hand Skeleton Structure (42 landmarks = 2 hands)**

```python
# Hand skeleton edges (21 landmarks per hand)
HAND_SKELETON_EDGES = [
    # Wrist to finger bases
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),  # Left hand
    (21, 22), (21, 26), (21, 30), (21, 34), (21, 38),  # Right hand
    
    # Thumb (4 joints)
    (1, 2), (2, 3), (3, 4),
    (22, 23), (23, 24), (24, 25),
    
    # Index finger (4 joints)
    (5, 6), (6, 7), (7, 8),
    (26, 27), (27, 28), (28, 29),
    
    # Middle finger (4 joints)
    (9, 10), (10, 11), (11, 12),
    (30, 31), (31, 32), (32, 33),
    
    # Ring finger (4 joints)
    (13, 14), (14, 15), (15, 16),
    (34, 35), (35, 36), (36, 37),
    
    # Pinky (4 joints)
    (17, 18), (18, 19), (19, 20),
    (38, 39), (39, 40), (40, 41),
]
```

### **Sequence Segmentation**

```python
def segment_videos(data, landmarks_per_frame=42):
    """
    Segment 320 videos into sequences
    Each video: 150 frames × 42 landmarks = 6300 rows
    """
    sequences = []
    video_length = 150 * landmarks_per_frame  # 6300 rows per video
    
    for video_idx in range(320):  # 40 videos × 8 gestures
        start_idx = video_idx * video_length
        end_idx = start_idx + video_length
        video_data = data[start_idx:end_idx]
        
        # Reshape: (6300, 3) → (150, 42, 3) → (150, 126)
        frames = video_data.reshape(150, landmarks_per_frame, 3)
        frames_flat = frames.reshape(150, landmarks_per_frame * 3)
        
        # Remove zero-padding frames
        non_zero_mask = ~np.all(np.abs(frames_flat) < 1e-6, axis=1)
        sequence = frames_flat[non_zero_mask]
        
        if len(sequence) >= 10:  # Minimum sequence length
            sequences.append(sequence)
    
    return sequences
```

---

## Next Steps

1. **Implement Spectral Temporal Clustering** (quickest win)
2. **Evaluate on your 320 videos** (40 per gesture)
3. **Compare with GMM/K-Means/DBSCAN baseline**
4. **Move to T-GCN** if results are promising

Would you like me to implement any of these methods? I recommend starting with **Spectral Temporal Clustering** as it's the easiest to implement and will give immediate improvements.

