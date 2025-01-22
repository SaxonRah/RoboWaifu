1. **Hyperdimensional Computing (HDC) Encoder**
```python
class HDCEncoder:
    def __init__(self, input_dim: int, output_dim: int):
        # Create projection matrix from input dimension to high-dimensional space
        self.projection = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.projection /= np.linalg.norm(self.projection, axis=0)

    def encode(self, x: np.ndarray) -> np.ndarray:
        # Convert input (mouse position) into high-dimensional representation 
        x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        hd_vector = x_norm @ self.projection
        return np.tanh(hd_vector)
```
When you click/drag in the window, your 2D mouse position (x,y) is encoded into a 100-dimensional vector (configurable via `hd_dimension`). This encoding preserves distance relationships - similar mouse positions produce similar high-dimensional vectors. The red dots you see are the visual representation of these encoded points.

2. **Temporal Processor**
```python
class TemporalProcessor:
    def process(self, t: float, x: np.ndarray) -> np.ndarray:
        # Add current input to time buffer
        self.time_buffer.append((t, x))
        # Remove old samples outside temporal window
        cutoff_time = t - self.config.temporal_window
        self.time_buffer = [(t_i, x_i) for t_i, x_i in self.time_buffer 
                           if t_i > cutoff_time]
        
        # Apply temporal integration with exponential decay
        result = x.copy()
        for t_i, x_i in self.time_buffer[:-1]:
            weight = np.exp(-self.config.decay_rate * (t - t_i))
            result += weight * (x_i - x) * (t - t_i)
```
The temporal processor maintains a history of recent inputs with exponential decay. This means newer inputs have more influence than older ones. You can see this visualized as the green trail following your mouse movements - it shows how the system integrates information over time.

3. **Dynamic Graph**
```python
class DynamicGraph:
    def update_weights(self, features: Dict[int, np.ndarray]) -> None:
        # Update graph structure based on similarity between nodes
        for i in range(n):
            for j in range(i+1, n):
                f_i = features[self.nodes[i]]
                f_j = features[self.nodes[j]]
                # Compute similarity between nodes
                sim = self._bounded_similarity(f_i, f_j)
                # Update edge weight using gradient descent
                grad = 2 * (self.weights[i,j] - sim) + 2 * self.config.reg_lambda * self.weights[i,j]
                new_weight = self.weights[i,j] - self.config.learning_rate * grad
```
The dynamic graph maintains relationships between encoded points. When nodes are similar (based on their high-dimensional representations), they form stronger connections. You see this as blue lines between nodes - thicker/darker lines indicate stronger relationships.

The complete UCS pipeline works like this:

1. **Input Processing**:
```python
def process_input(self, pos: Tuple[int, int]) -> np.ndarray:
    # Normalize mouse coordinates
    x = np.array([pos[0] / self.config.window_width, 
                  pos[1] / self.config.window_height])
    
    # 1. Convert to high-dimensional representation
    hd_x = self.hdc.encode(x)
    
    # 2. Process temporal relationships
    self.t += 0.01
    temporal_x = self.temporal.process(self.t, hd_x)
    
    # 3. Update graph structure
    node_id = len(self.graph.nodes)
    self.graph.nodes.append(node_id)
    self.graph.node_features[node_id] = temporal_x
    self.graph.node_positions[node_id] = pos
    self.graph.update_weights(self.graph.node_features)
```

When you interact with the visualization:
1. Your mouse position is encoded into a high-dimensional vector using HDC
2. The temporal processor integrates this with recent history
3. The dynamic graph updates to reflect relationships between all points
4. The visualization shows:
   - Red dots = Encoded input points
   - Green trail = Temporal processing/history
   - Blue lines = Graph relationships between points

This matches the core UCS architecture from the reference implementation, visualizing how it:
- Maintains robust representations (HDC encoding)
- Integrates information over time (Temporal processing)
- Discovers and maintains relationships between inputs (Dynamic graph)

The system demonstrates UCS's ability to build meaningful representations of input data while preserving both temporal and spatial relationships.