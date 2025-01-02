# Graph creation and manipulation

# Dynamic Graph Construction and Management
class DynamicGraph:
    def __init__(self):
        """
        Initialize a dynamic graph.
        """
        self.nodes = {}
        self.edges = {}

    def add_nodes(self, nodes):
        """
        Add nodes to the graph.
        :param nodes: List of node identifiers.
        """
        for node in nodes:
            if node not in self.nodes:
                self.nodes[node] = {}

    def add_edges(self, edges, weights):
        """
        Add edges with weights to the graph.
        :param edges: List of tuples representing edges (e.g., [("A", "B"), ("B", "C")]).
        :param weights: List of weights corresponding to the edges.
        """
        for (edge, weight) in zip(edges, weights):
            src, dst = edge
            self.edges[(src, dst)] = weight

    def update_weights(self, feedback):
        """
        Update edge weights based on feedback.
        :param feedback: Dictionary where keys are edge tuples and values are weight adjustments.
        """
        for edge, adjustment in feedback.items():
            if edge in self.edges:
                self.edges[edge] += adjustment
