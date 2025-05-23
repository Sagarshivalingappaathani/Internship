class CriticAgent:
    @staticmethod
    def run(graph):
        if graph.x.size(0) < 2:
            print("Critic: Not enough concept nodes!")
        elif graph.edge_index.size(1) == 0:
            print("Critic: No connections between nodes!")
        else:
            print("Critic: Graph looks structurally valid.")