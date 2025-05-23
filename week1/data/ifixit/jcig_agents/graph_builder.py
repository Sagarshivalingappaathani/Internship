import torch
from torch_geometric.data import Data
import numpy as np

class GraphBuilderAgent:
    @staticmethod
    def run(concepts, edges):
        x = torch.tensor([np.mean([concepts['embeddings'][i] for i in inds], axis=0) for inds in concepts['groups']], dtype=torch.float)
        edge_index = torch.tensor([[src, dst] for src, dst, _ in edges], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor([w for _, _, w in edges], dtype=torch.float)
        y = torch.tensor([1], dtype=torch.float)  # dummy label for now
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        print("\n✅ Final Graph Built: ", graph)
        return graph