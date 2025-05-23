import json
import torch
import networkx as nx
import matplotlib.pyplot as plt

from jcig_agents.step_extractor import StepExtractorAgent
from jcig_agents.embed_cluster import EmbedClusterAgent
from jcig_agents.edge_builder import EdgeBuilderAgent
from jcig_agents.graph_builder import GraphBuilderAgent
from jcig_agents.critic_agent import CriticAgent

# === Load documents ===
with open("Appliance.json") as f:
    documents = [json.loads(line) for line in f if line.strip()]

doc1_id, doc2_id = 21, 399

# === Run jcig_agents ===
steps = StepExtractorAgent.run(documents[doc1_id], documents[doc2_id])
clusters = EmbedClusterAgent.run(steps)
edges = EdgeBuilderAgent.run(clusters)
graph = GraphBuilderAgent.run(clusters, edges)

# === Final validation ===
CriticAgent.run(graph)

# === Visualization Logic ===
def visualize_graph(graph_data):
    print("skpsmpsap")
    G = nx.DiGraph()
    num_nodes = graph_data.x.shape[0]

    for i in range(num_nodes):
        G.add_node(i, label=f"Concept {i}")

    edge_index = graph_data.edge_index
    edge_attr = graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else None

    for idx in range(edge_index.shape[1]):
        src = int(edge_index[0, idx])
        dst = int(edge_index[1, idx])
        weight = float(edge_attr[idx]) if edge_attr is not None else 1.0
        G.add_edge(src, dst, weight=round(weight, 2))

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("JCIG Concept Graph (Built by Multi-Agent System)")
    plt.savefig("jcig_graph.png")
    print("✅ Graph saved as jcig_graph.png")


# === Call graph plot ===
visualize_graph(graph)
