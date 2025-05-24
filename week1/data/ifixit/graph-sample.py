import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Load Appliance Data ===
with open("Appliance.json", "r") as f:
    documents = [json.loads(line) for line in f if line.strip()]

# === Extract and clean step text ===
def extract_steps(doc):
    steps = []
    for step in doc.get("Steps", []):
        # Prefer Text_raw if available
        if "Text_raw" in step:
            steps.append(step["Text_raw"].strip())
        elif "Lines" in step:
            text_lines = [line.get("Text", "") for line in step["Lines"] if "Text" in line]
            combined = " ".join(text_lines).strip()
            if combined:
                steps.append(combined)
    return steps


# === Step Filter: remove very short or empty sentences ===
def filter_useful_steps(steps):
    return [s for s in steps if len(s.split()) > 2]

# === Get steps from both documents ===
doc1_steps = filter_useful_steps(extract_steps(documents[21]))
doc2_steps = filter_useful_steps(extract_steps(documents[399]))

# === Merge & check validity ===
sentences = doc1_steps + doc2_steps
if not sentences:
    raise ValueError("No valid sentences for TF-IDF. Check the step content in docs 21 & 399.")

# === Vectorize via TF-IDF ===
vectorizer = TfidfVectorizer(stop_words='english', min_df=1, token_pattern=r"(?u)\b\w+\b")
tfidf = vectorizer.fit_transform(sentences)
sim_matrix = cosine_similarity(tfidf)

# === Build node list ===
nodes = [f"doc1_s{i}" for i in range(len(doc1_steps))] + [f"doc2_s{i}" for i in range(len(doc2_steps))]

# === Create JCIG graph ===
G = nx.DiGraph()
for i, node in enumerate(nodes):
    G.add_node(node, text=sentences[i])

# === Combined Hamiltonian Path (C-HP): Add one best outgoing edge per node ===
for i, node_i in enumerate(nodes):
    sims = list(enumerate(sim_matrix[i]))
    sims = sorted([(j, score) for j, score in sims if j != i], key=lambda x: -x[1])
    if sims:
        j, weight = sims[0]
        G.add_edge(node_i, nodes[j], weight=weight)

# === Visualize JCIG ===
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(16, 10))
edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
nx.draw(G, pos, with_labels=True, node_size=600, font_size=8, arrows=True,
        node_color="skyblue", edge_color=edge_weights, edge_cmap=plt.cm.Blues, width=2)
labels = {n: G.nodes[n]["text"][:20] + "..." for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)
plt.title("JCIG (Combined + Hamiltonian Path) for Docs 21 & 399")
# plt.tight_layout()
plt.savefig("jcig_hp_output.png", dpi=300)
print("Graph saved as jcig_hp_output.png")

