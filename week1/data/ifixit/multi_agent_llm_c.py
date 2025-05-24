import os, json, pandas as pd, torch, networkx as nx
import torch.nn.functional as F
import numpy as np
from openai import OpenAI
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging
from collections import Counter
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

TARGET_FEATURE_DIM = 384

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sentence_embeddings(concepts, doc_a="", doc_b=""):
    """
    Create semantic embeddings using sentence transformers with document context.
    Always pads/truncates to TARGET_FEATURE_DIM for consistency.
    """

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')  
        concept_texts = []
        for concept_group in concepts:
            if isinstance(concept_group, list):
                text = " ".join(str(c) for c in concept_group)
            else:
                text = str(concept_group)
            # Add context from documents for better embeddings
            context_snippets = []
            for doc in [doc_a, doc_b]:
                if doc and text.lower() in doc.lower():
                    idx = doc.lower().find(text.lower())
                    start = max(0, idx - 50)
                    end = min(len(doc), idx + len(text) + 50)
                    context_snippets.append(doc[start:end])
            if context_snippets:
                enriched_text = f"{text}. Context: {' | '.join(context_snippets[:2])}"
            else:
                enriched_text = text
            concept_texts.append(enriched_text)

        embeddings = model.encode(concept_texts)
        feat = torch.tensor(embeddings, dtype=torch.float32)

        # Pad/truncate for consistency
        if feat.shape[1] < TARGET_FEATURE_DIM:
            pad = torch.zeros((feat.shape[0], TARGET_FEATURE_DIM - feat.shape[1]), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=1)
        elif feat.shape[1] > TARGET_FEATURE_DIM:
            feat = feat[:, :TARGET_FEATURE_DIM]
        return feat
    except Exception as e:
        logger.error(f"Error creating sentence embeddings: {e}")
        fallback = create_enhanced_simple_features(concepts, doc_a, doc_b)
        # Also pad fallback
        current_dim = fallback.shape[1]
        if current_dim < TARGET_FEATURE_DIM:
            pad = torch.zeros((fallback.shape[0], TARGET_FEATURE_DIM - current_dim), dtype=fallback.dtype)
            fallback = torch.cat([fallback, pad], dim=1)
        elif current_dim > TARGET_FEATURE_DIM:
            fallback = fallback[:, :TARGET_FEATURE_DIM]
        return fallback

# === LLM Agent Base ===
class GPT4oMiniAgent:
    def __init__(self, farm_key):
        self.client = OpenAI(
            api_key="dummy",
            base_url="https://aoai-farm.bosch-temp.com/api/openai/deployments/askbosch-prod-farm-openai-gpt-4o-mini-2024-07-18",
            default_headers={"genaiplatform-farm-subscription-key": farm_key}
        )

    def prompt(self, sys_prompt, user_prompt):
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                extra_query={"api-version": "2024-08-01-preview"}
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return None

class JSONCleanerAgent:
    @staticmethod
    def clean(text):
        if not text:
            return None
        try:
            if text.startswith("```json"):
                text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return None

# === Enhanced Concept Extraction ===
class EnhancedConceptAgent(GPT4oMiniAgent):
    def extract_concepts(self, doc_a, doc_b):
        sys_prompt = """You are an expert at identifying procedural concepts in technical documents. 
        Focus on actionable concepts, tools, components, and processes that are central to the procedures.
        Group semantically similar concepts together. Respond with raw JSON only."""
        
        user_prompt = f"""
Document A (first 1000 chars):
{doc_a[:1000]}

Document B (first 1000 chars):
{doc_b[:1000]}

Extract and group procedural concepts that are important for understanding document similarity.
Focus on:
1. Key actions/procedures
2. Important tools/components
3. Critical processes
4. Domain-specific terms

Group similar concepts into clusters. Each cluster should contain 1-4 related concepts.
Return 8-15 concept clusters. Respond with raw JSON only:
{{
  "concepts": [["concept1", "related_concept2"], ["concept3"], ["concept4", "concept5"]]
}}
"""
        resp = self.prompt(sys_prompt, user_prompt)
        logger.info(f"🧠 Enhanced ConceptAgent Response: {resp}")
        
        if not resp:
            return self._fallback_concept_extraction(doc_a, doc_b)
        
        parsed = JSONCleanerAgent.clean(resp)
        if not parsed or "concepts" not in parsed:
            logger.error("Invalid concept response format, using fallback")
            return self._fallback_concept_extraction(doc_a, doc_b)
        
        # Validate and clean concepts
        concepts = parsed["concepts"]
        cleaned_concepts = []
        for concept_group in concepts:
            if isinstance(concept_group, list) and len(concept_group) > 0:
                cleaned_group = [str(c).strip() for c in concept_group if str(c).strip()]
                if cleaned_group:
                    cleaned_concepts.append(cleaned_group)
            elif isinstance(concept_group, str) and concept_group.strip():
                cleaned_concepts.append([concept_group.strip()])
        
        return cleaned_concepts[:15]  # Limit to 15 concepts
    
    def _fallback_concept_extraction(self, doc_a, doc_b):
        """Simple fallback concept extraction using keyword extraction"""
        try:
            # Combine documents
            combined_text = f"{doc_a} {doc_b}".lower()
            
            # Simple keyword extraction (you could enhance this)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
            word_freq = Counter(words)
            
            # Get most common words as concepts
            common_words = [word for word, freq in word_freq.most_common(10) if freq > 1]
            
            # Return as individual concepts
            return [[word] for word in common_words]
            
        except Exception as e:
            logger.error(f"Fallback concept extraction failed: {e}")
            return [["default_concept"]]

# === Enhanced Edge Detection ===
class EnhancedEdgeAgent(GPT4oMiniAgent):
    def identify_edges(self, concepts, doc_a, doc_b):
        sys_prompt = """You identify semantic and procedural relationships between concept clusters.
        Focus on meaningful connections that indicate document similarity. Respond with raw JSON only."""
        
        concept_summary = []
        for i, concept_group in enumerate(concepts):
            if isinstance(concept_group, list):
                summary = ", ".join(str(c) for c in concept_group[:3])
            else:
                summary = str(concept_group)
            concept_summary.append(f"{i}: {summary}")
        
        user_prompt = f"""
Concept clusters:
{chr(10).join(concept_summary)}

Document context (combined, first 800 chars):
{(doc_a + " " + doc_b)[:800]}

Identify relationships between concept clusters based on:
1. Semantic similarity
2. Procedural dependencies
3. Co-occurrence patterns
4. Functional relationships

Return edges as cluster index pairs. Aim for 5-20 edges to create a well-connected graph.
Raw JSON only:
{{
  "edges": [[0, 1], [1, 2], [0, 3], [2, 4]]
}}
"""
        resp = self.prompt(sys_prompt, user_prompt)
        logger.info(f"🔗 Enhanced EdgeAgent Response: {resp}")
        
        if not resp:
            return self._create_fallback_edges(len(concepts))
        
        parsed = JSONCleanerAgent.clean(resp)
        if not parsed or "edges" not in parsed:
            logger.error("Invalid edge response format, using fallback")
            return self._create_fallback_edges(len(concepts))
        
        # Validate edges
        max_idx = len(concepts) - 1
        valid_edges = []
        for edge in parsed["edges"]:
            if (isinstance(edge, list) and len(edge) == 2 and 
                isinstance(edge[0], int) and isinstance(edge[1], int) and
                0 <= edge[0] <= max_idx and 0 <= edge[1] <= max_idx and
                edge[0] != edge[1]):
                valid_edges.append(edge)
        
        return valid_edges if valid_edges else self._create_fallback_edges(len(concepts))
    
    def _create_fallback_edges(self, num_concepts):
        """Create a simple connected graph as fallback"""
        if num_concepts < 2:
            return []
        
        edges = []
        # Create a path to ensure connectivity
        for i in range(num_concepts - 1):
            edges.append([i, i + 1])
        
        # Add some additional edges for better connectivity
        if num_concepts > 3:
            edges.append([0, num_concepts - 1])  # Connect first to last
            if num_concepts > 4:
                edges.append([1, num_concepts - 2])  # Additional connection
        
        return edges

# === NEW: Direction Agent ===
class DirectionAgent(GPT4oMiniAgent):
    def deduce_directions(self, edges, concepts, doc_a, doc_b):
        """
        Deduce edge directions based on conceptual relationships
        """
        sys_prompt = """You analyze concept relationships to determine edge directions in a graph.
        Consider semantic dependencies, procedural order, and causal relationships.
        Respond with raw JSON only."""
        
        concept_summary = []
        for i, concept_group in enumerate(concepts):
            if isinstance(concept_group, list):
                summary = ", ".join(str(c) for c in concept_group[:3])
            else:
                summary = str(concept_group)
            concept_summary.append(f"{i}: {summary}")
        
        edge_descriptions = []
        for edge in edges:
            if len(edge) == 2:
                i, j = edge
                concept_i = concept_summary[i] if i < len(concept_summary) else f"{i}: unknown"
                concept_j = concept_summary[j] if j < len(concept_summary) else f"{j}: unknown"
                edge_descriptions.append(f"[{i}, {j}]: {concept_i} <-> {concept_j}")
        
        user_prompt = f"""
Concepts:
{chr(10).join(concept_summary)}

Edges to analyze:
{chr(10).join(edge_descriptions)}

For each edge, determine if there's a clear directional relationship:
- Procedural dependency (A must happen before B)
- Causal relationship (A causes or enables B)
- Semantic hierarchy (A is a type of B, or A contains B)

If unclear, keep bidirectional. Return directed edges:
{{
  "directed_edges": [[0, 1], [2, 3], [1, 4]]
}}
"""
        try:
            resp = self.prompt(sys_prompt, user_prompt)
            logger.info(f"🧭 DirectionAgent Response: {resp}")
            
            if not resp:
                return edges  # Return original undirected edges
            
            parsed = JSONCleanerAgent.clean(resp)
            if not parsed or "directed_edges" not in parsed:
                logger.error("Invalid direction response format, using original edges")
                return edges
            
            # Validate directed edges
            max_idx = len(concepts) - 1
            valid_directed_edges = []
            for edge in parsed["directed_edges"]:
                if (isinstance(edge, list) and len(edge) == 2 and 
                    isinstance(edge[0], int) and isinstance(edge[1], int) and
                    0 <= edge[0] <= max_idx and 0 <= edge[1] <= max_idx and
                    edge[0] != edge[1]):
                    valid_directed_edges.append(edge)
            
            return valid_directed_edges if valid_directed_edges else edges
            
        except Exception as e:
            logger.error(f"Direction detection failed: {e}")
            return edges  # Return original edges

# Use enhanced agents
ConceptAgent = EnhancedConceptAgent
EdgeAgent = EnhancedEdgeAgent

# === Enhanced JCIG Graph Builder ===
def build_enhanced_jcig(doc_a, doc_b, label, concept_agent, edge_agent, direction_agent, feature_type="cross_document"):
    """
    Build enhanced JCIG with better error handling and features
    """
    try:
        # Extract concepts with retry mechanism
        concepts = None
        for attempt in range(2):  # Try twice
            concepts = concept_agent.extract_concepts(doc_a, doc_b)
            if concepts and len(concepts) >= 2:
                break
            logger.warning(f"Attempt {attempt + 1}: Insufficient concepts extracted")
        
        if not concepts or len(concepts) < 2:
            logger.warning("Failed to extract sufficient concepts")
            return None
            
        # Extract edges with retry
        edges = None
        for attempt in range(2):
            edges = edge_agent.identify_edges(concepts, doc_a, doc_b)
            if edges:
                break
            logger.warning(f"Attempt {attempt + 1}: No edges identified")
        
        if not edges:
            logger.warning("Failed to identify edges")
            return None
        
        # Get directed edges (optional - can work with undirected too)
        directed_edges = edges  # Use undirected edges directly
        try:
            directed_edges = direction_agent.deduce_directions(edges, concepts, doc_a, doc_b)
            if not directed_edges:
                directed_edges = edges  # Fallback to undirected
        except Exception as e:
            logger.warning(f"Direction detection failed, using undirected edges: {e}")
            directed_edges = edges

        # Build NetworkX graph for validation
        G = nx.DiGraph() if directed_edges != edges else nx.Graph()
        for idx, group in enumerate(concepts):
            label_text = ', '.join(str(item) for item in group[:2]) if isinstance(group, list) else str(group)
            G.add_node(idx, label=label_text)
        
        # Validate and add edges
        max_node_idx = len(concepts) - 1
        valid_edges = []
        for edge in directed_edges:
            if len(edge) == 2:
                u, v = edge
                if isinstance(u, int) and isinstance(v, int) and 0 <= u <= max_node_idx and 0 <= v <= max_node_idx:
                    valid_edges.append((u, v))
                else:
                    logger.warning(f"Invalid edge indices: {edge}")
        
        if not valid_edges:
            logger.warning("No valid edges after validation")
            return None
        
        for u, v in valid_edges:
            G.add_edge(u, v)
        
        # Ensure graph connectivity
        if not nx.is_connected(G.to_undirected()):
            logger.info("Graph is not connected, adding connectivity edges")
            components = list(nx.connected_components(G.to_undirected()))
            for i in range(len(components) - 1):
                u = list(components[i])[0]
                v = list(components[i + 1])[0]
                G.add_edge(u, v)
                valid_edges.append((u, v))
        
        # === CREATE ENHANCED NODE FEATURES ===
        x = create_sentence_embeddings(concepts, doc_a, doc_b)
        print("x of the document", x.size())
        
        # Convert to PyTorch Geometric format
        edge_index = torch.tensor(valid_edges).t().long()
        y = torch.tensor([float(label)], dtype=torch.float32)
        
        # Add graph-level features
        num_nodes = len(concepts)
        num_edges = len(valid_edges)
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
        density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        
        graph_data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            num_nodes=num_nodes,
            # Store additional graph statistics
            graph_stats=torch.tensor([avg_degree, density, num_edges], dtype=torch.float32)
        )
        
        return graph_data
        
    except Exception as e:
        logger.error(f"Enhanced graph building failed: {e}")
        return None

class JCIG_GCN_MLP(nn.Module):
    def __init__(self, in_dim=384, gcn_hidden=128, mlp_hidden=64, num_classes=1, dropout=0.3):
        super(JCIG_GCN_MLP, self).__init__()
        self.gcn1 = GCNConv(in_dim, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, gcn_hidden)
        self.gcn3 = GCNConv(gcn_hidden, gcn_hidden)
        
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(gcn_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gcn3(x, edge_index))

        # Global mean pooling to get graph embedding
        x = global_mean_pool(x, batch)
        out = self.mlp(x)
        return out

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch).squeeze()
        loss = criterion(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze()  

            # Ensure out and batch.y are always 1D tensors
            if out.dim() == 0:
                out = out.unsqueeze(0)
            if batch.y.dim() == 0:
                batch_y = batch.y.unsqueeze(0)
            else:
                batch_y = batch.y

            ys.append(batch_y.cpu())
            preds.append(torch.sigmoid(out).cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    y_pred_label = (y_pred > 0.5).astype(int)
    acc = (y_true == y_pred_label).mean()
    return acc


# === Enhanced Main Function ===
def main():
    # Configuration
    farm_key = os.getenv("FARM_API_KEY", "0c9f370035e0436989bb962b0d1bb9d0")
    path = "./"
    feature_type = "sentence" 
    
    logger.info(f"🚀 Starting Enhanced JCIG Document Similarity System")
    logger.info(f"Using feature type: {feature_type}")
    
    # Check required files
    appliance_file = os.path.join(path, "Appliance.json")
    pairs_file = os.path.join(path, "labeled_pairs_rule_based.csv")
    
    # Load data
    try:
        with open(appliance_file) as f:
            documents = [json.loads(line) for line in f if line.strip()]
        pairs_df = pd.read_csv(pairs_file)
        logger.info(f"✅ Loaded {len(documents)} documents and {len(pairs_df)} pairs")
    except Exception as e:
        logger.error(f"Failed to load data files: {e}")
        return

    # Initialize agents
    concept_agent = ConceptAgent(farm_key)
    edge_agent = EdgeAgent(farm_key)
    direction_agent = DirectionAgent(farm_key)

    # Build graphs with progress tracking
    graphs, labels = [], []
    logger.info("🔨 Building enhanced graphs...")
    
    successful_builds = 0
    total_pairs = len(pairs_df)
    
    for idx, row in enumerate(pairs_df.itertuples()):
        logger.info(f"Processing pair {idx + 1}/{total_pairs}")
        if(idx >= 10):
            break
        try:
            # Extract document text
            doc_a_steps = documents[row.doc1_id].get("Steps", [])
            doc_b_steps = documents[row.doc2_id].get("Steps", [])
            
            doc_a = " ".join([s.get("Text_raw", "") for s in doc_a_steps])
            doc_b = " ".join([s.get("Text_raw", "") for s in doc_b_steps])
            
            if not doc_a.strip() or not doc_b.strip():
                logger.warning(f"Empty documents for pair {row.Index}")
                continue
            
            # Build enhanced graph
            graph = build_enhanced_jcig(doc_a, doc_b, row.label, concept_agent, edge_agent, direction_agent, feature_type)
            if graph is not None:
                graphs.append(graph)
                labels.append(row.label)
                successful_builds += 1
                logger.info(f"✅ Pair {row.Index} processed successfully "
                          f"(nodes: {graph.num_nodes}, edges: {graph.edge_index.shape[1]})")
            else:
                logger.warning(f"❌ Failed to create graph for pair {row.Index}")
                
        except Exception as e:
            logger.error(f"Error processing pair {row.Index}: {e}")

    BATCH_SIZE = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_graphs = len(graphs)
    N_train = int(0.8 * total_graphs)
    N_test  = total_graphs - N_train

    train_loader = DataLoader(graphs[:N_train], batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(graphs[N_train:], batch_size=BATCH_SIZE)

    model = JCIG_GCN_MLP(in_dim=TARGET_FEATURE_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, 51):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        # Remove val_acc line since no validation set
        print(f'Epoch {epoch}, Loss: {train_loss:.4f}')

    # Add final test evaluation after training
    test_acc = test(model, test_loader, device)
    print(f'Final Test Accuracy: {test_acc:.4f}')

    logger.info(f"🎯 Successfully created {len(graphs)} graphs out of {total_pairs} pairs "
              f"({len(graphs)/total_pairs*100:.1f}% success rate)")

if __name__ == "__main__":
    main()