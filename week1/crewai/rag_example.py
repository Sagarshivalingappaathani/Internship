from sentence_transformers import SentenceTransformer, util

# Crew roles simulated as simple functions

def document_loader(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def embedder(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text, convert_to_tensor=True), model

def similarity_scorer(query_embedding, doc_embedding):
    return util.pytorch_cos_sim(query_embedding, doc_embedding).item()

def answer_generator(score, threshold=0.5):
    if score > threshold:
        return "The document is relevant to your query."
    else:
        return "The document does not closely match your query."

# === Crew Workflow ===

print("=== CrewAI Simulated RAG ===")

# Step 1: Load Document
document = document_loader("data.txt")

# Step 2: Embed Document
doc_embedding, model = embedder(document)

# Step 3: Get User Query and Embed
query = "What is this document about?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Step 4: Score Similarity
score = similarity_scorer(query_embedding, doc_embedding)

# Step 5: Generate Answer
answer = answer_generator(score)

# Output
print(f"Query: {query}")
print(f"Similarity Score: {score:.2f}")
print(f"Answer: {answer}")
