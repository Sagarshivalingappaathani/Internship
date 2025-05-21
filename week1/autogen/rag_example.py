from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from sentence_transformers import SentenceTransformer, util
import torch

# Load the document
with open("data.txt", "r") as f:
    doc_text = f.read()

# Embed document using sentence-transformers
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embedding = model.encode(doc_text, convert_to_tensor=True)

# Simulate query and embedding comparison
query = "What is this document about?"
query_embedding = model.encode(query, convert_to_tensor=True)
score = util.pytorch_cos_sim(query_embedding, doc_embedding).item()

# Define agents
user = UserProxyAgent(name="User")
retriever = AssistantAgent(name="Retriever")
answerer = AssistantAgent(name="Answerer")

# Simulate chat context (can be elaborated for more interactivity)
chat = GroupChat(agents=[user, retriever, answerer], messages=[], max_round=3)
manager = GroupChatManager(groupchat=chat, llm_config=False)

# Message simulation
print("=== AutoGen Simulated RAG ===")
print(f"User Query: {query}")
print(f"Similarity Score with Document: {score:.2f}")

# Basic logic for interpretation (you can extend this logic as needed)
if score > 0.5:
    print("Answer: The document is semantically relevant to the query.")
else:
    print("Answer: The document does not closely match the query.")
