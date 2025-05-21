from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# Load the document
loader = TextLoader("data.txt")
docs = loader.load()

# Embed the documents
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding)

llm = HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.0})

# Build the RAG pipeline
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Run a sample query
query = "What is this document about?"
response = qa.run(query)

print(f"Query: {query}")
print(f"Answer: {response}")
