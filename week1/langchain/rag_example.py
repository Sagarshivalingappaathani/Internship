from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline


def build_vector_store(file_path: str, embedding_model=None):
    # Load the text file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    # Create embeddings using an open-source SentenceTransformer
    embeddings = embedding_model or HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build the vector store (in-memory FAISS)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def main():
    # Path to your data file
    data_file = "data.txt"

    # Build FAISS vector store from data
    vector_store = build_vector_store(data_file)

    # Initialize a Hugging Face text-generation pipeline
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",  
        device=0,                        
        max_length=256,
        do_sample=False
    )

    # Wrap it for LangChain
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    print("Simple open-source RAG with LangChain. Type your questions below (type 'exit' to quit).\n")
    while True:
        query = input("Question: ")
        if query.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Get answer from the chain
        answer = qa_chain.run(query)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
