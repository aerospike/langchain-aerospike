"""Basic usage example for the langchain-aerospike package."""

from langchain_aerospike.vectorstores import Aerospike
from langchain_openai import OpenAIEmbeddings
from aerospike_vector_search import Client, HostPort

# Initialize the Aerospike client
# Replace with your Aerospike server connection details
client = Client(seeds=[HostPort(host="localhost", port=3000)])

# Initialize the embeddings model
# You need an OpenAI API key for this to work
embedding_model = OpenAIEmbeddings()

# Create an Aerospike vector store
vector_store = Aerospike(
    client=client,
    embedding=embedding_model,
    namespace="test",  # Replace with your namespace
    set_name="vectors",  # Replace with your set name
    text_key="text",
    metadata_key="metadata",
    vector_key="vector",
)

# Add documents to the vector store
texts = [
    "Aerospike is a real-time, distributed NoSQL database and vector database",
    "Vector databases store and retrieve vector embeddings for AI applications",
    "LangChain is a framework for developing applications powered by language models",
]
metadatas = [
    {"source": "aerospike", "category": "database"},
    {"source": "vector_db", "category": "database"},
    {"source": "langchain", "category": "framework"},
]

# Add texts to the vector store
document_ids = vector_store.add_texts(
    texts=texts,
    metadatas=metadatas,
)
print(f"Added {len(document_ids)} documents to Aerospike")

# Search for similar documents
query = "Tell me about vector databases"
docs = vector_store.similarity_search(query, k=2)

# Print the results
print("\nSearch Results:")
for i, doc in enumerate(docs):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

# Example of searching with scores
docs_and_scores = vector_store.similarity_search_with_score(query, k=2)
print("\nSearch Results with Scores:")
for i, (doc, score) in enumerate(docs_and_scores):
    print(f"Result {i+1} (Score: {score}):")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()

# Example of deleting documents
if document_ids:
    # Delete the first document
    vector_store.delete(ids=[document_ids[0]])
    print(f"Deleted document with ID: {document_ids[0]}") 