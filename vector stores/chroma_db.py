from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# 1. Embedding model (same one used in EmbdedModels/)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

# 2. Sample documents with metadata
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful batsmen in IPL history.",
        metadata={"team": "Royal Challengers Bangalore"},
    ),
    Document(
        page_content="Rohit Sharma has led Mumbai Indians to five IPL titles.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="MS Dhoni is known for his calm captaincy and finishing skills.",
        metadata={"team": "Chennai Super Kings"},
    ),
    Document(
        page_content="Jasprit Bumrah is famous for his yorkers and death-over bowling.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="Ravindra Jadeja is an all-rounder who contributes with bat, ball and fielding.",
        metadata={"team": "Chennai Super Kings"},
    ),
]

# 3. Create a persistent Chroma vector store
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db",  # saved to disk in this folder
    collection_name="cricket",
)

# 4. Add documents (embeddings are generated automatically)
ids = vector_store.add_documents(docs)
print("Added IDs:", ids)

# 5. View what's stored
print(vector_store.get(include=["embeddings", "documents", "metadatas"]))

# 6. Similarity search
results = vector_store.similarity_search(query="Who is a good bowler?", k=2)
for r in results:
    print(r.page_content, r.metadata)

# 7. Similarity search with scores (lower score = more similar)
results = vector_store.similarity_search_with_score(query="Who is a good bowler?", k=2)
for doc, score in results:
    print(score, doc.page_content)

# 8. Filter by metadata
results = vector_store.similarity_search_with_score(
    query="Who is a good player?", k=5, filter={"team": "Chennai Super Kings"}
)
for doc, score in results:
    print(doc.page_content, doc.metadata)

# 9. Update a document
updated_doc = Document(
    page_content="Virat Kohli, former RCB captain, is the highest run-scorer in IPL history.",
    metadata={"team": "Royal Challengers Bangalore"},
)
vector_store.update_document(document_id=ids[0], document=updated_doc)

# 10. Delete a document
vector_store.delete(ids=[ids[0]])
print(vector_store.get())
