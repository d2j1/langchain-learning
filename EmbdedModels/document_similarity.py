from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

documents =["Delhi is the capital of India",
 "Mumbai is the financial capital of India", 
 "Kolkata is the cultural capital of India"]


vector = embeddings.embed_documents(documents)
query = "tell me about kolkata"
query_vector = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_vector], vector)[0]
print(list(enumerate(similarity_scores)))
