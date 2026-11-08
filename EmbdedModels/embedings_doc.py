from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

documents =["Delhi is the capital of India", "Mumbai is the financial capital of India", "Kolkata is the cultural capital of India"]
vector = embeddings.embed_documents(documents)

print(vector)
