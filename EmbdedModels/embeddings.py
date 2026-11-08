from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

vector = embeddings.embed_query("Delhi is the capital of India")

print(vector)
