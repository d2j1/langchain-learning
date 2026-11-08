from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

result = llm.invoke("Write a poem about the ocean.")

print(result.content[0].text)



