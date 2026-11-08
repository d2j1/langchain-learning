from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

primary_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash",temperature=0.7, max_output_tokens=10, max_retries=3)

chat_model = primary_model
chat_response = chat_model.invoke("Write a poem about the ocean.")

print(chat_response.text)
