from pathlib import Path

from dotenv import load_dotenv
from langchain_core.load import loads
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

TEMPLATE_PATH = Path(__file__).parent / "template.json"

# Load the template saved by prompt_generator.py.
prompt_template = loads(TEMPLATE_PATH.read_text(encoding="utf-8"))

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)

# Fill in the template, then send the finished prompt to the chat model.
prompt = prompt_template.invoke(
    {"topic": "embeddings", "num_lines": 3, "audience": "beginner"}
)
response = chat_model.invoke(prompt)

chain = prompt_template | chat_model
response = chain.invoke({"topic": "embeddings", "num_lines": 3, "audience": "beginner"})


print(response.text)
