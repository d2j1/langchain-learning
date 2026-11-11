from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

file_path = Path(__file__).parent / "cricket.txt"

loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", max_retries=3)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary of the following text:\n\n{text}",
    input_variables=["text"],
)

chain = prompt | chat_model | parser

# docs is a list of Document objects; page_content holds the file's text.
result = chain.invoke({"text": docs[0].page_content})

print(result)
