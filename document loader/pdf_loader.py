from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

file_path = Path(__file__).parent / "cricket.pdf"
loader = PyPDFLoader(str(file_path))

docs = loader.load()
print(docs[0].page_content)